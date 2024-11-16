import os
import json
import clip
import torch
import hydra
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl

from dlisa.common_ops.functions import common_ops
from dlisa.model.vision_module.pointgroup import PointGroupNMS
from dlisa.model.cross_modal_module.match_module import MatchModule
from dlisa.model.vision_module.object_renderer import ObjectRenderer
from dlisa.model.vision_module.clip_image_encoder import CLIPImageEncoder


class DLISA(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = cfg.data.lang_dataset
        self.stop_training=False
        self.optimal_inference_thres = 0
        self.inference_thres_list = cfg.model.inference.output_threshold_list

        # clip model & text encoder
        self.clip_model = clip.load(cfg.model.network.clip_model, device=self.device)[0]
        self.text_encoder = hydra.utils.instantiate(cfg.model.network.clip_word_encoder, clip_model=self.clip_model)

        # detector
        self.detector_input_channel = 3 + 3 * cfg.data.point_features.use_rgb + 3 * cfg.data.point_features.use_normal + \
                128 * cfg.data.point_features.use_multiview
        self.detector = PointGroupNMS(
        input_channel=self.detector_input_channel, output_channel=cfg.model.network.detector.output_channel,
        max_proposals=cfg.model.network.max_num_proposals, semantic_class=cfg.data.semantic_class,
        use_gt=cfg.model.network.detector.use_gt_proposal, **cfg.model.network.dynamic_box_module
        )
      
        # renderer
        if self.hparams.cfg.model.network.use_2d_feature:
            self.object_renderer = ObjectRenderer(camera_pose_generator=cfg.model.network.camera_pose_generator, **cfg.model.network.object_renderer)
            self.clip_image = CLIPImageEncoder(clip_model=self.clip_model, **cfg.model.network.clip_img_encoder)
    
        # matching module
        match_feature_dim = cfg.model.network.detector.output_channel * self.hparams.cfg.model.network.use_3d_features + \
            self.hparams.cfg.model.network.use_2d_feature * self.hparams.cfg.model.network.clip_img_encoder.output_channel 

        self.match_module = MatchModule(
            **cfg.model.network.matching_module,
            input_channel=match_feature_dim 
        )

        # loss
        self.dynamic_loss = hydra.utils.instantiate(cfg.model.loss.dynamic_loss)

        if self.dataset_name in ("ScanRefer", "Nr3D"):
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_ce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        elif self.dataset_name == "Multi3DRefer":
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_bce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        else:
            raise NotImplementedError

        self.contrastive_loss = hydra.utils.instantiate(cfg.model.loss.contrastive_loss)

        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # evaluator
        self.evaluator = hydra.utils.instantiate(cfg.data.evaluator)
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.count_parameters()

    def count_parameters(self):
            print("Number of parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, data_dict):
        output_dict = self.detector(data_dict)
        batch_size = len(data_dict["scene_id"])
        if self.hparams.cfg.model.network.use_3d_features:
            aabb_features = output_dict["aabb_features"]
        else:
            aabb_features = torch.empty(
                size=(output_dict["aabb_features"].shape[0], 0),
                dtype=output_dict["aabb_features"].dtype, device=self.device
            )
        
        self.text_encoder(data_dict, output_dict)
        data_dict["lang_attention_mask"] = None

        if self.hparams.cfg.model.network.use_2d_feature:
            rendered_imgs = self.object_renderer(data_dict, output_dict)
            img_features = self.clip_image(rendered_imgs.permute(dims=(0, 3, 1, 2)))
            views = len(self.hparams.cfg.model.network.object_renderer.eye)
            aabb_img_features = torch.nn.functional.avg_pool1d(
                img_features.permute(1, 0), kernel_size=views, stride=views
            ).permute(1, 0)

            aabb_features = torch.nn.functional.normalize(torch.cat((aabb_features, aabb_img_features), dim=1), dim=1)

        output_dict["aabb_features"] = common_ops.convert_sparse_tensor_to_dense(
            aabb_features, output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        )

        output_dict["pred_aabb_min_max_bounds"] = common_ops.convert_sparse_tensor_to_dense(
            output_dict["pred_aabb_min_max_bounds"].reshape(-1, 6), output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        ).reshape(batch_size, self.hparams.cfg.model.network.max_num_proposals, 2, 3)

        self.match_module(data_dict, output_dict)
        return output_dict
    
    def _loss(self, data_dict, output_dict):
        # detector loss
        loss_dict = self.detector.loss(data_dict, output_dict)

        # reference loss
        loss_dict["reference_loss"] = self.ref_loss(
            output_dict,
            output_dict["pred_aabb_min_max_bounds"],
            output_dict["pred_aabb_scores"],
            data_dict["gt_aabb_min_max_bounds"],
            data_dict["gt_target_obj_id_mask"].permute(dims=(1, 0)),
            data_dict["aabb_count_offsets"],
        )

        # contrastive loss
        if self.hparams.cfg.model.network.use_contrastive_loss:
            loss_dict["contrastive_loss"] = self.contrastive_loss(
                output_dict["aabb_features_inter"],
                output_dict["sentence_features"],
                output_dict["gt_labels"]
            )
        
        # dynamic_box regularization
        loss_dict["dynamic_loss"] = self.dynamic_loss(
            output_dict["thres_score"]
        )
        return loss_dict

    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)

        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)

        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        return total_loss
    
    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)

        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)
        self.val_step_outputs.append((self._parse_pred_results_val(data_dict, output_dict), self._parse_gt(data_dict)))

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        self.test_step_outputs.append(
            (self._parse_pred_results_test(data_dict, output_dict), self._parse_gt(data_dict))
        )

    def on_validation_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_step_outputs:
            total_gt_results.update(gt_results)
            for thres, info in pred_results.items():
                if thres not in total_pred_results.keys():
                    total_pred_results[thres] = dict()
                total_pred_results[thres].update(info)

        self.val_step_outputs.clear()
        self.evaluator.set_ground_truths(total_gt_results)
        results, thres = self.evaluator.evaluate(total_pred_results)
        self.optimal_inference_thres = thres

        for metric_name, result in results.items():
            for breakdown, value in result.items():
                self.log(f"val_eval/{metric_name}_{breakdown}", value)
        self.log("val_eval/optimal_thres", thres)

        if self.hparams.cfg.scheduled_job:
            self.stop_training = True

    def on_test_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.test_step_outputs:
            total_gt_results.update(gt_results)
            total_pred_results.update(pred_results)
        self.test_step_outputs.clear()
        self._save_predictions(total_pred_results)

    def _parse_pred_results_val(self, data_dict, output_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        if self.dataset_name in ("ScanRefer", "Nr3D"):
            pred_aabb_score_masks = (output_dict["pred_aabb_scores"].argmax(dim=1)).reshape(
                shape=(batch_size, lang_chunk_size, -1)
            )
        elif self.dataset_name == "Multi3DRefer":
            mask_dict = dict()
            for thres in self.inference_thres_list:
                mask_dict[thres] = (
                    torch.sigmoid(output_dict["pred_aabb_scores"]) >= thres
                ).reshape(shape=(batch_size, lang_chunk_size, -1))
        else:
            raise NotImplementedError

        pred_results = {}

        for thres, pred_aabb_score_masks in mask_dict.items():
            pred_results[thres]= dict()
            pred_aabb_score_masks_numpy = pred_aabb_score_masks.cpu().numpy()
            pred_aabb_bounds_numpy = output_dict["pred_aabb_min_max_bounds"].cpu().numpy()
            pred_aabb_score_numpy = output_dict["pred_aabb_scores"].reshape(batch_size, lang_chunk_size, -1).cpu().numpy()

            for i in range(batch_size):
                for j in range(lang_chunk_size):
                    pred_aabbs = output_dict["pred_aabb_min_max_bounds"][i][pred_aabb_score_masks[i, j]]
                    aabb_bounds = (pred_aabbs + data_dict["scene_center_xyz"][i]).cpu().numpy()
                    pred_results[thres][
                        (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                        data_dict["ann_id"][i][j].item())
                    ] = {
                        "aabb_bound": (aabb_bounds)
                    }
                    
        return pred_results
    
    def _parse_pred_results_test(self, data_dict, output_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        if self.dataset_name in ("ScanRefer", "Nr3D"):
            pred_aabb_score_masks = (output_dict["pred_aabb_scores"].argmax(dim=1)).reshape(
                shape=(batch_size, lang_chunk_size, -1)
            )
        elif self.dataset_name == "Multi3DRefer":
            pred_aabb_score_masks = (
                    torch.sigmoid(output_dict["pred_aabb_scores"]) >= self.optimal_inference_thres
            ).reshape(shape=(batch_size, lang_chunk_size, -1))
        else:
            raise NotImplementedError

        pred_results = {}
        pred_aabb_score_masks_numpy = pred_aabb_score_masks.cpu().numpy()
        pred_aabb_bounds_numpy = output_dict["pred_aabb_min_max_bounds"].cpu().numpy()
        pred_aabb_score_numpy = output_dict["pred_aabb_scores"].reshape(batch_size, lang_chunk_size, -1).cpu().numpy()

        for i in range(batch_size):
            for j in range(lang_chunk_size):
                pred_aabbs = output_dict["pred_aabb_min_max_bounds"][i][pred_aabb_score_masks[i, j]]
                aabb_bounds = (pred_aabbs + data_dict["scene_center_xyz"][i]).cpu().numpy()
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                    data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": (aabb_bounds)
                }
        return pred_results
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())
        return optimizer

    def _parse_gt(self, data_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        gts = {}
        gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)
        for i in range(batch_size):
            aabb_start_idx = data_dict["aabb_count_offsets"][i]
            aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
            for j in range(lang_chunk_size):
                gts[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound":
                        (data_dict["gt_aabb_min_max_bounds"][aabb_start_idx:aabb_end_idx][gt_target_obj_id_masks[j]
                    [aabb_start_idx:aabb_end_idx]] + data_dict["scene_center_xyz"][i]).cpu().numpy(),
                    "eval_type": data_dict["eval_type"][i][j]
                }
        return gts

    def _save_predictions(self, predictions):
        scene_pred = {}
        for key, value in predictions.items():
            scene_id = key[0]
            if key[0] not in scene_pred:
                scene_pred[scene_id] = []
            corners = np.empty(shape=(value["aabb_bound"].shape[0], 8, 3), dtype=np.float32)
            for i, aabb in enumerate(value["aabb_bound"]):
                min_point = aabb[0]
                max_point = aabb[1]
                corners[i] = np.array([
                    [x, y, z]
                    for x in [min_point[0], max_point[0]]
                    for y in [min_point[1], max_point[1]]
                    for z in [min_point[2], max_point[2]]
                ], dtype=np.float32)

            if self.dataset_name in ("ScanRefer", "Nr3D"):
                scene_pred[scene_id].append({
                    "object_id": key[1],
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
            elif self.dataset_name == "Multi3DRefer":
                scene_pred[scene_id].append({
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
        prediction_output_root_path = os.path.join(
            self.hparams.cfg.pred_path, self.hparams.cfg.data.inference.split
        )
        os.makedirs(prediction_output_root_path, exist_ok=True)
        for scene_id in tqdm(scene_pred.keys(), desc="Saving predictions"):
            with open(os.path.join(prediction_output_root_path, f"{scene_id}.json"), "w") as f:
                json.dump(scene_pred[scene_id], f, indent=2)
        self.print(f"==> Complete. Saved at: {os.path.abspath(prediction_output_root_path)}")