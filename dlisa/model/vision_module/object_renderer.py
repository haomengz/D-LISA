import torch
import torch.nn as nn
import lightning.pytorch as pl
from dlisa.common_ops.functions import common_ops
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

import os
from PIL import Image


class CameraPoseGenerator(pl.LightningModule):
    def __init__(self, number_of_views, input_channel, hidden_size):
        super().__init__()
        self.number_of_views = number_of_views
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, number_of_views * 3)
        )
    
    def forward(self, feature):
        pose_offset = self.pose_encoder(feature.float())
        pose_offset = pose_offset.reshape(self.number_of_views, 3)
        return pose_offset


class ObjectRenderer(pl.LightningModule):
    def __init__(self, eye, rasterizer_setting, camera_pose_generator):
        super().__init__()
        self.eye = eye
        self.R, self.T = look_at_view_transform(eye=eye, at=((0, 0, 0),), up=((0, 0, 1),), device=self.device)
        self.image_size = rasterizer_setting.image_size
        self.views = len(eye)
        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=None, raster_settings=PointsRasterizationSettings(**rasterizer_setting)
            ), compositor=AlphaCompositor()
        )
        for param in self.parameters():
            param.requires_grad = False

        self.camera_pose_generator = CameraPoseGenerator(number_of_views=len(eye), **camera_pose_generator)
        for param in self.camera_pose_generator.parameters():
            param.requires_grad = True

    def forward(self, data_dict, output_dict):
        batch_size = len(data_dict["scene_id"]) # 4
        total_num_aabbs = output_dict["pred_aabb_min_max_bounds"].shape[0] # 172
        output_imgs = torch.zeros(
            size=(total_num_aabbs * self.views, self.image_size, self.image_size, 3),
            dtype=torch.float32, device=self.device
        )

        for i in range(batch_size):
            batch_points_start_idx = data_dict["all_point_count_offsets"][i]
            batch_points_end_idx = data_dict["all_point_count_offsets"][i + 1]
            current_pcd_xyz = data_dict["all_point_xyz"][batch_points_start_idx:batch_points_end_idx] 
            current_pcd_rgb = data_dict["all_point_rgb"][batch_points_start_idx:batch_points_end_idx]

            pred_aabb_start_idx = output_dict["proposal_batch_offsets"][i]
            pred_aabb_end_idx = output_dict["proposal_batch_offsets"][i + 1]

            output_masks = common_ops.crop_pcd_from_aabbs(
                output_dict["pred_aabb_min_max_bounds"][pred_aabb_start_idx:pred_aabb_end_idx],
                current_pcd_xyz
            )
            aabb_xyz_list = []
            aabb_rgb_list = []
            for obj_i in range(output_masks.shape[0]):
                current_obj_point_indicies = output_masks[obj_i]
                if not current_obj_point_indicies.any():
                    obj_pcd_xyz = torch.empty(size=(0, 3), device=self.device, dtype=torch.float32)
                    obj_pcd_rgb = torch.empty(size=(0, 3), device=self.device, dtype=torch.float32)
                else:
                    obj_pcd_xyz = current_pcd_xyz[current_obj_point_indicies]
                    obj_pcd_rgb = current_pcd_rgb[current_obj_point_indicies]
                    obj_pcd_xyz -= obj_pcd_xyz.mean(dim=0)
                    obj_pcd_xyz /= obj_pcd_xyz.abs().max()
                for _ in range(self.views):
                    aabb_xyz_list.append(obj_pcd_xyz)
                    aabb_rgb_list.append(obj_pcd_rgb)
            pytorch3d_pcd = Pointclouds(points=aabb_xyz_list, features=aabb_rgb_list)
            pytorch3d_pcd.device = self.device
            num_aabbs = len(aabb_xyz_list) // self.views

            aabbs = output_dict["pred_aabb_min_max_bounds"]
            avg_box_size = (aabbs[:, 1, :] - aabbs[:, 0, :]).mean(dim=0)
            pose_offset = self.camera_pose_generator(avg_box_size)

            R_, T_ = look_at_view_transform(eye= torch.tensor(self.eye, device=self.device) + pose_offset, at=((0, 0, 0),), up=((0, 0, 1),), device=self.device)
            R = R_.expand(num_aabbs, -1, -1, -1).flatten(0, 1)
            T = T_.expand(num_aabbs, -1, -1).flatten(0, 1)

            temp_imgs = self.renderer(
                pytorch3d_pcd, dtype=torch.float32, device=self.device,
                cameras=FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01)
            )

            output_imgs[pred_aabb_start_idx * self.views:pred_aabb_end_idx * self.views] = temp_imgs

        return output_imgs