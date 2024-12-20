import os
import hydra
import lightning.pytorch as pl
from dlisa.data.data_module import DataModule
import torch


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.test_seed, workers=True)

    # create directories for inference outputs
    os.makedirs(os.path.join(cfg.pred_path, cfg.data.inference.split), exist_ok=True)

    # initialize data
    data_module = DataModule(cfg.data)

    # initialize model
    cfg.data.evaluator.verbose = True  # print out evaluation results after inference
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # initialize trainer
    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1, max_epochs=1, logger=False)

    # check the checkpoint
    assert cfg.ckpt_path is not None, "Error: Checkpoint path is not provided."
    assert os.path.exists(cfg.ckpt_path), f"Error: Checkpoint path {cfg.ckpt_path} does not exist."

    ckpt = torch.load(cfg.ckpt_path)
    ckpt_weights = ckpt["state_dict"]
    model.load_state_dict(ckpt_weights)
    model.optimal_inference_thres = ckpt.get('opt_thres', 0)

    # start inference
    trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()