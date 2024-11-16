import os
import torch
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from dlisa.data.data_module import DataModule
from dlisa.callback.gpu_cache_clean_callback import GPUCacheCleanCallback
from dlisa.callback.lr_decay_callback import LrDecayCallback
from dlisa.callback.stop_training_callback import StopTrainingCallback
from dlisa.callback.set_seed_callback import SetSeedCallback
from dlisa.callback.save_thres_callback import SaveThresCallback


def init_callbacks(cfg):
    set_seed_callback = SetSeedCallback()
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_decay_callback = LrDecayCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    stop_training_callback = StopTrainingCallback()
    save_thres_callback = SaveThresCallback()
    callbacks = [set_seed_callback, checkpoint_monitor, gpu_cache_clean_monitor, lr_decay_callback, lr_monitor]
    if cfg.scheduled_job:
        callbacks.append(stop_training_callback)
    callbacks.append(save_thres_callback)
    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    data_module = DataModule(cfg.data)

    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # load the pre-trained detector
    if "detector_path" in cfg:
        detector_weights = torch.load(cfg.detector_path)["state_dict"]
        model.detector.load_state_dict(detector_weights, strict=False)

    # initialize logger
    if cfg.resume:
        with open(os.path.join(cfg.experiment_output_path, "training", "wandb_run_id.txt"), "r") as file:
            saved_run_id = file.readline().strip()
        cfg.logger.id = saved_run_id
        cfg.logger.resume = 'must'
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = hydra.utils.instantiate(cfg.logger)
        with open(os.path.join(cfg.experiment_output_path, "training", "wandb_run_id.txt"), "w") as file:
            file.write(logger.experiment.id)

    # initialize callbacks
    callbacks = init_callbacks(cfg)

    # initialize trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    # check the checkpoint
    if cfg.scheduled_job and cfg.resume:
        cfg.ckpt_path = os.path.join(cfg.experiment_output_path, "training", "last.ckpt")
        
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()