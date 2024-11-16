from lightning.pytorch.callbacks import Callback

class SaveThresCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['opt_thres'] = pl_module.optimal_inference_thres