from lightning.pytorch.callbacks import Callback

class StopTrainingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.stop_training:
            pl_module.stop_training = False
            trainer.should_stop = True