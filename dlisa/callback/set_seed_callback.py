from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl

class SetSeedCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        pl.seed_everything(trainer.current_epoch, workers=True)