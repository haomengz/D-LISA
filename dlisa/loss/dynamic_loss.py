import lightning.pytorch as pl
import torch.nn as nn
import torch


class DynamicLoss(pl.LightningModule):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, scores):
        loss = self.alpha * torch.mean(torch.sigmoid(scores))
        return loss