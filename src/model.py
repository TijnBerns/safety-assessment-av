import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch
from config import Config as cfg
from torchmetrics import MeanSquaredError

from typing import Tuple, List

class FeedForward(pl.LightningModule):
    def __init__(self, num_in, num_out, num_hidden, num_layers=0) -> None:
        super().__init__()
        self.model = self._construct_net(num_in, num_out, num_hidden, num_layers)
        self.val_mse = MeanSquaredError()

    def _construct_net(self, num_in, num_out, num_hidden, num_layers=0):
        args = []
        for _ in range(num_layers - 1):
            args.extend([nn.Linear(num_hidden, num_hidden), nn.ReLU()])
        return nn.Sequential(nn.Linear(num_in, num_hidden), nn.ReLU(), *args, nn.Linear(num_hidden, num_out))

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.reshape(x, (-1, 1))
        y = torch.reshape(y, [-1,1])
       
        
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.reshape(x, (-1, 1))
        y = torch.reshape(y, [-1,1])
        
        y_hat = self.forward(x)
        self.val_mse(y_hat, y)
        return 
    
    def validation_epoch_end(self, outputs) -> None:
        self.log('val_mse', self.val_mse)
        return super().validation_epoch_end(outputs)


    def configure_optimizers(self) -> None:
        # setup the optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        schedule = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.max_steps,
                eta_min=0),
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return [optimizer], [schedule]

    
