import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch
from config import UVParameters as uv_params
from torchmetrics import MeanSquaredError
from typing import Tuple, List, Any

from torch.distributions import MultivariateNormal, Normal


class FeedForward(pl.LightningModule):
    def __init__(self, num_in, num_out, num_hidden, num_layers=0) -> None:
        super().__init__()
        self.model = self._construct_net(
            num_in, num_out, num_hidden, num_layers)
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.norm = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def _construct_net(self, num_in, num_out, num_hidden, num_layers=0):
        args = []
        for _ in range(num_layers):
            args.extend([nn.Linear(num_hidden, num_hidden), nn.Sigmoid()])
        return nn.Sequential(nn.Linear(num_in, num_hidden), nn.Sigmoid(), *args, nn.Linear(num_hidden, num_out))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        x = torch.reshape(x, [-1, 1]).type(torch.float32)
        y = torch.reshape(y, [-1, 1]).type(torch.float32)
        return self.model(x), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        self.log('train_se', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.forward(batch)
        self.val_mse(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        self.log('val_mse', self.val_mse)

    def test_step(self,  batch, batch_idx):
        y_hat, y = self.forward(batch)
        self.test_mse(y_hat, y)

    def on_test_epoch_end(self) -> None:
        self.log('test_mse', self.test_mse)

    def compute_pdf(self, points):
        pdf = torch.zeros_like(points)
        points = torch.reshape(points, (-1, 1))
        for i in range(points.shape[0]):
            x = points[i]
            x.requires_grad = True
            y_hat = self.model.forward(x)
            y_hat.backward()

            pdf[i] = x.grad
            x.grad.zero_()

        return pdf

    def compute_cdf(self, points):
        cdf = torch.zeros_like(points)
        points = torch.reshape(points, (-1, 1))
        with torch.no_grad():
            for i in range(points.shape[0]):
                x = points[i]
                y_hat = self.model.forward(x)
                cdf[i] = y_hat
        return cdf

    def configure_optimizers(self) -> None:
        # setup the optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=uv_params.nn_lr)
        schedule = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=uv_params.nn_training_steps,
                eta_min=0),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return [optimizer], [schedule]
