from abc import ABC
import scipy
import scipy.stats
import scipy.integrate
import numpy as np
from nn_approach.model import FeedForward

from typing import List, Tuple
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from config import Config as cfg
import torch


class Estimator(ABC):
    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs):
        pass

    def estimate(self, *args, **kwargs):
        pass


class KDE_Estimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.baseline_fit = None
        self.normal_fit = None
        self.edge_fit = None

    def fit_baseline(self, data):
        self.baseline_fit = scipy.stats.gaussian_kde(data.T)

    def fit_normal(self, data: np.array):
        self.normal_fit = scipy.stats.gaussian_kde(data.T)

    def fit_edge(self, data: np.array):
        self.edge_fit = scipy.stats.gaussian_kde(data.T)

    def baseline_estimate(self, x):
        # return scipy.integrate.quad(lambda y: self.baseline_fit([x, y]), -np.inf, np.inf)[0]
        return self.baseline_fit(x)

    def improved_estimate(self, x, c, p_normal, p_edge):
        # integral_till_c = scipy.integrate.quad(lambda y: self.normal_fit([x, y]) + self.normal_fit([x, 2 * c - y] if y < c else 0), -np.inf, c)[0]
        # integral_from_c = scipy.integrate.quad(lambda y: self.edge_fit([x, y]) + self.edge_fit([x, 2 * c - y] if y >= c else 0), c, np.inf)[0]
        # return p_normal * integral_till_c + p_edge * integral_from_c
        return p_normal * self.normal_fit(x) + p_edge * self.edge_fit(x)

    def estimate(self, x_values, estimate_fn, *args, **kwargs):
        estimate = np.empty_like(x_values)

        for i in range(len(x_values)):
            estimate[i] = estimate_fn(x_values[i], **kwargs)
        return estimate


class NN_Estimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = FeedForward(
            1, 1, cfg.nn_num_hidden_nodes, cfg.nn_num_hidden_layers)

    def fit(self, train_loader, val_loader, device, pattern):
        # Initialize checkpointer
        ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
        checkpointer = ModelCheckpoint(
            save_top_k=1,
            every_n_train_steps=500,
            monitor="val_mse",
            filename=pattern + ".best",
            save_last=True,
            auto_insert_metric_name=False,
        )

        # Fit the model
        trainer = pl.Trainer(max_steps=cfg.nn_training_steps,
                             inference_mode=False,
                             callbacks=[checkpointer],
                             accelerator=device)

        trainer.fit(self.model, train_loader, val_loader)

        return self.model

    def estimate(self, x_values):
        return self.model.compute_pdf(torch.Tensor(x_values))
