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
from torch.utils.data import DataLoader
import data_utils
import json 
from itertools import product
from pathlib import Path
from utils import save_csv
import pandas as pd

class Estimator(ABC):
    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs):
        pass

    def estimate(self, *args, **kwargs):
        pass

def combined_estimation_pipline(baseline_estimator: Estimator, combined_estimator: Estimator, root: Path, *args, **kwargs):
    """Default estimation pipeline. Takes two estimators, then fits normal data one estimator and combined data on the other estimator.

    Args:
        baseline_estimator (Estimator): Estimator on which normal data is fitted
        combined_estimator (Estimator): Estimator on which combined data is fitted
        root (Path): Root folder to which results will be saved.
    """
    # Define evaluation interval
    x_values = cfg.evaluation_interval

    for distribution_str, distribution in cfg.distributions.items():
        true = cfg.single_distributions[distribution_str].pdf(x_values)

        for p_edge, num_normal, num_edge in product(cfg.p_edge, cfg.num_normal, cfg.num_edge):
            thresholds = {}

            # Initialize dicts to store results
            baseline_estimates = {"x": x_values, "true": true}
            improved_estimates = {"x": x_values, "true": true}

            for run in tqdm(range(cfg.num_estimates), desc=f'{distribution_str}: norm={num_normal} edge={num_edge} p_edge={p_edge}'):
                # Generate data
                normal_data, edge_data, threshold = data_utils.generate_data(
                    distribution, p_edge, num_normal, num_edge)

                # Filter edge and normal data
                p_edge_estimate = data_utils.compute_p_edge(normal_data, threshold)
                combined_data = data_utils.combine_data(normal_data, edge_data, threshold, p_edge_estimate)
                
                # Fit data to estimators
                baseline_estimator.fit(normal_data[:, 0], **kwargs)
                combined_estimator.fit(combined_data[:, 0], **kwargs)

                # Obtain estimates
                baseline_estimates[f'run_{run}'] = baseline_estimator.estimate(x_values)
                improved_estimates[f'run_{run}'] = combined_estimator.estimate(x_values)

                # Store results
                parent = root / distribution_str / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}'
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.baseline.csv',
                         df=pd.DataFrame(baseline_estimates))
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.improved.csv',
                         df=pd.DataFrame(improved_estimates))

                thresholds[f"run_{run}"] = threshold
                with open(parent / 'thresholds.json', 'w') as f:
                    json.dump(thresholds, f, indent=2)


class KDE_Estimator_a(Estimator):
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
    
class KDE_Estimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
    def fit(self, data: np.ndarray):
        self.model = scipy.stats.gaussian_kde(data.T)
        
    def estimate(self, x_values):
        res = np.empty_like(x_values)
        for i in range(len(x_values)):
            res[i] = self.model(x_values[i])

        return res


class NN_Estimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = FeedForward(
            1, 1, cfg.nn_num_hidden_nodes, cfg.nn_num_hidden_layers)
        
    def _construct_dataloader(self, data):
        # Label training data
        _, bins = np.histogram(data, len(data))
        samples, _ = data_utils.annotate_data(data, bins)
        
        # construct dataloaders
        loader = DataLoader(samples, shuffle=True, batch_size=cfg.nn_batch_size, drop_last=True)
        return loader
    
    def fit(self, data, pattern, device):
        loader = self._construct_dataloader(data)
        
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

        trainer.fit(self.model, loader, loader)

        return self.model

    def estimate(self, x_values):
        return self.model.compute_pdf(torch.Tensor(x_values))
    
    

