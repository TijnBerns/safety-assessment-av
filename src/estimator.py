import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import pandas as pd
from utils import save_csv
from itertools import product
import data_utils
from torch.utils.data import DataLoader
import torch
from config import Config as cfg
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from nn_approach.model import FeedForward
import numpy as np
import scipy.integrate
import scipy.stats
import scipy
from abc import ABC


class Estimator(ABC):
    def fit(self, *args, **kwargs):
        pass

    def estimate(self, *args, **kwargs):
        pass


class KDEEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model: scipy.stats.gaussian_kde = None

    def fit(self, data: np.ndarray):
        self.model = scipy.stats.gaussian_kde(data)

    def estimate(self, x_values):
        return self.model(x_values)


class NNEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = FeedForward(
            1, 1, cfg.nn_num_hidden_nodes, cfg.nn_num_hidden_layers)

    def _construct_train_loader(self, data):
        samples = data_utils.annotate_data(data)
        loader = DataLoader(samples, shuffle=True,
                            batch_size=cfg.nn_batch_size, drop_last=True)
        return loader

    def _construct_validation_loader(self, single_distribution, x_values):
        targets = single_distribution.cdf(x_values)
        samples = list(zip(x_values, targets))
        loader = DataLoader(samples, shuffle=False, batch_size=len(x_values))
        return loader

    def fit(self, data, single_distribution, x_values, pattern, device):
        train_loader = self._construct_train_loader(data)
        val_loader = self._construct_validation_loader(
            single_distribution, x_values)

        # Initialize checkpointer
        ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
        checkpointer = ModelCheckpoint(
            save_top_k=1,
            every_n_train_steps=100,
            monitor="val_mse",
            filename=pattern + ".best",
            save_last=True,
            auto_insert_metric_name=False,
        )

        # Fit the model
        trainer = pl.Trainer(max_steps=cfg.nn_training_steps,
                             inference_mode=False,
                             callbacks=[checkpointer],
                             #  logger=False,
                             log_every_n_steps=500,
                             accelerator=device)

        trainer.fit(self.model, train_loader, val_loader)

        return self.model

    def estimate(self, x_values):
        return self.model.compute_pdf(torch.Tensor(x_values))


class DefaultPipeline(ABC):
    def run_pipeline(self, estimator, root, *args, **kwargs):
        parameters = list(product(
            cfg.distributions,
            cfg.p_event,
            cfg.num_normal,
            cfg.num_event,
            cfg.correlation
        ))

        for distribution_str, p_event, num_normal, num_event, correlation in parameters:
            single_distribution, mv_distribution = cfg.get_distributions(
                distribution_str, correlation)
            x_values = data_utils.get_evaluation_interval(
                single_distribution, cfg.num_eval)
            true = single_distribution.pdf(x_values)
            threshold = scipy.stats.norm.ppf(1 - p_event)

            # Initialize dicts to store results
            baseline_estimates = {"x": x_values, "true": true}
            improved_estimates = {"x": x_values, "true": true}

            for run in tqdm(range(cfg.num_estimates), desc=f'{distribution_str}: norm={num_normal} edge={num_event} p_edge={p_event}'):
                # Generate data
                normal_data, edge_data = data_utils.generate_data(
                    mv_distribution, num_normal, num_event, threshold, random_state=run)

                # Obtain estimates
                baseline, improved = self.obtain_estimates(
                    estimator, single_distribution, normal_data, edge_data, threshold, x_values, distribution_str, *args, **kwargs)
                baseline_estimates[f'run_{run}'] = baseline
                improved_estimates[f'run_{run}'] = improved

                # Store results
                parent = root / distribution_str / \
                    f'p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}'
                save_csv(path=parent / f'p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.baseline.csv',
                         df=pd.DataFrame(baseline_estimates))
                save_csv(path=parent / f'p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.improved.csv',
                         df=pd.DataFrame(improved_estimates))

        def obtain_estimates(self, *args, **kwargs):
            raise NotImplementedError


class CombinedDataPipeline(DefaultPipeline):
    def obtain_estimates(estimator: Estimator, single_distribution, normal_data, edge_data, threshold, x_values, distribution_str, *args, **kwargs):
        # Construct estimators
        baseline_estimator = estimator.__class__()
        combined_estimator = estimator.__class__()

        # Filter edge and normal data
        p_edge_estimate = data_utils.compute_p_edge(normal_data, threshold)
        combined_data = data_utils.combine_data(
            normal_data, edge_data, threshold, p_edge_estimate)

        # Fit data to estimators
        baseline_estimator.fit(
            normal_data[:, 0], x_values=x_values, single_distribution=single_distribution, **kwargs)
        combined_estimator.fit(
            combined_data[:, 0], x_values=x_values, single_distribution=single_distribution, **kwargs)

        # Obtain estimates
        return baseline_estimator.estimate(x_values), combined_estimator.estimate(x_values)


class NaiveEnsemblePipeline(DefaultPipeline):
    def obtain_estimates(self, estimator: Estimator, single_distribution, normal_data, edge_data, threshold, x_values, *args, **kwargs):
        normal_data_filtered, edge_data_filtered = data_utils.filter_data(
            normal_data, edge_data, threshold)

        # Construct estimators
        baseline_estimator = estimator.__class__()
        normal_estimator = estimator.__class__()
        edge_estimator = estimator.__class__()

        # Filter edge and normal data
        p_edge_estimate = data_utils.compute_p_edge(normal_data, threshold)
        p_normal_estimate = 1 - p_edge_estimate

        # Fit data to estimators
        baseline_estimator.fit(normal_data[:, 0], **kwargs)
        normal_estimator.fit(normal_data_filtered[:, 0], **kwargs)
        edge_estimator.fit(edge_data_filtered[:, 0], **kwargs)

        # Obtain estimates
        return baseline_estimator.estimate(x_values), p_normal_estimate * normal_estimator.estimate(x_values) + p_edge_estimate * edge_estimator.estimate(x_values)
