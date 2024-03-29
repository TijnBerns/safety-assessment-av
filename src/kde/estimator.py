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

import json
from abc import ABC, abstractmethod
import scipy
import scipy.stats
import scipy.integrate
from scipy.optimize import fsolve
import numpy as np
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from src.kde.parameters import UVParameters as uv_params
from src.kde.parameters import MVParameters as mv_params
import torch
from torch.utils.data import DataLoader
import data.data_utils as data_utils
from itertools import product
from utils import save_csv, save_json
import pandas as pd
from typing import Tuple
from pathlib import Path
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt


# Various types of estimators


class Estimator(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        return

    @abstractmethod
    def estimate_pdf(self, *args, **kwargs):
        return


class KDEEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.model: scipy.stats.gaussian_kde = None

    def fit(self, data: np.ndarray):
        self.model = scipy.stats.gaussian_kde(data.T)

    def estimate_pdf(self, x_values):
        return self.model(x_values.T)


# Wrapper classes for the full estimation pipeline
class EstimatorType(ABC):
    def obtain_estimates(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class CombinedData(EstimatorType):
    @staticmethod
    def obtain_estimates(
        estimator: Estimator,
        single_distribution,
        normal_data,
        edge_data,
        threshold,
        x_values,
        distribution_str,
        *args,
        **kwargs,
    ):
        # Construct estimators
        baseline_estimator = estimator.__class__()
        combined_estimator = estimator.__class__()

        # Filter edge and normal data
        p_edge_estimate = data_utils.compute_p_edge(normal_data, threshold)
        combined_data = data_utils.combine_data(
            normal_data, edge_data, threshold, p_edge_estimate
        )

        # Fit data to estimators
        baseline_estimator.fit(normal_data[:, :-1], **kwargs)
        combined_estimator.fit(combined_data[:, :-1], **kwargs)

        # Obtain estimates of pdf
        baseline_pdf = baseline_estimator.estimate_pdf(x_values)
        improved_pdf = combined_estimator.estimate_pdf(x_values)

        # Obtain estimates of epsilon support
        baseline_eps = data_utils.get_epsilon_support_uv(
            baseline_estimator.estimate_pdf, uv_params.epsilon, x_values
        )
        improved_eps = data_utils.get_epsilon_support_uv(
            combined_estimator.estimate_pdf, uv_params.epsilon, x_values
        )

        # Obtain estimates
        return baseline_pdf, improved_pdf, baseline_eps, improved_eps


class NaiveEnsemble(EstimatorType):
    @staticmethod
    def obtain_estimates(
        estimator: Estimator,
        single_distribution,
        normal_data,
        edge_data,
        threshold,
        x_values,
        *args,
        **kwargs,
    ):
        normal_data_filtered, edge_data_filtered = data_utils.filter_data(
            normal_data, edge_data, threshold
        )

        # Construct estimators
        baseline_estimator = estimator.__class__()
        normal_estimator = estimator.__class__()
        edge_estimator = estimator.__class__()

        # Filter edge and normal data
        p_edge_estimate = data_utils.compute_p_edge(normal_data, threshold)
        p_normal_estimate = 1 - p_edge_estimate

        # Fit data to estimators
        # TODO: Determine whether to reduce dims here e.g. X2 in case of 2d data
        normal_data = normal_data[:, 0]
        edge_data = edge_data[:, 0]
        normal_data_filtered = normal_data_filtered[:, 0]
        edge_data_filtered = edge_data_filtered[:, 0]
        baseline_estimator.fit(normal_data, **kwargs)
        normal_estimator.fit(normal_data_filtered, **kwargs)
        edge_estimator.fit(edge_data_filtered, **kwargs)
        improved_estimator = (
            lambda x: p_normal_estimate * normal_estimator.estimate_pdf(x)
            + p_edge_estimate * edge_estimator.estimate_pdf(x)
        )

        # Obtain estimates of pdf
        baseline_pdf = baseline_estimator.estimate_pdf(x_values)
        improved_pdf = improved_estimator(x_values)

        # TODO: Fix multi variate epsilon support
        # Obtain estimates of epsilon support
        # baseline_eps = data_utils.get_epsilon_support_uv(baseline_estimator.estimate_pdf, uv_params.epsilon, x_values)
        # improved_eps = data_utils.get_epsilon_support_uv(improved_estimator, uv_params.epsilon, x_values)
        baseline_eps, improved_eps = 0, 0

        return baseline_pdf, improved_pdf, baseline_eps, improved_eps


class Pipeline(ABC):
    @abstractmethod
    def run_pipeline(self, *args, **kwargs):
        raise NotImplementedError


class UnivariatePipeline(Pipeline):
    def __init__(self, estimator_type: EstimatorType) -> None:
        self.estimator_type = estimator_type

    def run_pipeline(self, estimator, root, *args, **kwargs):
        parameters = list(
            product(
                uv_params.distributions,
                uv_params.p_event,
                uv_params.num_normal,
                uv_params.num_event,
                uv_params.correlation,
            )
        )
        print(f"number of configurations: {len(parameters)}")
        for distribution_str, p_event, num_normal, num_event, correlation in tqdm(
            parameters
        ):
            single_distribution, mv_distribution = uv_params.get_distributions(
                distribution_str, correlation
            )

            x_values = data_utils.get_evaluation_interval(
                single_distribution, uv_params.num_eval
            )
            true_pdf = single_distribution.pdf(x_values)
            true_epsilon_supp = data_utils.get_epsilon_support_uv(
                single_distribution.pdf, uv_params.epsilon, x_values
            )
            threshold = scipy.stats.norm.ppf(1 - p_event)

            # Initialize dicts to store results
            baseline_pdfs = {"x": x_values, "true": true_pdf}
            improved_pdfs = {"x": x_values, "true": true_pdf}
            baseline_epss = {"true": true_epsilon_supp}
            improved_epss = {"true": true_epsilon_supp}

            for run in tqdm(
                range(uv_params.num_estimates),
                desc=f"{distribution_str}: norm={num_normal} edge={num_event} p_edge={p_event} corr={correlation}",
            ):
                # Generate data
                normal_data, edge_data = data_utils.generate_data(
                    mv_distribution, num_normal, num_event, threshold, random_state=run
                )

                # Obtain estimates
                (
                    baseline_pdf,
                    improved_pdf,
                    baseline_eps,
                    improved_eps,
                ) = self.estimator_type.obtain_estimates(
                    estimator,
                    single_distribution,
                    normal_data,
                    edge_data,
                    threshold,
                    x_values,
                    distribution_str,
                    *args,
                    **kwargs,
                )
                baseline_pdfs[f"run_{run}"] = baseline_pdf
                improved_pdfs[f"run_{run}"] = improved_pdf
                baseline_epss[f"run_{run}"] = baseline_eps
                improved_epss[f"run_{run}"] = improved_eps

            # Store results
            path = (
                root
                / distribution_str
                / f"p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.corr_{correlation}/p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.corr_{correlation}"
            )
            save_csv(
                path=Path(str(path) + ".baseline_pdf.csv"),
                df=pd.DataFrame(baseline_pdfs),
            )
            save_csv(
                path=Path(str(path) + ".improved_pdf.csv"),
                df=pd.DataFrame(improved_pdfs),
            )
            save_json(path=Path(str(path) + ".baseline_eps.json"), data=baseline_epss)
            save_json(path=Path(str(path) + ".improved_eps.json"), data=improved_epss)


class MultivariatePipeline:
    def __init__(self, estimator_type: EstimatorType) -> None:
        self.estimator_type = estimator_type

    def _store_results(
        self,
        root: Path,
        distribution_str: str,
        p_event,
        num_normal,
        num_event,
        baseline_estimates,
        improved_estimates,
    ):
        # Store results
        parent = (
            root
            / distribution_str
            / f"p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}"
        )
        save_json(
            parent
            / f"p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.baseline.json",
            baseline_estimates,
        )
        save_json(
            parent
            / f"p_edge_{p_event}.n_normal_{num_normal}.n_edge_{num_event}.improved.json",
            improved_estimates,
        )

    def run_pipeline(self, estimator, root, *args, **kwargs):
        distribution_str = "mv_gaussian"

        distributions, uv_distribution, mv_distribution = mv_params.get_distributions()
        parameters = list(
            product(
                mv_params.p_event,
                mv_params.num_normal,
                mv_params.num_event,
            )
        )

        for p_event, num_normal, num_event in parameters:
            # Construct evaluation grid
            x_values = data_utils.get_evaluation_interval(distributions)

            # Get true distribution
            true = uv_distribution.pdf(x_values)
            threshold = scipy.stats.norm.ppf(1 - p_event)

            # Initialize dicts to store results
            baseline_estimates = {"x": x_values.tolist(), "true": true.tolist()}
            improved_estimates = {"x": x_values.tolist(), "true": true.tolist()}

            for run in tqdm(
                range(mv_params.num_estimates),
                desc=f"Multivariate Gaussian: norm={num_normal} edge={num_event} p_edge={p_event}",
            ):
                normal_data, event_data = data_utils.generate_data(
                    mv_distribution, num_normal, num_event, threshold, random_state=run
                )

                # Obtain estimates
                baseline, improved, _, _ = self.estimator_type.obtain_estimates(
                    estimator,
                    uv_distribution,
                    normal_data,
                    event_data,
                    threshold,
                    x_values,
                    distribution_str,
                    *args,
                    **kwargs,
                )
                baseline_estimates[f"run_{run}"] = baseline.tolist()
                improved_estimates[f"run_{run}"] = improved.tolist()

                # Save results every n runs
                if run % 10 == 0:
                    self._store_results(
                        root,
                        distribution_str,
                        p_event,
                        num_normal,
                        num_event,
                        baseline_estimates,
                        improved_estimates,
                    )

            # Save results at end of all runs
            self._store_results(
                root,
                distribution_str,
                p_event,
                num_normal,
                num_event,
                baseline_estimates,
                improved_estimates,
            )
