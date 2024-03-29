from dataclasses import dataclass
import scipy.stats
from scipy.stats import wishart, chi2
from pathlib import Path
import numpy as np
import distributions as dist
from typing import List, Union


@dataclass
class SharedParameters:
    # Seed
    seed = 2023

    # Path variables
    path_estimates = Path(
        "/home/tberns/safety-assessment-av/estimates"
    )  # For run on cluster
    # path_estimates = Path(
    #     '/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates')  # For local run

    # Probability of observing an event
    p_event = [0.02, 0.04, 0.08, 0.16, 0.32]

    LABELS = {
        "num_norm": "$N_\\textrm{norm}$",
        "num_event": "$N_\\textrm{event}$",
        "p_event": "$p_\\textrm{event}$",
        "correlation": "$\\rho$",
    }

    FIGSIZE = (3.3, 2.0)


@dataclass
class UVParameters(SharedParameters):
    """Class containing the variables used troughout the experiments"""

    # The number of normal observations
    num_normal = [100, 1000, 10_000]

    # The number of event observations
    num_event = [
        100,
        1000,
        10_000,
    ]

    # Probability of observing an event
    p_event = [0.02, 0.04, 0.08, 0.16, 0.32]

    # The correlation between X_1 and X_2
    correlation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Value used for epsilon support
    epsilon = 0.01

    # Distributions
    distributions = [
        # 'gaussian',
        # 'gumbel',
        # 'beta',
        # 't'
    ]

    # Other parameters
    num_estimates = 100
    num_estimates_mv = 100
    num_eval = 400
    num_eval_mv = 100

    # Neural network parameters
    nn_training_steps = 10_000
    nn_lr = 1e-3
    nn_batch_size = 100
    nn_num_hidden_nodes = 25
    nn_num_hidden_layers = 3

    def get_distributions(distribution_str: str, correlation: float):
        distributions = {
            "gaussian": [scipy.stats.norm(0, 2), scipy.stats.norm()],
            "gumbel": [scipy.stats.gumbel_r(), scipy.stats.norm()],
            "beta": [scipy.stats.beta(0.5, 0.5), scipy.stats.norm()],
            "t": [scipy.stats.t(1), scipy.stats.norm()],
        }

        c_target = np.array(
            [
                [
                    1.0,
                    correlation,
                ],
                [correlation, 1.0],
            ]
        )
        uv_distribution = distributions[distribution_str][0]
        mv_distribution = dist.Gaussian_Copula(
            c_target, distributions[distribution_str]
        )
        return uv_distribution, mv_distribution


@dataclass
class MVParameters(SharedParameters):
    """Class containing the variables used troughout the experiments involving multivariate distributions"""

    # The number of normal observations
    num_normal = [1000]

    # The number of event observations
    num_event = [1000]

    # Other parameters
    num_estimates = 10
    num_eval = 100

    # Neural network parameters
    nn_training_steps = 10_000
    nn_lr = 1e-3
    nn_batch_size = 100
    nn_num_hidden_nodes = 25
    nn_num_hidden_layers = 3

    def get_distributions():
        # mean = np.array([-1, 1, 0])
        # cov = np.array([
        #     [2.0, 0.5, 0.7],
        #     [0.5, 1.0, 0.3],
        #     [0.7, 0.3, 1.0]])
        mean = np.array([6, 3])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])

        distributions = [scipy.stats.norm(mean[i], cov[i, i]) for i in range(len(mean))]
        uv_distribution = scipy.stats.multivariate_normal(mean, cov)
        mv_distribution = scipy.stats.multivariate_normal(mean, cov)

        return distributions, uv_distribution, mv_distribution
