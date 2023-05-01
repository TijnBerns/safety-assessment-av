from dataclasses import dataclass
import scipy.stats
from scipy.stats import wishart, chi2
from pathlib import Path
import numpy as np
import distributions as dist
from typing import Dict


@dataclass
class Config():
    """Class containing the variables used troughout the experiments
    """
    
    # Seed
    seed = 2023

    # The number of normal observations
    num_normal = [
        100,
        1000,
        10_000
    ]

    # The number of event observations
    num_event = [
        100,
        1000,
        10_000,
    ]

    # Probability of observing an event
    p_event = [
        0.02,
        0.04,
        0.08,
        0.16,
        0.32
    ]

    # The correlation between X_1 and X_2
    correlation = [
        0.1,
        0.5,
        0.7,
        0.9
    ]

    # Distributions
    distributions = [
        'gaussian',
        # 'gumbel',
        # 'beta'
    ]

    # Other parameters
    num_estimates = 100
    num_eval = 400

    # Path variables
    # path_estimates = Path('/home/tberns/safety-assessment-av/estimates') # For run on cluster
    path_estimates = Path(
        '/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates')  # For local run

    # Neural network parameters
    nn_training_steps = 10_000
    nn_lr = 1e-3
    nn_batch_size = 100
    nn_num_hidden_nodes = 25
    nn_num_hidden_layers = 3

    def get_distributions(distribution_str, correlation):
        c_target = np.array([[1.0, correlation,], [correlation,  1.0]])

        mv_distributions = {
            'gaussian': dist.Gaussian_Copula(c_target, [scipy.stats.norm(0, 2), scipy.stats.norm()]),
            'gumbel': dist.Gaussian_Copula(c_target, [scipy.stats.gumbel_r(), scipy.stats.norm()]),
            'beta': dist.Gaussian_Copula(c_target, [scipy.stats.beta(0.5, 0.5), scipy.stats.norm()]),
            'laplace': dist.Gaussian_Copula(c_target, [scipy.stats.laplace(), scipy.stats.norm()]),
        }

        distributions = {
            'gaussian': scipy.stats.norm(0, 2),
            'gumbel': scipy.stats.gumbel_r(),
            'laplace': scipy.stats.laplace(),
            'beta': scipy.stats.beta(0.5, 0.5)
        }

        return  distributions[distribution_str], mv_distributions[distribution_str]

