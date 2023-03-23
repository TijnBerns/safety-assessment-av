from dataclasses import dataclass
import scipy.stats
from scipy.stats import wishart, chi2
from pathlib import Path
import numpy as np
import distributions as dist


@dataclass
class Config():
    # Seed
    seed = 2023

    # Variables
    num_normal = [
        100,
        1000, 
        10_000
        ]
    
    num_edge = [
        100, 
        1000, 
        10_000
        ]
    
    p_edge = [
        0.01,
        0.02,
        0.04,
        0.10,
        0.20,
        0.30
    ]
    num_estimates = 100
    evaluation_interval = np.linspace(-8, 8, 400)

    # Distributions
    distributions = {
        # 'bivariate_guassian_a': scipy.stats.multivariate_normal([0, 0], [[2, 0.5], [0.5, 2]]),
        'bivariate_guassian_b': scipy.stats.multivariate_normal([0, 0], [[2, 1.3], [1.3, 1]]),
        # 'bivariate_guassian_c': scipy.stats.multivariate_normal([0, 0], [[8, 0.5], [0.5, 2]]),
        # 'multimodal_a': dist.UnivariateThreshold(
        #     dist.Mixture([0.7,0.3], [scipy.stats.norm(-2, 1), scipy.stats.norm(2, 1)]),
        #     scipy.stats.norm(0.0, np.sqrt(2.0))),
        # 'gumbel_a': dist.UnivariateThreshold(
        #     scipy.stats.gumbel_r(),
        #     scipy.stats.norm(0.0, np.sqrt(2.0)))
        
    }
    
    single_distributions = {
        # 'bivariate_guassian_a': scipy.stats.norm(0.0, np.sqrt(2.0)), 
        'bivariate_guassian_b': scipy.stats.norm(0.0, np.sqrt(4.0)), 
        # 'bivariate_guassian_c': scipy.stats.norm(0.0, np.sqrt(8.0)), 
        # 'multimodal_a': dist.Mixture([0.7,0.3], [scipy.stats.norm(-2, 1), scipy.stats.norm(2, 1)]),
        # 'gumbel_a' : scipy.stats.gumbel_r()
    }

    # Path variables
    path_estimates = Path('/home/tberns/safety-assessment-av/estimates') # For run on cluster
    # path_estimates = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates') # For local run
    
    # Neural network parameters
    nn_training_steps = 10_000
    nn_lr = 1e-3
    nn_batch_size = 100
    nn_num_hidden_nodes = 25
    nn_num_hidden_layers = 3
