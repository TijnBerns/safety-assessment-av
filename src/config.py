from dataclasses import dataclass
import scipy.stats
from scipy.stats import wishart, chi2
from pathlib import Path
import numpy as np
import distributions as dist
from typing import Dict


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
    num_estimates = 10
    num_eval = 400
    evaluation_interval = {}
    
    # Correlation matrix used in Copulas
    c_target = np.array([[  1.0, 0.7,],
                        [0.7,  1.0]])

    # Distributions
    distributions = {
        'bivariate_guassian_a': scipy.stats.multivariate_normal([0, 0], [[2, 0.6], [0.6, 1.0]]),
        'bivariate_guassian_b': scipy.stats.multivariate_normal([0, 0], [[2, 1.0], [1.0, 1.0]]),
        'bivariate_gaussian_c': scipy.stats.multivariate_normal([0, 0], [[2, 1.4], [1.4, 1.0]]),  
        # 'gumbel_a': dist.Gaussian_Copula(c_target, [scipy.stats.gumbel_r(), scipy.stats.norm()]), 
        # 'beta_a': dist.Gaussian_Copula(c_target, [scipy.stats.beta(0.5,0.5), scipy.stats.norm()]), 
        # 'laplace_a': dist.Gaussian_Copula(c_target, [scipy.stats.laplace(), scipy.stats.norm()]),
    }
    
    single_distributions = {
        'bivariate_guassian_a': scipy.stats.norm(0.0, np.sqrt(2.0)), 
        'bivariate_guassian_b': scipy.stats.norm(0.0, np.sqrt(2.0)), 
        'bivariate_gaussian_c': scipy.stats.norm(0.0, np.sqrt(2.0)), 
        # 'gumbel_a' : scipy.stats.gumbel_r(),
        # 'laplace_a': scipy.stats.laplace(),
        # 'beta_a': scipy.stats.beta(0.5,0.5)
    }

    # Path variables
    # path_estimates = Path('/home/tberns/safety-assessment-av/estimates') # For run on cluster
    path_estimates = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates') # For local run
    
    # Neural network parameters
    nn_training_steps = 10_000
    nn_lr = 1e-3
    nn_batch_size = 100
    nn_num_hidden_nodes = 25
    nn_num_hidden_layers = 3
    

