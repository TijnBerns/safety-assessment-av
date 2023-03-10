from dataclasses import dataclass
import scipy.stats
from pathlib import Path

@dataclass
class GlobalConfig():
    # Seed
    seed = 2023
    
    # Variables
    num_normal = [100, 1000, 10_000]
    num_edge = [100, 1000, 10_000]
    p_edge = [0.01, 0.02, 0.04, 0.10, 0.20, 0.30]
    num_estimates = 200

    # Distributions
    distributions = {
        'bivariate_guassian_a': scipy.stats.multivariate_normal([0, 0], [[2, 0.5], [0.5, 2]])
    }

    # Path variables
    path_estimates = Path('../estimates')