from abc import ABC
import scipy
import scipy.stats
import scipy.integrate
import numpy as np

from typing import List, Tuple
from tqdm import tqdm

class Estimator(ABC):
    def __init__(self) -> None:
        pass

    def fit(data: List[np.array]):
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
        
    
    

