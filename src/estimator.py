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
        self.combined_fit = None

    def fit_baseline(self, data):
        self.baseline = scipy.stats.gaussian_kde(data.T)

    def fit_normal(self, data: np.array):
        self.normal_fit = scipy.stats.gaussian_kde(data.T)

    def fit_edge(self, data: np.array):
        self.combined_fit = scipy.stats.gaussian_kde(data.T)

    def baseline_estimate(self, x):
        return scipy.integrate.quad(lambda y: self.normal_fit([x, y]), -np.inf, np.inf)[0]
        
    def improved_estimate(self, x, c, p_normal, p_edge):
        integral_till_c = scipy.integrate.quad(lambda y: self.normal_fit([x, y]) + self.normal_fit([x, 2 * c - y] if y < c else 0), -np.inf, c)[0]
        integral_from_c = scipy.integrate.quad(lambda y: self.combined_fit([x, y]) + self.combined_fit([x, 2 * c - y] if y >= c else 0), c, np.inf)[0]
        return p_normal * integral_till_c + p_edge * integral_from_c
    
    def estimate(self, x_values, estimate_fn, *args, **kwargs):
        estimate = np.empty_like(x_values)
        
        for i in range(len(x_values)):
            estimate[i] = estimate_fn(x_values[i], **kwargs)
        return estimate
        
    
    

