from typing import Tuple, List
import numpy as np
import scipy.stats
from scipy.optimize import fsolve
from itertools import product 


def compute_p_edge(data, threshold, dim=-1):
    """Computes the fraction of edge data in the provided samples. 
    This fraction is based on the number of samples where x is larger than c along dimension dim
    """
    return len(data[data[:, dim] > threshold]) / len(data)

def determine_threshold(frac_edge: float, distribution=scipy.stats.norm()):
    assert frac_edge > 0 and frac_edge < 1
    try:
        return determine_threshold_analytical(frac_edge, distribution)
    except:
        print('PPF function not available, intersection with CDF solved numerically')
        return determine_threshold_numerical(frac_edge, distribution)

def determine_threshold_analytical(frac_edge: float, distribution=scipy.stats.norm()):
    return distribution.ppf(1 - frac_edge)

def determine_threshold_numerical(frac_edge: float, distribution=scipy.stats.norm()):
    xr = np.sort(distribution.rvs(int(10e3)))
    x0 = np.percentile(xr, frac_edge * 100)
    
    def _f(x):
        return distribution.cdf(x) - (1-frac_edge)
    return fsolve(_f, x0)

def generate_data(ditribution, num_norm, num_edge, threshold, dim:int=-1, random_state=0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate normal and edge data from multivariate normal distribution
    """
    data = ditribution.rvs(1_000_000, random_state=random_state)

    # Filter edge data, and redraw sample for normal data
    edge_data = np.array([])
    i = 0
    while True:
        i += 1
        new_edge_data = data[data[:, dim] > threshold]
        edge_data = np.concatenate((edge_data, new_edge_data)) if edge_data.size else new_edge_data
        
        if len(edge_data) > num_edge:
            edge_data = edge_data[:num_edge]
            break
        
        data = ditribution.rvs(100_000, random_state=random_state + i)
        
    normal_data = ditribution.rvs(num_norm, random_state=random_state + 1 + i)
    return normal_data, edge_data


def split_data(data: np.ndarray, threshold: float, dim=-1):
    """Splits a dataset at the given threshold at column dim
    """
    data_split_a = data[data[:,dim] <= threshold]
    data_split_b = data[data[:,dim] > threshold]
    return data_split_a, data_split_b


def filter_data(normal_data: np.ndarray, edge_data: np.ndarray, threshold: float=0, dim=-1) -> Tuple[np.ndarray, np.ndarray]: 
    """Filters data be adding all edge data points in normal data array to the edge data array
    """
    normal_data_only, edge_from_normal_data = split_data(normal_data, threshold, dim)
    edge_data_only = np.concatenate((edge_data, edge_from_normal_data))
    return normal_data_only, edge_data_only
    

def combine_data(normal_data: np.ndarray, edge_data: np.ndarray, threshold: float, p_edge: float):
    """Combines normal and edge data retaining the fraction of edge cases by duplicating samples from the normal data.
    """
    if p_edge == 0:
        print("WARNING: estimated p_edge equals 0, thus data cannot be combined.")
        return normal_data
    
    # Split the normal data based on threshold
    normal_data_only, edge_from_normal_data = split_data(normal_data, threshold)
    
    repeat_factor = ((1-p_edge) * (len(edge_from_normal_data) + len(edge_data)) / (p_edge * len(normal_data_only)))
    N = np.floor(repeat_factor)
    M = round((repeat_factor % 1) * len(normal_data_only))

    combined_data = np.repeat(normal_data_only, N, axis=0)
    combined_data = np.concatenate((combined_data, normal_data_only[:M], edge_data, edge_from_normal_data))
    return combined_data    
    

class EmpericalCDF():
    """Wrapper class for constructing an emperical CDF
    """
    def __init__(self) -> None:
        self.bins = None
        self.x_values = None
        self.y_values = None
    
    def fit(self, data: np.ndarray, num_bins:int=None):
        """Fits an empercical CDF to provided data

        Args:
            data (np.ndarray): 1D numpy array containing the data
            num_bins (int, optional): The number of bins used when constructing the CDF. Defaults to 10 * len(data)
        """

        if num_bins is None:
            num_bins = 10 * len(data)
        
        counts, bins = np.histogram(data, num_bins)
        
        self.bins = bins
        self.x_values = bins[1:]
        self.y_values = np.cumsum(counts) / len(data)
  
    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """Returns the values on the emperical CDF of the given datapoints

        Args:
            data (np.ndarray): Numpy array containing the data the evaluate

        Returns:
            np.ndarray: Numpy array containing values on CDF for givend datapoints
        """
        data = np.sort(data)
        emp_cdf = np.zeros_like(data)
    
        j = 0
        for i in range(len(data)):
            while j < len(self.x_values) - 1 and data[i] > self.x_values[j]:
                j += 1
            emp_cdf[i] = self.y_values[j]
        
        return emp_cdf
    

def annotate_data(data: np.ndarray) -> List[Tuple[float, float]]:
    """Adds a target to every datapoints which corresponds to the value on the emp. cdf.
    """
    emp_cdf = EmpericalCDF()
    emp_cdf.fit(data)
    y_values = emp_cdf.evaluate(data)
    return list(zip(data, y_values))


def get_evaluation_interval(distribution , n: int=400, reduce_dim=0):
    """Gets the interval on which most of the data is distributed.
    """
    if type(distribution) is not list:
        distribution = [distribution]
        
    intervals = []    
    for d in distribution:        
        min_x = d.ppf(0.0005)
        max_x = d.ppf(0.9995)
        intervals.append(np.linspace(min_x, max_x, n))
    
    # Univariate case: return single interval
    if len(intervals) == 1:
        return intervals[0]
    
    # Multivariate case: return grid values
    if reduce_dim == 0:
        x_values = np.array(list(product(*intervals)))
    else:
        x_values = np.array(list(product(*intervals[:reduce_dim])))
    return  x_values

def get_epsilon_support_uv(fn, epsilon, x_values):
    # List of intervals
    out = []
    y_values = fn(x_values)
    
    # Step 1: Get rough estimate of interval using predefined values for x
    gt_epsilon = y_values > epsilon
    mask = np.r_[False, gt_epsilon, False]
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    s0,s1 = idx[::2], idx[1::2]
    
    for (i,j) in zip(s0, s1-1): 
        out.append((x_values[i], x_values[j]))
    
    # Step 2: Refine estimates using fsolve
    def _g(x):
        return fn(x) - epsilon
    
    for i in range(len(out)):
        out[i] = list(fsolve(_g, out[i]))
    return out
    