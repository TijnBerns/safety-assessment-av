from typing import Tuple, List
import numpy as np


def compute_p_edge(data, threshold, dim=-1):
    """Computes the fraction of edge data in the provided samples. 
    This fraction is based on the number of samples where x is larger than c along dimension dim
    """
    return len(data[data[:, dim] > threshold]) / len(data)


def determine_threshold(data: np.ndarray, frac_edge: float, dim: int = -1):
    """Sets a threshold on the final column of data such that frac_edge * 100 percent of samples are larger than the threshold
    """
    y_sorted = np.sort(data[:, dim])
    i = int(len(y_sorted) * (1- frac_edge))
    return y_sorted[i]


def generate_data(ditribution, frac_edge: float, num_norm, num_edge, dim: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate normal and edge data from multivariate normal distribution
    """
    # Set threshold on dim such that
    data = ditribution.rvs(1_000_000)
    threshold = determine_threshold(data, frac_edge)

    # Filter edge data, and redraw sample for normal data
    edge_data = data[data[:, dim] > threshold][:num_edge]
    normal_data = ditribution.rvs(num_norm)
    return normal_data, edge_data, threshold


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
    
    repeat_factor = int(1 + 1 // p_edge)
    remainder = round((1 % p_edge) * (len(normal_data)))

    # Create combined data tensor
    combined_data = np.repeat(normal_data_only, repeat_factor, axis=0)
    combined_data = np.concatenate((combined_data, normal_data_only[:remainder], edge_data, edge_from_normal_data))
    return combined_data


def annotate_data(data: np.ndarray, bins: np.ndarray, targets: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """Adds a target to every datapoints which corresponds to the value on the emp. cdf.
    """
    counts, _ = np.histogram(data, bins)
    if targets is None:
        targets = np.cumsum(counts) / len(data)

    # Sort data, as targets, are sorted by definition
    data_sorted = np.sort(data)
    samples = []

    # Construct samples
    k = 0
    for i in range(len(counts)):
        if counts[i] == 0:
            continue

        for _ in range(int(counts[i].item())):
            samples.append((data_sorted[k], targets[i]))
            k += 1

    return samples, targets
