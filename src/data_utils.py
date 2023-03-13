from typing import Tuple, List
import numpy as np


def compute_p_edge(data, threshold, dim):
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
    data = ditribution.rvs(10_000_000)
    threshold = determine_threshold(data, frac_edge)

    # Filter edge data, and redraw sample for normal data
    edge_data = data[data[:, dim] > threshold][:num_edge]
    normal_data = ditribution.rvs(num_norm)
    return normal_data, edge_data, threshold


def filter_data(normal_data, edge_data, threshold: float, dim=-1) -> Tuple[np.ndarray, np.ndarray]: 
    """Filters data be adding all edge data points in normal data array to the edge data array
    """
    normal_data_filtered = normal_data[normal_data[:,dim] <= threshold]
    edge_data_filtered = np.concatenate((edge_data, normal_data[normal_data[:,dim] > threshold]))
    return normal_data_filtered, edge_data_filtered
    


# def combine_data(normal_data, edge_data, threshold, p_edge: float):
#     """Combines normal and edge data retaining the fraction of edge cases by duplicating samples from the normal data.
#     """
#     # Split the normal data based on c
#     normal_only_data = normal_data[normal_data[:, 1] <= threshold]
#     edge_from_normal_data = normal_data[normal_data[:, 1] > threshold]

#     # Compute how often we need to duplicate the normal data
#     repeat_factor = int(1 + 1 // p_edge)
#     remainder = round((1 % p_edge) * (len(normal_data)))

#     # Create combined data tensor
#     combined_data = normal_only_data.repeat(repeat_factor, 1)
#     combined_data = torch.cat(
#         (combined_data, normal_only_data[:remainder], edge_data, edge_from_normal_data))
#     return combined_data


# def annotate_data(data: torch.Tensor, bins: torch.Tensor, targets: torch.Tensor = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
#     """Adds a target to every datapoints which corresponds to the value on the emp. cdf.
#     """
#     counts, _ = torch.histogram(data, bins)
#     if targets is None:
#         targets = torch.cumsum(counts, dim=0) / len(data)

#     # Sort data, as targets, are sorted by definition
#     data_sorted = torch.sort(data, dim=0)[0]
#     samples = []

#     # Construct samples
#     k = 0
#     for i in range(len(counts)):
#         if counts[i] == 0:
#             continue

#         for _ in range(int(counts[i].item())):
#             samples.append((data_sorted[k], targets[i]))
#             k += 1

#     return samples, targets
