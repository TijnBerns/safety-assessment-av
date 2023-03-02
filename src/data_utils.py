import torch
from torch.distributions import MultivariateNormal
from typing import Tuple, List
from config import Config as cfg


def compute_p_edge(data, dim):
    """Computes the fraction of edge data in the provided samples. 
    This fraction is based on the number of samples where x is larger than c along dimension dim
    """
    return len(data[data[:, dim] > cfg.c]) / len(data)


def generate_data(mean: torch.Tensor, cov: torch.Tensor, threshold: float, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate normal and edge data from multivariate normal distribution
    """
    mv = MultivariateNormal(mean, cov)
    data = mv.sample_n(cfg.N)

    # Filter edge data, and redraw sample for normal data
    edge_data = data[data[:, dim] > cfg.c][:cfg.M]
    normal_data = mv.sample_n(cfg.M)
    return normal_data, edge_data, compute_p_edge(data, dim)


def combine_data(normal_data: torch.Tensor, edge_data: torch.Tensor, p_edge: float):
    """Combines normal and edge data retaining the fraction of edge cases by duplicating samples from the normal data.
    """
    # Split the normal data based on c
    normal_only_data = normal_data[normal_data[:, 1] <= cfg.c]
    edge_from_normal_data = normal_data[normal_data[:, 1] > cfg.c]

    # Compute how often we need to duplicate the normal data
    repeat_factor = int(1 + 1 // p_edge)
    remainder = round((1 % p_edge) * (len(normal_data)))

    # Create combined data tensor
    combined_data = normal_only_data.repeat(repeat_factor, 1)
    combined_data = torch.cat(
        (combined_data, normal_only_data[:remainder], edge_data, edge_from_normal_data))
    return combined_data


def annotate_data(data: torch.Tensor, bins: torch.Tensor, targets: torch.Tensor = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Adds a target to every datapoints which corresponds to the value on the emp. cdf.
    """
    counts, _ = torch.histogram(data, bins)
    if targets is None:
        targets = torch.cumsum(counts, dim=0) / len(data)

    # Sort data, as targets, are sorted by definition
    data_sorted = torch.sort(data, dim=0)[0]
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
