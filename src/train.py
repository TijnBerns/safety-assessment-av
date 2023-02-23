import numpy as np
import torch
from torch.distributions import MultivariateNormal
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from model import FeedForward
from torch.utils.data.dataloader import default_collate
from config import Config as cfg


import pytorch_lightning as pl

import matplotlib.pyplot as plt

from pprint import pprint


# Constants
N = int(10e6)
M = int(300)
c = 1


# Distribution parameters
mu_X = 0
mu_Y = 0
sigma_X_sq = 2
sigma_Y_sq = 1
mean = torch.tensor([mu_X, mu_Y], dtype=torch.float32)
cov = torch.tensor([[sigma_X_sq, 0.8], [0.8, sigma_Y_sq]], dtype=torch.float32)


def generate_data(mean: torch.Tensor, cov: torch.Tensor, threshold: float, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    # Define the multivariate distribution
    mv = MultivariateNormal(mean, cov)
    data = mv.sample_n(N)

    # Filter edge data, and redraw sample for normal data
    edge_data = data[data[:, dim] > c][:M]
    normal_data = mv.sample_n(M)
    return normal_data, edge_data


def annotate_data(data: torch.Tensor, bins: torch.Tensor) -> List[torch.Tensor]:
    counts, _ = torch.histogram(data[:, 0], bins)
    targets = torch.cumsum(counts, dim=0) / len(data[:, 0])

    # Sort data, as targets, are sorted by definition
    data_sorted = torch.sort(data[:, 0], dim=0)[0]
    samples = []

    # Construct samples
    k = 0
    for i in range(len(counts)):
        if counts[i] == 0:
            continue

        for _ in range(int(counts[i].item())):
            samples.append((data_sorted[k], targets[i]))
            k += 1

    return samples


if __name__ == "__main__":
    d_norm, d_edge = generate_data(mean, cov, c)
    counts, bins = torch.histogram(d_norm[:, 0], M)
    emp_cdf = torch.cumsum(counts, dim=0) / len(d_norm[:, 0])

    # Annotate data
    samples = annotate_data(d_edge, bins)
    samples.extend(annotate_data(d_norm, bins))
    train_loader = DataLoader(samples,shuffle=True, batch_size=cfg.batch_size, collate_fn=default_collate, drop_last=True)
    val_loader = DataLoader(samples,shuffle=True, batch_size=1)

    trainer = pl.Trainer(max_steps=cfg.max_steps, inference_mode=False)
    model = FeedForward(1, 1, 256, 3)
    trainer.fit(model, train_loader, val_loader)
    
    
    
    