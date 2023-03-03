from dataclasses import dataclass
import torch


class Config():
    seed = 2023
    batch_size = 32
    max_steps = 10000
    lr = 1e-3
    
    # Constants
    N = int(1e7)
    M = 500
    num_bins = 10 * M
    c = 1

    # Distribution parameters
    mu_X = 0
    mu_Y = 0
    sigma_X_sq = 2
    sigma_Y_sq = 1
    mean = torch.tensor([mu_X, mu_Y], dtype=torch.float32)
    cov = torch.tensor([[sigma_X_sq, 0.8], [0.8, sigma_Y_sq]], dtype=torch.float32)