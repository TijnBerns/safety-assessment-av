from pprint import pprint
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from model import FeedForward
from torch.utils.data.dataloader import default_collate
from config import Config as cfg
import scipy
from itertools import product
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from collections import defaultdict
import json


# Constants
N = int(1e7)
M = cfg.batch_size
c = 1

# Distribution parameters
mu_X = 0
mu_Y = 0
sigma_X_sq = 2
sigma_Y_sq = 1
mean = torch.tensor([mu_X, mu_Y], dtype=torch.float32)
cov = torch.tensor([[sigma_X_sq, 0.8], [0.8, sigma_Y_sq]], dtype=torch.float32)


def compute_p_edge(data, dim):
    """Computes the fraction of edge data in the provided samples. 
    This fraction is based on the number of samples where x is larger than c along dimension dim
    """
    return len(data[data[:,dim] > c]) / len(data)


def generate_data(mean: torch.Tensor, cov: torch.Tensor, threshold: float, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate normal and edge data from multivariate normal distribution
    """
    mv = MultivariateNormal(mean, cov)
    data = mv.sample_n(N)

    # Filter edge data, and redraw sample for normal data
    edge_data = data[data[:, dim] > c][:M]
    normal_data = mv.sample_n(M)
    return normal_data, edge_data, compute_p_edge(data, dim)


def combine_data(normal_data: torch.Tensor, edge_data: torch.Tensor, p_edge: float):
    """Combines normal and edge data retaining the fraction of edge cases by duplicating samples from the normal data.
    """
    # Split the normal data based on c
    normal_only_data = normal_data[normal_data[:,1] <= c]
    edge_from_normal_data = normal_data[normal_data[:,1] > c]
    
    # Compute how often we need to duplicate the normal data
    repeat_factor = int(1 + 1 // p_edge)
    remainder = round((1 % p_edge) * (len(normal_data)))
    
    # Create combined data tensor
    combined_data = normal_only_data.repeat(repeat_factor, 1)
    combined_data = torch.cat((combined_data, normal_only_data[:remainder], edge_data, edge_from_normal_data))
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


def plot(model, bins, baseline, save=None):
    # plot estimated pdf
    x_values = torch.linspace(bins[0], bins[-1], len(bins))
    pdf_true = scipy.stats.norm.pdf(x_values, mu_X, np.sqrt(sigma_X_sq))
    pdf_nn = model.compute_pdf(x_values)
    pdf_kde = [baseline(x) for x in x_values]
    _, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].plot(x_values, pdf_true, label='true pdf')
    axs[0].plot(x_values, pdf_nn, label='NN estimate')
    axs[0].plot(x_values, pdf_kde, label='KDE estimate')
    axs[0].legend()

    # Plot CDF
    cdf_nn = model.compute_cdf(x_values)
    axs[1].plot(emp_cdf, label='emp. cdf (target)')
    axs[1].plot(cdf_nn, label='NN estimate')
    axs[1].legend()

    if save is None:
        plt.show()
        return
    plt.savefig(save)


def train_test_pipeline(samples, eval_loader_norm, eval_loader_edge):
    results_dict = defaultdict(lambda: defaultdict(dict))

    # Construct dataloaders
    train_loader = DataLoader(
        samples, shuffle=True, batch_size=cfg.batch_size, collate_fn=default_collate, drop_last=True)
    val_loader = DataLoader(
        samples, shuffle=False, batch_size=cfg.batch_size, collate_fn=default_collate, drop_last=True)

    # # Fit models on data
    # baseline_kde = scipy.stats.gaussian_kde(d_norm[:,0][:M])
    num_layers = [1, 2, 3]
    num_hidden = [10, 25, 50]
    for nl, nh in product(num_layers, num_hidden):
        # Initialize checkpointer
        pattern = "epoch_{epoch:04d}.step_{step:09d}.val-mse_{val_mse:.4f}"
        ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
        checkpointer = ModelCheckpoint(
            save_top_k=1,
            every_n_train_steps=500,
            monitor="val_mse",
            filename=pattern + ".best",
            save_last=True,
            auto_insert_metric_name=False,
        )

        # Fit the model
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                             inference_mode=False, callbacks=[checkpointer])
        model = FeedForward(1, 1, nh, nl)
        trainer.fit(model, train_loader, val_loader)
        res_norm = trainer.test(model, eval_loader_norm)[0]['test_mse']
        res_edge = trainer.test(model, eval_loader_edge)[0]['test_mse']
        results_dict[nl][nh] = (res_norm, res_edge)
        print(f"\n\n============={(res_norm, res_edge)}=============\n\n")

    # plot(model, bins, kde_baseline)
    return results_dict


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    results_dicts = {}

    # Generate data from mv Gaussian
    normal_data, edge_data, p_edge = generate_data(mean, cov, c)
    combined_data = combine_data(normal_data, edge_data, p_edge)

    num_bins = M
    counts, bins = torch.histogram(normal_data[:, 0], num_bins)
    emp_cdf = torch.cumsum(counts, dim=0) / len(normal_data[:, 0])

    # Label data and construct data loaders
    samples_norm, targets_norm = annotate_data(normal_data[:, 0], bins)
    samples_edge, targets_edge = annotate_data(edge_data[:, 0], bins)
    samples_combined, targets_combined = annotate_data(combined_data[:,0], bins)
    eval_loader_norm = DataLoader(samples_norm)
    eval_loader_edge = DataLoader(samples_edge)
    
    # Fit models on normal data
    results_norm = train_test_pipeline(samples_norm, eval_loader_norm, eval_loader_edge)
    results_dicts['normal'] = results_norm
    with open("test.json", 'w+') as f:
        json.dump(results_dicts, f, indent=2)

    # Fit model on combined data
    results_edge = train_test_pipeline(combined_data, eval_loader_norm, eval_loader_edge)
    results_dicts['normal+edge'] = results_edge
    with open("test.json", 'w+') as f:
        json.dump(results_dicts, f, indent=2)
