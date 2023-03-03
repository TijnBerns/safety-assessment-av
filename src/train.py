import torch
from typing import Tuple, List
from torch.utils.data import DataLoader
from model import FeedForward
from torch.utils.data.dataloader import default_collate
from config import Config as cfg
from itertools import product
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
import json
import scipy.stats
import data_utils
import numpy as np

import matplotlib.pyplot as plt

from scipy import signal


def train_test_pipeline(samples, eval_loader_norm, eval_loader_edge):
    results_dict = defaultdict(lambda: defaultdict(dict))

    # Construct dataloaders
    train_loader = DataLoader(
        samples, shuffle=True, batch_size=cfg.batch_size, collate_fn=default_collate, drop_last=True)
    val_loader = DataLoader(
        samples, shuffle=False, batch_size=cfg.batch_size, collate_fn=default_collate, drop_last=True)

    # # Fit models on data
    # baseline_kde = scipy.stats.gaussian_kde(d_norm[:,0][:M])
    num_layers = [3]
    num_hidden = [50]
    for nl, nh in product(num_layers, num_hidden):
        # Initialize checkpointer
        pattern = f"layers_{nl}.neuros_{nh}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
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
        trainer = pl.Trainer(max_steps=cfg.max_steps,
                             inference_mode=False, callbacks=[checkpointer])
        model = FeedForward(1, 1, nh, nl)
        trainer.fit(model, train_loader, val_loader)
        res_norm = trainer.test(model, eval_loader_norm)[0]['test_mse']
        res_edge = trainer.test(model, eval_loader_edge)[0]['test_mse']
        results_dict[nl][nh] = (res_norm, res_edge)
        print(f"\n\n============={(res_norm, res_edge)}=============\n\n")

    return results_dict


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    results_dicts = {}

    # Generate data from mv Gaussian
    normal_data, edge_data, p_edge = data_utils.generate_data(
        cfg.mean, cfg.cov, cfg.c)
    combined_data = data_utils.combine_data(normal_data, edge_data, p_edge)

    # Label training data
    _, bins = torch.histogram(normal_data[:, 0], cfg.num_bins)
    samples_norm, targets_norm = data_utils.annotate_data(
        normal_data[:, 0], bins)
    # samples_edge, targets_edge = data_utils.annotate_data(edge_data[:, 0], bins, targets_norm)
    samples_combined, targets_combined = data_utils.annotate_data(
        combined_data[:, 0], smooth=False)

    # Normal test set
    step_size = (bins[1] - bins[0]) / 5
    breakpoint()
    x_values_norm = torch.arange(
        min(normal_data[:, 0]), max(normal_data[:, 0]), step_size)
    true_cdf_norm = scipy.stats.norm.cdf(
        x_values_norm, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    test_samples_norm = list(zip(list(x_values_norm), list(true_cdf_norm)))
    eval_loader_norm = DataLoader(test_samples_norm)

    # Edge test set
    x_values_edge = torch.arange(
        min(edge_data[:, 0]), max(edge_data[:, 0]), step_size)
    true_cdf_edge = scipy.stats.norm.cdf(
        x_values_edge, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    test_samples_edge = list(zip(list(x_values_edge), list(true_cdf_edge)))
    eval_loader_edge = DataLoader(test_samples_edge)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot([x for x, _ in samples_norm], [
    #             y for _, y in samples_norm], alpha=0.5, color='black', linestyle='dotted')
    # axs[0].plot(x_values_norm, true_cdf_norm, alpha=0.5)
    # axs[0].plot(x_values_edge, true_cdf_edge, alpha=0.5)

    # axs[1].plot([x for x, _ in samples_combined], [
    #             y for _, y in samples_combined], alpha=0.5, color='black', linestyle='dotted')
    # axs[1].plot(x_values_norm, true_cdf_norm, alpha=0.5)
    # axs[1].plot(x_values_edge, true_cdf_edge, alpha=0.5)

    # print(f"p_edge: {p_edge * 100:.2f}%")
    # plt.show()

    # Fit models on normal data
    results_norm = train_test_pipeline(
        samples_norm, eval_loader_norm, eval_loader_edge)
    results_dicts['normal'] = results_norm
    with open("test.json", 'w+') as f:
        json.dump(results_dicts, f, indent=2)

    # Fit model on combined data
    results_edge = train_test_pipeline(
        samples_combined, eval_loader_norm, eval_loader_edge)
    results_dicts['normal+edge'] = results_edge
    with open("test.json", 'w+') as f:
        json.dump(results_dicts, f, indent=2)
