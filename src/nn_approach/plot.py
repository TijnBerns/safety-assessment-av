import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from model import FeedForward
from collections import defaultdict
from config import Config as cfg

def main(baseline_path, improved_path):
    baseline_model = FeedForward.load_from_checkpoint(baseline_path, num_in=1, num_hidden=cfg.nn_num_hidden_nodes, num_out=1, num_layers=cfg.nn_num_hidden_layers)
    improved_model = FeedForward.load_from_checkpoint(improved_path, num_in=1, num_hidden=cfg.nn_num_hidden_nodes, num_out=1, num_layers=cfg.nn_num_hidden_layers)

    x_values = torch.linspace(-6,6, 500)
    base_cdf_hat = baseline_model.compute_cdf(x_values)
    base_pdf_hat = baseline_model.compute_pdf(x_values)
    imp_cdf_hat = improved_model.compute_cdf(x_values)
    imp_pdf_hat = improved_model.compute_pdf(x_values)

    _, axs = plt.subplots(1,2)
    axs[0].plot(x_values, cfg.single_distributions_x1['bivariate_guassian_b'].cdf(x_values), label='true', linestyle='dotted')
    axs[0].plot(x_values, base_cdf_hat, label='baseline', alpha=0.5)
    axs[0].plot(x_values, imp_cdf_hat, label='improved', alpha=0.5)
    axs[0].legend()
    
    axs[1].plot(x_values, cfg.single_distributions_x1['bivariate_guassian_b'].pdf(x_values), label='true', linestyle='dotted')
    axs[1].plot(x_values, base_pdf_hat, label='baseline', alpha=0.5)
    axs[1].plot(x_values, imp_pdf_hat, label='improved', alpha=0.5)
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    baseline_path = 'lightning_logs/version_51/checkpoints/layers_3.neuros_25.epoch_0002.step_000020300.val-mse_0.0000.last.ckpt'
    improved_path = 'lightning_logs/version_51/checkpoints/layers_3.neuros_25.epoch_0002.step_000020300.val-mse_0.0000.last.ckpt'
    
    main(baseline_path, improved_path)