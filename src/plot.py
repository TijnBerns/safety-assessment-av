import torch
import scipy
import numpy as np
from config import Config as cfg
import data_utils
import matplotlib.pyplot as plt
from model import FeedForward
from collections import defaultdict

def plot(model, save=None, titles=defaultdict):
    normal_data, edge_data, _ = data_utils.generate_data(cfg.mean, cfg.cov, cfg.c)
    x_values = torch.linspace(min(normal_data[:, 0]), max(edge_data[:, 0]), 10000)
    true_cdf = scipy.stats.norm.cdf(x_values, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    
    # plot estimated pdf
    pdf_true = scipy.stats.norm.pdf(x_values, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    pdf_nn = model.compute_pdf(x_values)
    # pdf_kde = [baseline(x) for x in x_values]
    _, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].plot(x_values, pdf_true, label='true pdf')
    axs[0].plot(x_values, pdf_nn, label='NN estimate')
    # axs[0].plot(x_values, pdf_kde, label='KDE estimate')
    axs[0].legend()
    axs[1].set_title(titles[0])
    
    # Plot estimated CDF vs tru
    cdf_nn = model.compute_cdf(x_values)
    axs[1].plot(true_cdf, label='True CDF')
    axs[1].plot(cdf_nn, label='NN estimate')
    axs[1].legend()
    axs[1].set_title(titles[1])

    if save is None:
        plt.show()
        return
    plt.savefig(save)


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    
    model = FeedForward.load_from_checkpoint('lightning_logs/version_31/checkpoints/layers_3.neuros_50.epoch_0666.step_000010000.val-mse_0.0001.last.ckpt', 
                                             num_in=1, num_out=1,num_hidden=50, num_layers=3)
    # breakpoint()
    plot(model, save='img/normal-no_sigmoid')