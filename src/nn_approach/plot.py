import torch
import scipy
import numpy as np
from config import Config as cfg
import data
import matplotlib.pyplot as plt
from model import FeedForward

def plot(model, bins, emp_cdf, save=None):
    # plot estimated pdf
    x_values = torch.linspace(bins[0], bins[-1], len(bins))
    pdf_true = scipy.stats.norm.pdf(x_values, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    pdf_nn = model.compute_pdf(x_values)
    # pdf_kde = [baseline(x) for x in x_values]
    _, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].plot(x_values, pdf_true, label='true pdf')
    axs[0].plot(x_values, pdf_nn, label='NN estimate')
    # axs[0].plot(x_values, pdf_kde, label='KDE estimate')
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


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    normal_data, edge_data, p_edge = data.generate_data(cfg.mean, cfg.cov, cfg.c)
    counts, bins = torch.histogram(normal_data[:, 0], cfg.num_bins)
    emp_cdf = torch.cumsum(counts, dim=0) / len(normal_data[:, 0])
    
    model = FeedForward.load_from_checkpoint('lightning_logs/version_127/checkpoints/layers_3.neuros_50.epoch_0045.step_000005000.val-mse_0.0001.last.ckpt', 
                                             num_in=1, num_out=1,num_hidden=50, num_layers=3)
    # breakpoint()
    plot(model, bins, emp_cdf)