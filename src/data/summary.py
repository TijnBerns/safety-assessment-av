"""
Plotting functionalities for summary statistics of UCI data
"""

import sys

sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

from base import CustomDataset
from miniboone import MiniBoone
from power import Power
from gas import Gas
from hepmass import Hepmass
from bsds300 import BSDS300Dataset

import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

FIGSIZE = (6, 7)


def print_summary(dataset: CustomDataset):
    mean = np.mean(dataset.data, axis=0)
    std = np.std(dataset.data, axis=0)
    skewness = scipy.stats.skew(dataset.data, axis=0)
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"skewness: {skewness}")
    return mean, std, skewness


def scatter(normal, event, ax):
    ax.scatter(np.arange(len(normal)), normal, alpha=0.5)
    ax.scatter(np.arange(len(event)), event, alpha=0.5)


column_labels = ["Mean", "Standard deviation", "Skewness"]

if __name__ == "__main__":
    datasets = [Hepmass, Gas, Power, MiniBoone]

    fig, axes2d = plt.subplots(
        nrows=len(datasets), ncols=3, sharex=False, sharey=False, figsize=FIGSIZE
    )

    for i, row in enumerate(axes2d):
        normal = datasets[i](split="normal_sampled")
        event = datasets[i](split="event_sampled")
        normal_summary = print_summary(normal)
        event_summary = print_summary(event)

        for j, cell in enumerate(row):
            scatter(normal_summary[j], event_summary[j], cell)

            if i == 0:
                cell.set_title(f"{column_labels[j]}")
            if j == 0:
                cell.set_ylabel(f"{normal.name.upper()}")

        plt.tight_layout()
        plt.savefig("img/summary_statistics_sampled.pgf")
