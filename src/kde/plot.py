import sys
sys.path.append('src')

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

from parameters import UVParameters as uv_params
from utils import variables_from_filename


import data.data_utils as data_utils
from utils import FIGSIZE_1_1, FIGSIZE_1_3

import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

FIGSIZE = (3, 2.2)


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def exp0(path: Path, results_files="*corr_0.5*results.csv"):
    files = list(path.rglob(results_files))
    results = np.zeros((6, len(files), uv_params.num_eval))

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        x_values = df["x"]
        results[0][i] = df["baseline_mse"]
        results[1][i] = df["baseline_mean"]
        results[2][i] = df["baseline_std"]
        results[3][i] = df["improved_mse"]
        results[4][i] = df["improved_mean"]
        results[5][i] = df["improved_std"]

    # Plot sd improved / sd baseline
    fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    sd = results[5] / results[2]
    axs.plot(x_values, sd.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(sd, 2.5, axis=0),
        np.percentile(sd, 97.5, axis=0),
        alpha=0.25,
        label='$2.5-97.5$ percentile'
    )
    axs.legend()
    axs.set_xlabel('$\\mathbf{x}$')
    axs.set_ylabel('improved SD / baseline SD')
    plt.tight_layout()
    plt.savefig(f'img/kde_sd_{path.name}.pgf')

    fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    # breakpoint()
    mse = results[3] / results[0]
    axs.plot(x_values, mse.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(mse, 2.5, axis=0),
        np.percentile(mse, 97.5, axis=0),
        alpha=0.25,
        label='$2.5-97.5$ percentile'
    )
    axs.set_xlabel('$\\mathbf{x}$')
    axs.set_ylabel('improved MSE / baseline MSE')
    axs.legend()
    plt.tight_layout()
    plt.savefig(f'img/kde_mse_{path.name}.pgf')

#######################################################################


def _plot(axs, df, label, tp: str):
    if tp.lower() == 'sd':
        tp = 'std'
    axs.plot(df["x"], df[f"improved_{tp}".lower()] /
             df[f"baseline_{tp}".lower()], label=label)
    axs.legend()


def plot(files, save):
    results = np.zeros((6, len(files), uv_params.num_eval))


def exp1():
    labels = [
        f'$\\rho = 0.1$',
        f'$\\rho = 0.5$',
        f'$\\rho = 0.9$',
    ]

    files = [Path("estimates/kde_naive_ensemble/gaussian/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.1.results.csv"),
             Path("estimates/kde_naive_ensemble/gaussian/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.5.results.csv"),
             Path("estimates/kde_naive_ensemble/gaussian/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.9.results.csv")]

    for tp in ['SD', 'MSE']:
        _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        axs.set_xlabel('$\\mathbf{x}$')
        axs.set_ylabel(f'improved {tp} / baseline {tp}')
        for label, f in zip(labels, files):
            df = pd.read_csv(f)

            _plot(axs, df, label, tp)
        plt.tight_layout()
        plt.savefig(f'img/naive_kde_correlation_{tp}.pgf'.lower())


def exp2(root: Path):
    files = list(root.rglob("*p_edge_0.08.*corr_0.5*results.csv"))
    N = [100, 1000, 10000]
    for tp in ['SD', 'MSE']:
        _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        axs.set_xlabel('$\\mathbf{x}$')
        axs.set_ylabel(f'improved {tp} / baseline {tp}')
        for n in N:
            files = list(root.rglob(
                f"*p_edge_0.08.n_normal_{n}.n_edge_{n}.*corr_0.5*results.csv"))
            for f in files:
                df = pd.read_csv(f)

                _plot(
                    axs, df, '$N_\\textrm{norm}=N_\\textrm{event}=' + str(n) + '$', tp)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'img/naive_kde_observations_{tp}.pgf'.lower())


def exp3(root: Path):
    # Sort the files based on p_edge
    files = list(root.rglob(
        "*.n_normal_1000.n_edge_1000.corr_0.5.results.csv"))

    def f_sort(f: Path):
        p_edge = variables_from_filename(f.name)[0]
        return p_edge

    files.sort(key=f_sort)
    for tp in ['SD', 'MSE']:
        _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        axs.set_xlabel('$\\mathbf{x}$')
        axs.set_ylabel(f'improved {tp} / baseline {tp}')
        for f in files:
            p_edge, num_normal, num_edge, _ = variables_from_filename(f.name)
            if num_normal != num_edge:
                continue

            df = pd.read_csv(f)

            _plot(axs, df, '$p_\\textrm{event}=' + str(p_edge) + '$', tp)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'img/naive_kde_pedge_{tp}.pgf'.lower())


def exp4(path: Path):
    df = pd.read_csv(path)
    fig, axs = plt.subplots(1, 1, figsize=FIGSIZE_1_3)
    axs.plot(df['x'], df['improved_mse'] / df['baseline_mse'],
             label='MSE')
    axs.plot(df['x'], df['improved_std'] / df['baseline_std'],
             label='SD')
    axs.legend()
    axs.set_xlabel('$\\mathbf{x}$')
    plt.tight_layout()
    plt.savefig(f'img/{path.parts[-3]}.pgf')

    _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    axs.plot(df['x'], df['true'], label='true', linestyle='dotted')
    axs.plot(df['x'], df['baseline_mean'], label='baseline', alpha=0.5)
    axs.plot(df['x'], df['improved_mean'], label='improved', alpha=0.5)

    max_y = max(max(df['true']), max(df['baseline_mean']),
                max(df['improved_mean']))
    min_y = min(min(df['true']), min(df['baseline_mean']),
                min(df['improved_mean'])) - 0.05
    axs.set_xlabel('$\\mathbf{x}$')
    axs.set_ylabel(f'$f(x)$')
    if max_y > 1:
        axs.set_ylim(bottom=min_y, top=2)
    axs.legend()
    plt.tight_layout()
    plt.savefig(f'img/naive_kde_pred_{path.parts[-3]}.pgf')

def main():
    root_ensemble = Path('estimates/kde_naive_ensemble/gaussian')
    root_data_combination = Path('estimates/combined_data')
    exp0(root_ensemble)
    exp0(root_data_combination)

    # Experiment 1: Change correlation between x and y
    # distribution_roots = [
    #     Path("estimates/kde_naive_ensemble/gaussian/results/p_edge_0.02.n_normal_100.n_edge_100.corr_0.1.results.csv"),
    #     Path("estimates/kde_combined_estimator/bivariate_guassian_b"),
    #     Path("estimates/kde_combined_estimator/bivariate_gaussian_c"),
    # ]
    exp1()

    # Experiment 2: Change number of observations
    # root = Path("estimates/kde_combined_estimator/bivariate_guassian_b/results")
    exp2(root_ensemble)

    # Experiment 3: Change p_event
    exp3(root_ensemble)

    # Experiment 4: Beta and gumbell distribution
    # results_files = '*p_edge_0.04.n_normal_1000.n_edge_1000.results.csv'
    # results_files = '*results.csv'
    # exp0(Path('estimates/kde_combined_estimator/beta_a'), results_files)
    # exp0(Path('estimates/kde_combined_estimator/gumbel_a'), results_files)
    exp4(Path('estimates/naive_ensemble/t/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.5.results.csv'))
    exp4(Path('estimates/naive_ensemble/beta/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.5.results.csv'))
    exp4(Path('estimates/naive_ensemble/gumbel/results/p_edge_0.08.n_normal_1000.n_edge_1000.corr_0.5.results.csv'))


if __name__ == "__main__":
    main()

