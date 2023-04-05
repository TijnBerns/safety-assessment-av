import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from evaluate import evaluate
from config import Config as cfg
from utils import variables_from_filename
from evaluate import evaluation_pipeline

import tikzplotlib

import matplotlib
matplotlib.use('pgf')
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "10"

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def exp0(path: Path):
    files = list(path.rglob("*results.csv"))
    results = np.zeros((6, len(files), cfg.num_eval))

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        x_values = df["x"]
        results[0][i] = df["baseline_mse"]
        results[1][i] = df["baseline_mean"]
        results[2][i] = df["baseline_std"]
        results[3][i] = df["improved_mse"]
        results[4][i] = df["improved_mean"]
        results[5][i] = df["improved_std"]

    # Plot std improved / std baseline
    fig, axs = plt.subplots(1, 1, figsize=(3.5,2.8))
    std = results[5] / results[2]
    axs.plot(x_values, std.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(std, 2.5, axis=0),
        np.percentile(std, 97.5, axis=0),
        alpha=0.25,
        label = '95\% confidence interval'
    )
    axs.legend()
    axs.set_xlabel('$x$')
    axs.set_ylabel('improved std / baseline std')
    plt.tight_layout()
    plt.savefig('kde_std.pgf')

    fig, axs = plt.subplots(1, 1, figsize=(3.5,2.8))
    mse = results[3] / results[0]
    axs.plot(x_values, mse.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(mse, 2.5, axis=0),
        np.percentile(mse, 97.5, axis=0),
        alpha=0.25,
        label = '95\% confidence interval'
    )
    axs.set_xlabel('$x$')
    axs.set_ylabel('improved mse / baseline mse')
    axs.legend()
    plt.tight_layout()
    plt.savefig('kde_mse.pgf')

#######################################################################
def _plot(axs, df, label):
    axs[0].plot(df["x"], df["improved_std"] / df["baseline_std"], label=label)
    axs[0].legend()
    axs[1].plot(df["x"], df["improved_mse"] / df["baseline_mse"], label=label)
    axs[1].legend()


def exp1(distribution_roots):
    labels = {
        "bivariate_guassian_a": f'rho = 0.1',
        "bivariate_guassian_b": f'rho = 0.5',
        "bivariate_gaussian_c": f'rho = 0.9',
    }

    fig, axs = plt.subplots(1, 2, figsize=(4,2))

    for f in distribution_roots:
        df = pd.read_csv(            
                         f / "results/p_edge_0.08.n_normal_1000.n_edge_1000.results.csv"
        )
        _plot(axs, df, labels[f.name])

    plt.savefig('kde_exp1.pgf')


def exp2(root: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7,4))
    files = list(root.rglob("*p_edge_0.08.*.csv"))

    for f in root.rglob("*p_edge_0.08.*.csv"):
        _, num_normal, num_edge = variables_from_filename(f.name)
        if num_normal != num_edge:
            continue

        df = pd.read_csv(f)
        _plot(axs, df, f'{num_normal} observations')
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save('kde_exp2.tex')


def exp3(root: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7,4))
    files = list(root.rglob("*.n_normal_1000.n_edge_1000.results.csv"))

    # Sort the files based on p_edge
    def f_sort(f: Path):
        p_edge, _, _ = variables_from_filename(f.name)
        return p_edge

    files.sort(key=f_sort)

    for f in files:
        p_edge, num_normal, num_edge = variables_from_filename(f.name)
        if num_normal != num_edge:
            continue

        df = pd.read_csv(f)
        _plot(axs, df, f'p_edge={p_edge}')
        


if __name__ == "__main__":
    root = Path('estimates/kde_combined_estimator/bivariate_guassian_b')
    exp0(root)

    # Experiment 1: Change correlation between x and y
    distribution_roots = [
        Path("estimates/kde_combined_estimator/bivariate_guassian_a"),
        Path("estimates/kde_combined_estimator/bivariate_guassian_b"),
        Path("estimates/kde_combined_estimator/bivariate_gaussian_c"),
    ]
    exp1(distribution_roots)

    # Experiment 2: Change number of observations
    root = Path("estimates/kde_combined_estimator/bivariate_guassian_b/results")
    exp2(root)

    # Experiment 3: Change p_event
    exp3(root)


# cmap = plt.get_cmap('viridis')
# # colors = cmap(np.linspace(0, 1, len(cfg.p_edge)))
# # bounds = cfg.p_edge
# # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
# # p_edge_colors = {str(k): v for k, v in zip(cfg.p_edge, colors)}


# def plot_diff_grouped_by_n(results_df: pd.DataFrame, save:str):
#     _, axs = plt.subplots(1,2, figsize=(10,5) )
#     axs[0].set_ylabel('improved var / baseline var')
#     axs[1].set_ylabel('improved error / baseline error')

#     for ax in axs:
#         ax.set_xlabel('p_edge')

#     for n in pd.unique(results_df['n_edge']):
#         p = results_df.where((results_df['n_edge'] == n) & (results_df['n_normal'] == n)).dropna()
#         p = p.sort_values(by='p_edge')

#         axs[0].plot(p['p_edge'], p['improved_var'] - p['baseline_var'], label=f'n={n}')
#         axs[1].plot(p['p_edge'], p['improved_sse'] - p['baseline_sse'], label=f'n={n}')
#     plt.tight_layout()
#     plt.savefig(save, bbox_inches = "tight")


# def plot_diff_grouped_by_p_edge(results_df: pd.DataFrame, save:str):
#     _, axs = plt.subplots(1,2, figsize=(10,5) )
#     axs[0].set_ylabel('improved var - baseline var')
#     axs[1].set_ylabel('improved error - baseline error')

#     for ax in axs:
#         ax.set_xlabel('n')

#     for p_edge in pd.unique(results_df['p_edge']):
#         p = results_df.where((results_df['n_edge'] == results_df['n_normal']) & (results_df['p_edge'] == p_edge)).dropna()
#         p = p.sort_values(by='n_normal')

#         axs[0].plot(p['n_normal'], p['improved_var'] - p['baseline_var'], label=f'p_edge={p_edge}')
#         axs[1].plot(p['n_normal'], p['improved_sse'] - p['baseline_sse'], label=f'p_edge={p_edge}')
#     plt.tight_layout()
#     plt.savefig(save, bbox_inches = "tight")


# def plot_diff(path: Path, save):
#     _, axs = plt.subplots(1,2, figsize=(10,5))
#     plt.tight_layout()

#     # Set limits and x-labels
#     for ax in axs:
#         ax.set_ylim(bottom=-0.5, top=2)
#         # ax.set_xlim(left=-5, right=5)
#         ax.set_xlabel('x')

#     axs[0].set_ylabel('improved var / baseline var')
#     axs[1].set_ylabel('improved error / baseline error')

#     for f in tqdm(list(path.glob('**/*'))):
#         if not f.is_dir():
#             continue
#         try:
#             # This fails if not all columns in the cvs have equal rows
#             baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
#             improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
#             baseline_sse, baseline_var = evaluate(baseline_df)
#             improved_sse, improved_var = evaluate(improved_df)
#         except Exception as e:
#             print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
#             continue

#         # Extract variables from file name
#         p_edge, _, _ = variables_from_filename(f.name)

#         axs[0].plot(baseline_df['x'], improved_var/baseline_var, alpha =0.5, color=p_edge_colors[p_edge])
#         axs[1].plot(baseline_df['x'], improved_sse/baseline_sse, alpha =0.5, color=p_edge_colors[p_edge])

#     plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='p_edge')
#     plt.tight_layout()
#     plt.savefig(save, bbox_inches = "tight")


# def plot_pdf(path: Path, save):
#     _, ax = plt.subplots(1,2, figsize=(10,5))
#     plt.tight_layout()

#     # Set limits and x-labels
#     ax[0].set_xlabel('x')
#     ax[1].set_xlabel('x')
#     ax[0].set_title('baseline')
#     ax[1].set_title('improved')


#     for f in tqdm(list(path.glob('**/*'))):
#         if not f.is_dir():
#             continue
#         try:
#             # This fails if not all columns in the cvs have equal rows
#             baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
#             improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
#         except Exception as e:
#             print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
#             continue

#         # Extract variables from file name
#         p_edge, _, _ = variables_from_filename(f.name)

#         run_cols = [col for col in baseline_df if col.startswith('run')]
#         baseline_estimates = baseline_df[run_cols].to_numpy()

#         run_cols = [col for col in improved_df if col.startswith('run')]
#         improved_estimates = improved_df[run_cols].to_numpy()


#         ax[0].plot(baseline_df['x'], baseline_estimates.mean(axis=1), alpha =0.5, color=p_edge_colors[p_edge])
#         ax[1].plot(baseline_df['x'], improved_estimates.mean(axis=1), alpha =0.5, color=p_edge_colors[p_edge])

#     plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='p_edge')
#     plt.tight_layout()
#     plt.savefig(save, bbox_inches = "tight")
