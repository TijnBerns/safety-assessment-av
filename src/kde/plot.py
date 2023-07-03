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
    axs.set_xlabel('$x$')
    axs.set_ylabel('improved SD / baseline SD')
    plt.tight_layout()
    plt.savefig(f'img/kde_sd_{path.name}.pgf')

    fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    breakpoint()
    mse = results[3] / results[0]
    axs.plot(x_values, mse.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(mse, 2.5, axis=0),
        np.percentile(mse, 97.5, axis=0),
        alpha=0.25,
        label='$2.5-97.5$ percentile'
    )
    axs.set_xlabel('$x$')
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
        axs.set_xlabel('$x$')
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
        axs.set_xlabel('$x$')
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
        axs.set_xlabel('$x$')
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
    fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    axs.plot(df['x'], df['improved_mse'] / df['baseline_mse'],
             label='improved MSE / baseline MSE')
    axs.plot(df['x'], df['improved_std'] / df['baseline_std'],
             label='improved SD / baseline SD')
    axs.legend()
    axs.set_xlabel('$x$')
    plt.tight_layout()
    plt.savefig(f'img/naive_kde_{path.parts[-3]}.pgf')

    _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    axs.plot(df['x'], df['true'], label='true', linestyle='dotted')
    axs.plot(df['x'], df['baseline_mean'], label='baseline', alpha=0.5)
    axs.plot(df['x'], df['improved_mean'], label='improved', alpha=0.5)

    max_y = max(max(df['true']), max(df['baseline_mean']),
                max(df['improved_mean']))
    min_y = min(min(df['true']), min(df['baseline_mean']),
                min(df['improved_mean'])) - 0.05
    axs.set_xlabel('$x$')
    axs.set_ylabel(f'$f(x)$')
    if max_y > 1:
        axs.set_ylim(bottom=min_y, top=2)
    axs.legend()
    plt.tight_layout()
    plt.savefig(f'img/naive_kde_pred_{path.parts[-3]}.pgf')

    # axs.plot(x_values, sd.mean(axis=0), label='mean')
    # axs.fill_between(
    #     x_values,
    #     np.percentile(sd, 2.5, axis=0),
    #     np.percentile(sd, 97.5, axis=0),
    #     alpha=0.25,
    #     label = '$2.5-97.5$ percentile'
    # )
    # axs.legend()
    # axs.set_xlabel('$x$')
    # axs.set_ylabel('improved sd / baseline sd')
    # plt.tight_layout()
    # plt.savefig(f'img/kde_sd_{path.name}.pgf')


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
    # exp1()

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
    exp4(Path('estimates/kde_combined_estimator/beta_a/results/p_edge_0.04.n_normal_1000.n_edge_1000.results.csv'))
    exp4(Path('estimates/kde_combined_estimator/gumbel_a/results/p_edge_0.04.n_normal_1000.n_edge_1000.results.csv'))


if __name__ == "__main__":
    main()


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
