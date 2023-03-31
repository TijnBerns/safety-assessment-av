from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from tqdm import tqdm
from config import Config as cfg
from utils import save_csv, variables_from_filename


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def evaluate(dataframe: pd.DataFrame):
    # Collect all estimates in single ndarray
    run_cols = [col for col in dataframe if col.startswith('run')]
    estimates = dataframe[run_cols].to_numpy()

    # The true pdf values
    true = dataframe['true'].to_numpy()

    # Compute various metrics
    mean = np.mean(estimates.T, axis=0)
    var = np.std(estimates.T, axis=0)

    # Compute mean squared error at every point
    mse = np.zeros_like(true)
    for e in range(len(estimates)):
        se = np.square(estimates[e] - true[e])
        mse[e] = np.mean(se)

    return mse, mean, var


def plot_estimate(true, estimates, x_values):
    _, axs = plt.subplots(1, 1)

    # Plot predictions
    axs.plot(x_values, np.mean(estimates.T, axis=0))
    upper = np.percentile(estimates.T, 97.5, axis=0)
    lower = np.percentile(estimates.T, 2.5, axis=0)
    axs.plot(x_values, upper, color='tab:blue')
    axs.plot(x_values, lower, color='tab:blue')
    axs.fill_between(x_values, upper, lower, color='tab:blue', alpha=0.5)
    axs.plot(x_values, true, color='tab:orange')
    plt.savefig('estimate')
    pass


def plot_var(baseline_var, improved_var, x_values, save=None):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(x_values, baseline_var, label='baseline variance')
    axs[0].plot(x_values, improved_var, label='improved variance')
    axs[0].legend()
    axs[1].plot(x_values, improved_var / baseline_var,
                label='improved / baseline')
    axs[1].legend()
    axs[1].set_ylim(bottom=-0.5, top=2)
    plt.title(save.name[:-4])
    plt.savefig(save)
    pass


def evaluation_pipeline(path: Path):
    results = []
    # _, ax = plt.subplots(1, 1, figsize=(5,5))
    # ax.set_ylim(bottom=-0.5, top=2)
    # ax.set_xlim(left=-6, right=6)
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir():
            continue
        try:
            # This fails if not all columns in the cvs have equal rows
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
            baseline_sse, baseline_var = evaluate(baseline_df)
            improved_sse, improved_var = evaluate(improved_df)
        except Exception as e:
            print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
            continue

        # Extract variables from file name
        p_edge, n_normal, n_edge = variables_from_filename(f.name)

        # plot_var(baseline_var, improved_var, baseline_df['x'], save=Path('img')/ (f.name + '_var.png'))

        # Add the results to the dict
        results.append({
            'p_edge': p_edge,
            'n_normal': n_normal,
            'n_edge': n_edge,
            'baseline_sse': baseline_sse.mean(),
            'improved_sse': improved_sse.mean(),
            'delta_sse': improved_sse.mean() - baseline_sse.mean(),
            'baseline_var': baseline_var.mean(),
            "improved_var": improved_var.mean(),
            'delta_var': improved_var.mean() - baseline_var.mean()
        })

    # plt.savefig('test')
    save_csv(path / 'results.csv', pd.DataFrame(results))
    

if __name__ == "__main__":
    path = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates/kde_combined_estimator/bivariate_gaussian_c/p_edge_0.1.n_normal_500.n_edge_500/p_edge_0.1.n_normal_500.n_edge_500.improved.csv')
    df = pd.read_csv(path)
    mse, mean, var = evaluate(df)
    x_values = df['x']
    
    fig, axs = plt.subplots(1,1)
    axs.plot(x_values, mean)
    axs.fill_between(x_values, mean + var, mean - var, alpha=0.5)
    axs.plot(x_values, df['true'])
    plt.show()
    
    # fig, axs = plt.subplots(1,1)
    # run_cols = [col for col in df if col.startswith('run')]
    # estimates = df[run_cols].to_numpy()
    # axs.plot(x_values, estimates.mean(axis=1), label='estimate')
    # axs.plot(x_values, df['true'], label='true')
    # axs.legend()
    # plt.savefig('pdf')