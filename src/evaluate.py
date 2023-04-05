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
    std = np.std(estimates.T, axis=0)

    # Compute mean squared error at every point
    mse = np.zeros_like(true)
    for e in range(len(estimates)):
        se = np.square(estimates[e] - true[e])
        mse[e] = np.mean(se)

    return mse, mean, std


# def plot_estimate(true, estimates, x_values):
#     _, axs = plt.subplots(1, 1)

#     # Plot predictions
#     axs.plot(x_values, np.mean(estimates.T, axis=0))
#     upper = np.percentile(estimates.T, 97.5, axis=0)
#     lower = np.percentile(estimates.T, 2.5, axis=0)
#     axs.plot(x_values, upper, color='tab:blue')
#     axs.plot(x_values, lower, color='tab:blue')
#     axs.fill_between(x_values, upper, lower, color='tab:blue', alpha=0.5)
#     axs.plot(x_values, true, color='tab:orange')
#     plt.savefig('estimate')
#     pass


# def plot_var(baseline_var, improved_var, x_values, save=None):
#     _, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].plot(x_values, baseline_var, label='baseline variance')
#     axs[0].plot(x_values, improved_var, label='improved variance')
#     axs[0].legend()
#     axs[1].plot(x_values, improved_var / baseline_var,
#                 label='improved / baseline')
#     axs[1].legend()
#     axs[1].set_ylim(bottom=-0.5, top=2)
#     plt.title(save.name[:-4])
#     plt.savefig(save)
#     pass


def evaluation_pipeline(path: Path):
    results = []

    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir() or f.name.find('results') != -1:
            continue
        try:
            # This fails if not all columns in the cvs have equal rows
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
            baseline_mse, baseline_mean, baseline_std = evaluate(baseline_df)
            improved_mse, improved_mean, improved_std = evaluate(improved_df)
        except Exception as e:
            print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
            continue

        # Extract variables from file name
        p_edge, n_normal, n_edge = variables_from_filename(f.name)
        
        results_path = path / 'results' / (f.name + '.results.csv')
        
        save_csv(results_path, pd.DataFrame({
            'x': baseline_df['x'],
            'baseline_mse': baseline_mse,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'improved_mse': improved_mse,
            'improved_mean': improved_mean,
            'improved_std': improved_std,
            }))
        
        
        
        

    

if __name__ == "__main__":
    for path in Path('estimates/kde_combined_estimator').glob('*'):
        if not path.is_dir() or path.name.find('result') != -1:
            continue

        evaluation_pipeline(path)
    
    # df = pd.read_csv('estimates/kde_combined_estimator/bivariate_guassian_a/results.csv')
    # breakpoint()
    
    # df = pd.read_csv(path)
    # mse, mean, var = evaluate(df)
    # x_values = df['x']
    
    # fig, axs = plt.subplots(1,1)
    # axs.plot(x_values, mean)
    # axs.fill_between(x_values, mean + var, mean - var, alpha=0.5)
    # axs.plot(x_values, df['true'])
    # plt.show()
    
    # fig, axs = plt.subplots(1,1)
    # run_cols = [col for col in df if col.startswith('run')]
    # estimates = df[run_cols].to_numpy()
    # axs.plot(x_values, estimates.mean(axis=1), label='estimate')
    # axs.plot(x_values, df['true'], label='true')
    # axs.legend()
    # plt.savefig('pdf')