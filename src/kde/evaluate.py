import sys 
sys.path.append('src')

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import save_csv, variables_from_filename, load_json_as_df, load_json
import json
import click
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utils
from typing import List, Tuple
import scipy.stats
import geometry


test_a = [[0, 1], [3,4]]
test_b = [[1,2],[3,5],[6,7]]

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def evaluate_pdf(dataframe: pd.DataFrame):
    """Evaluate the estimated pdf of 1D data
    """
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
    p5 = np.zeros_like(true)
    p95 = np.zeros_like(true)
    for e in range(len(true)):
        se = np.square(estimates[e] - true[e])
        mse[e] = np.mean(se)
        p5[e] = np.percentile(se, 2.5)
        p95[e] = np.percentile(se, 97.5)

    return mse, mean, std, p5, p95


# def evaluate_epsilon_support(data: dict):
#     """Evaluate the estimated epsilon support of 1D-data
#     """
#     true = geometry.MultiInterval(data['true'])
    
#     keys = [k for k in data.keys() if k.startswith('run')]
#     estimates = np.zeros(len(keys))
#     absolute_error = np.zeros_like(estimates)
#     jacobian_distance = np.zeros_like(estimates)
#     for i, k in enumerate(keys):
#         estimate = geometry.MultiInterval()
#         estimates[i] = np.abs(data[k][0][1])
#         absolute_error[i] = np.abs(data[k][0][1] - true[0][1])
#         jacobian_distance[i] = compute_jacobian_distance(true[i], data[k])

#     return true, estimates.mean(), estimates.std(), absolute_error.mean(), absolute_error.std(), jacobian_distance.mean(), jacobian_distance.std()

def evaluate_epsilon_support(data: dict):
    """Evaluate the estimated epsilon support of 1D-data
    """

    true = geometry.MultiInterval(data['true'])
    keys = [k for k in data.keys() if k.startswith('run')]
    estimates = np.zeros(len(keys))
    absolute_error = np.zeros_like(estimates)
    jacobian_distance = np.zeros_like(estimates)
    for i, k in enumerate(keys):
        jacobian_distance[i] = true.jaccard_distance(data[k])
        
        
    return true.as_tuples, jacobian_distance.mean(), jacobian_distance.std()


@click.command()
@click.option('--path', '-t', type=Path)
def uv_evaluation_pipeline(path: Path):
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir() or f.name.find('results') != -1:
            continue
        # try:
            # This fails if not all columns in the cvs have equal rows
        baseline_df = pd.read_csv(f / (f.name + '.baseline_pdf.csv'))
        improved_df = pd.read_csv(f / (f.name + '.improved_pdf.csv'))
        baseline_eps = load_json(f / (f.name + '.baseline_eps.json'))
        improved_eps = load_json(f / (f.name + '.improved_eps.json'))
        # true, baseline_estimate_mean, baseline_estimate_std, baseline_error_mean, baseline_error_std, baseline_jacobian_mean, baseline_jacobian_std = evaluate_epsilon_support(
        #     baseline_eps)
        # _, improved_estimate_mean, improved_estimate_std, improved_error_mean, improved_error_std, improved_jacobian_mean, improved_jacobian_std = evaluate_epsilon_support(
        #     improved_eps)
        true, baseline_jaccard_mean, baseline_jaccard_std = evaluate_epsilon_support(baseline_eps)
        _, improved_jaccard_mean, improved_jaccard_std = evaluate_epsilon_support(improved_eps)
        baseline_mse, baseline_mean, baseline_std, _, _ = evaluate_pdf(
            baseline_df)
        improved_mse, improved_mean, improved_std, _, _ = evaluate_pdf(
            improved_df)

        # except Exception as e:
        #     print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
        #     continue

        utils.save_csv(path / 'results' / (f.name + '.results.csv'), pd.DataFrame({
            'x': baseline_df['x'],
            'true': baseline_df['true'],
            'baseline_mse': baseline_mse,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'improved_mse': improved_mse,
            'improved_mean': improved_mean,
            'improved_std': improved_std,
        }))

        utils.save_json(path / 'results' / (f.name + '.eps.results.json'), {
            # 'true_eps': true,
            # 'baseline_estimate_mean': baseline_estimate_mean,
            # "baseline_estimate_std": baseline_estimate_std,
            # 'baseline_error_mean': baseline_error_mean,
            # 'baseline_error_std': baseline_error_std,
            # 'baseline_jacobian_mean': baseline_jacobian_mean,
            # "baseline_jacobian_std": baseline_jacobian_std,
            # 'improved_estimate_mean': improved_estimate_mean,
            # "improved_estimate_std": improved_estimate_std,
            # 'improved_error_mean': improved_error_mean,
            # 'improved_error_std': improved_error_std,
            # 'improved_jacobian_mean': improved_jacobian_mean,
            # "improved_jacobian_std": improved_jacobian_std,
            'true_eps': true,
            'baseline_jaccard_mean': baseline_jaccard_mean,
            'baseline_jaccard_std': baseline_jaccard_std,
            'improved_jaccard_mean': improved_jaccard_mean,
            'improved_jaccard_std': improved_jaccard_std

        })


@click.command()
@click.option('--path', '-p', type=Path)
def mv_evaluation_pipeline(path: Path):
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir() or f.name.find('results') != -1:
            continue
        try:
            # This fails if not all columns in the cvs have equal rows
            baseline_df = load_json_as_df(f / (f.name + '.baseline_pdf.json'))
            improved_df = load_json_as_df(f / (f.name + '.improved_pdf.json'))
            baseline_mse, baseline_mean, baseline_std, _, _ = evaluate_pdf(
                baseline_df)
            improved_mse, improved_mean, improved_std, _, _ = evaluate_pdf(
                improved_df)
        except Exception as e:
            print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
            continue

        results_path = path / 'results' / (f.name + '.results.csv')

        save_csv(results_path, pd.DataFrame({
            'x': baseline_df['x'],
            'true': baseline_df['true'],
            'baseline_mse': baseline_mse,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'improved_mse': improved_mse,
            'improved_mean': improved_mean,
            'improved_std': improved_std,
        }))


if __name__ == "__main__":
    # mv_evaluation_pipeline()
    uv_evaluation_pipeline()
    # temp('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/test.json')

