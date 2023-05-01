from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
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
    p5 = np.zeros_like(true)
    p95 = np.zeros_like(true)
    for e in range(len(true)):
        se = np.square(estimates[e] - true[e])
        mse[e] = np.mean(se)
        p5[e] = np.percentile(se, 2.5)
        p95[e] = np.percentile(se, 97.5)
        

    return mse, mean, std, p5, p95


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
            'true': baseline_df['true'],
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
