from estimator import KDE_Estimator
import data_utils
from global_config import GlobalConfig as cfg
import sklearn.metrics
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from itertools import product
from pathlib import Path
import json


def save_csv(path: Path, df: pd.DataFrame):
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    df.to_csv(path)


if __name__ == "__main__":
    # Initialize estimator
    kde_estimator = KDE_Estimator()

    # Define evaluation interval
    x_values = np.linspace(-10, 10, 1000)

    for distribution_str, distribution in cfg.distributions.items():
        true = [distribution.pdf(x) for x in x_values]

        # Initialize dicts to store results
        baseline_estimates = {"x": x_values, "true": true}
        improved_estimates = {"x": x_values, "true": true}
        thresholds = {}

        for num_normal, num_edge, p_edge in product(cfg.num_normal, cfg.num_edge, cfg.p_edge):
            for run in range(cfg.num_estimates):
                # Generate data
                normal_data, edge_data, threshold = data_utils.generate_data(
                    distribution, p_edge, num_normal, num_edge)

                # Filter edge and normal data
                normal_filtered, edge_filtered = data_utils.filter_data(
                    normal_data, edge_data, threshold)

                # Fit data to estimators
                kde_estimator.fit_baseline(normal_data)
                kde_estimator.fit_normal(normal_filtered)
                kde_estimator.fit_edge(edge_data)

                # Obtain estimates
                baseline_estimates[f'run_{run}'] = kde_estimator.estimate(
                    x_values=x_values, estimate_fn=kde_estimator.baseline_estimate)
                improved_estimates[f'run_{run}'] = kde_estimator.estimate(
                    x_values=x_values, estimate_fn=kde_estimator.improved_estimate,
                    c=threshold, p_normal=1-p_edge, p_edge=p_edge)

                # Store results
                parent = cfg.path_estimates / distribution_str / \
                    f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}'
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.baseline.csv',
                         df=pd.DataFrame(baseline_estimates))
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.improved.csv',
                         df=pd.DataFrame(improved_estimates))

                thresholds[f"run_{run}"] = threshold
                with open(parent / 'thresholds.json', 'w') as f:
                    json.dump(thresholds, f, indent=2)
