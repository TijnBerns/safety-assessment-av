import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import pandas as pd
from itertools import product
from tqdm import tqdm
from pathlib import Path
import json

import data_utils
from config import Config as cfg
from estimator import KDE_Estimator_a, KDE_Estimator, combined_estimation_pipline
from utils import save_csv

def improved_estimation_pipeline(root: Path):
    # Initialize estimator
    kde_estimator = KDE_Estimator_a()

    # Define evaluation interval
    x_values = cfg.evaluation_interval

    for distribution_str, distribution in cfg.distributions.items():
        true = [cfg.single_distributions[distribution_str].pdf(
            x) for x in x_values]

        for p_edge, num_normal, num_edge in product(cfg.p_edge, cfg.num_normal, cfg.num_edge):
            thresholds = {}

            # Initialize dicts to store results
            baseline_estimates = {"x": x_values, "true": true}
            improved_estimates = {"x": x_values, "true": true}

            for run in tqdm(range(cfg.num_estimates), desc=f'{distribution_str}: norm={num_normal} edge={num_edge} p_edge={p_edge}'):
                # Generate data
                normal_data, edge_data, threshold = data_utils.generate_data(
                    distribution, p_edge, num_normal, num_edge)

                # Filter edge and normal data
                normal_filtered, edge_filtered = data_utils.filter_data(normal_data, edge_data, threshold)

                # Fit data to estimators
                kde_estimator.fit_baseline(normal_data[:, 0])
                kde_estimator.fit_normal(normal_filtered[:, 0])
                kde_estimator.fit_edge(edge_filtered[:, 0])

                # Obtain estimates
                baseline_estimates[f'run_{run}'] = kde_estimator.estimate(
                    x_values=x_values, estimate_fn=kde_estimator.baseline_estimate)
                improved_estimates[f'run_{run}'] = kde_estimator.estimate(
                    x_values=x_values, estimate_fn=kde_estimator.improved_estimate,
                    c=threshold, p_normal=1-p_edge, p_edge=p_edge)

                # Store results
                parent = root / distribution_str / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}'
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.baseline.csv',
                         df=pd.DataFrame(baseline_estimates))
                save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.improved.csv',
                         df=pd.DataFrame(improved_estimates))

                thresholds[f"run_{run}"] = threshold
                with open(parent / 'thresholds.json', 'w') as f:
                    json.dump(thresholds, f, indent=2)
    

def main(type: str = 'combined'):
    if type == 'combined':
        root = cfg.path_estimates / 'kde_combined'
        combined_estimation_pipline(KDE_Estimator(), KDE_Estimator(), root)
    else: 
        root = cfg.path_estimates / 'kde'
        improved_estimation_pipeline(root)
    


if __name__ == "__main__":
    main()
