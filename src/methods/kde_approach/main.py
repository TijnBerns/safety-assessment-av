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

from config import UVParameters as uv_params
import estimator
import click

@click.command()
@click.option('--type', '-t', default='asd', type=str)
def main(type: str = 'asd'):
    if type == 'combined':
        root = uv_params.path_estimates / 'kde_combined_data'
        estimator.CombinedData().run_pipeline(estimator.KDEEstimator(), root)
    elif type == 'naive_ensemble': 
        root = uv_params.path_estimates / 'kde_combined_estimator'
        estimator.UnivariatePipeline(estimator.NaiveEnsemble).run_pipeline(estimator.KDEEstimator(), root)
    else:
        root = uv_params.path_estimates / 'test'
        estimator.MultivariatePipeline(estimator.NaiveEnsemble).run_pipeline(estimator.KDEEstimator(), root)
    
if __name__ == "__main__":
    main()
