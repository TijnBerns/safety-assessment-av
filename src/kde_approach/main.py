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

from config import Config as cfg
from estimator import KDEEstimator, NaiveEnsemblePipeline, CombinedDataPipeline
import click

@click.command()
@click.option('--type', '-t', default='asd', type=str)
def main(type: str = 'asd'):
    if type == 'combined':
        root = cfg.path_estimates / 'kde_combined_data'
        CombinedDataPipeline().run_pipeline(KDEEstimator(), root)
    else: 
        root = cfg.path_estimates / 'kde_combined_estimator'
        NaiveEnsemblePipeline().run_pipeline(KDEEstimator(), root)
    
if __name__ == "__main__":
    main()
