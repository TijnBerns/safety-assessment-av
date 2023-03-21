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
from estimator import KDE_Estimator, combined_data_pipline, combined_estimator_pipeline


def main(type: str = 'asd'):
    if type == 'combined':
        root = cfg.path_estimates / 'kde_combined_data'
        combined_data_pipline(KDE_Estimator(), KDE_Estimator(), root)
    else: 
        root = cfg.path_estimates / 'kde_combined_estimator'
        combined_estimator_pipeline(KDE_Estimator(), KDE_Estimator(), KDE_Estimator(), root)
    
if __name__ == "__main__":
    main()
