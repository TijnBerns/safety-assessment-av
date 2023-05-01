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


import utils
from config import Config as cfg
import torch
from estimator import NNEstimator, CombinedDataPipeline



def main():
    device, _ = utils.set_device()
    pattern = f"layers_{cfg.nn_num_hidden_layers}.neurons_{cfg.nn_num_hidden_nodes}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
    CombinedDataPipeline().run_pipeline(NNEstimator(), root=cfg.path_estimates / 'nn_approach', device=device, pattern=pattern)


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    main()
    