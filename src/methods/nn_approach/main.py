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
from config import UVParameters as uv_params
import torch
from estimator import NNEstimator, CombinedData



def main():
    device, _ = utils.set_device()
    pattern = f"layers_{uv_params.nn_num_hidden_layers}.neurons_{uv_params.nn_num_hidden_nodes}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
    CombinedData().run_pipeline(NNEstimator(), root=uv_params.path_estimates / 'nn_approach', device=device, pattern=pattern)


if __name__ == "__main__":
    torch.manual_seed(uv_params.seed)
    main()
    