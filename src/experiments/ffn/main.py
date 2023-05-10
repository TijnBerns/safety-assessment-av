import sys
sys.path.append('src')

import utils
from config import UVParameters as uv_params
import torch
import estimator



def main():
    device, _ = utils.set_device()
    root = uv_params.path_estimates / 'ffn'
    nn_estimator = estimator.NNEstimator(uv_params.nn_num_hidden_nodes, uv_params.nn_num_hidden_layers, 1, 1)
    estimator.UnivariatePipeline(estimator.CombinedData).run_pipeline(estimator.KDEEstimator(), root)


if __name__ == "__main__":
    torch.manual_seed(uv_params.seed)
    main()
    