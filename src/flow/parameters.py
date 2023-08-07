""" 
Module containing hyperparamter configurations of each of the considered UCI datasets
"""

import sys

sys.path.append("src")

from dataclasses import dataclass
from data.base import CustomDataset
from data.power import Power
from data.miniboone import MiniBoone
from data.gas import Gas
from data.hepmass import Hepmass
from data.bsds300 import BSDS300Dataset


@dataclass
class Parameters:
    # Num workers
    num_workers = 2
    # The batch size
    batch_size = 512
    # The number of training steps in both training stages
    training_steps = 100_000
    # The learning rate used during both training stages
    learning_rate = 0.0005
    # The probability of observing an event in the normal data
    p_event = 0.08
    # Number of blocks to use in flowsa
    # Number of hidden features to use in coupling/autoregressive nets
    hidden_features = 256
    # Box is on [-bound, bound]^2
    tail_bound = 3
    # Number of bins to use for piecewise transforms.
    num_bins = 8
    # The number of flow steps
    num_flow_steps = 20
    # Number of blocks to use in coupling/autoregressive nets.
    num_transform_blocks = 2
    # Whether to use batch norm in coupling/autoregressive nets.
    use_batch_norm = False
    # Dropout probability for coupling/autoregressive nets.
    dropout_probability = 0.2
    # Whether to unconditionally transform 'identity' features in coupling layer.
    apply_unconditional_transform = True
    # Logging interval used by pytorch lightning trainer
    logging_interval = 50


@dataclass
class PowerParameters(Parameters):
    batch_size = 512
    num_flow_steps = 10
    num_transform_blocks = 2
    hidden_features = 256
    num_bins = 8
    dropout_probability = 0.0


@dataclass
class GasParameters(Parameters):
    batch_size = 512
    num_flow_steps = 10
    num_transform_blocks = 2
    hidden_features = 256
    num_bins = 8
    dropout_probability = 0.1


@dataclass
class HepmassParameters(Parameters):
    batch_size = 256
    num_flow_steps = 20
    num_transform_blocks = 1
    hidden_features = 128
    num_bins = 8
    dropout_probability = 0.2


@dataclass
class MiniBooneParameters(Parameters):
    learning_rate_stage_1 = 0.0003
    learning_rate_stage_1 = 0.0003
    batch_size = 128
    num_flow_steps = 10
    num_transform_blocks = 1
    hidden_features = 32
    num_bins = 4
    dropout_probability = 0.2


@dataclass
class BSDS300Parameters(Parameters):
    learning_rate_stage_1 = 0.0005
    learning_rate_stage_1 = 0.0005
    batch_size = 512
    num_flow_steps = 20
    num_transform_blocks = 1
    hidden_features = 128
    num_bins = 8
    dropout_probability = 0.2


def get_dataset(dataset: str) -> CustomDataset:
    if dataset == "power":
        return Power
    elif dataset == "hepmass":
        return Hepmass
    elif dataset == "gas":
        return Gas
    elif dataset == "bsds300":
        return BSDS300Dataset
    elif dataset == "miniboone":
        return MiniBoone
    else:
        print(f"Got unexpected dataset string: {dataset}")
        raise ValueError


def get_parameters(dataset: str) -> Parameters:
    if dataset == "power":
        return PowerParameters
    elif dataset == "hepmass":
        return HepmassParameters
    elif dataset == "gas":
        return GasParameters
    elif dataset == "bsds300":
        return BSDS300Parameters
    elif dataset == "miniboone":
        return MiniBooneParameters
    else:
        print(f"Got unexpected dataset string: '{dataset}'")
        raise ValueError
