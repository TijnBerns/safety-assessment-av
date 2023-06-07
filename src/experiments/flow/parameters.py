import sys
sys.path.append('src')

from dataclasses import dataclass
from data.power import Power
from data.miniboone import MiniBoone
from data.gas import Gas
from data.hepmass import Hepmass
from data.bsds300 import BSDS300Dataset


def get_dataset(dataset: str):
    if dataset == 'power':
        return Power
    elif dataset == 'hepmass':
        return Hepmass
    elif dataset == 'gas':
        return Gas
    elif dataset =='basds300':
        return BSDS300Dataset
    elif dataset == 'miniboone':
        return MiniBoone
    else:
        raise ValueError

def get_parameters(dataset:str):
    if dataset == 'power':
        return PowerParamametrs
    elif dataset == 'hepmass':
        return HepmassParamters
    elif dataset == 'gas':
        return Parameters
    elif dataset =='basds300':
        return Parameters
    elif dataset == 'miniboone':
        return Parameters
    else:
        raise ValueError
    

@dataclass
class Parameters():
    # Num workers
    4
    # The batch size
    batch_size = 512
    # The number of training steps in both training stages
    training_steps_stage_1 = 0
    training_steps_stage_2 = 400_000
    # The learning rate used during both training stages
    learning_rate_stage_1 = 0.0005
    learning_rate_stage_2 = 0.0005
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
    # Number of flow steps to freeze
    # TODO:


@dataclass
class PowerParamametrs(Parameters):
    pass


@dataclass
class HepmassParamters(Parameters):
    batch_size = 256
    num_flow_steps = 20
    num_transform_blocks = 1
    hidden_features = 128
    dropout_probability = 0.2
    
@dataclass
class GasParamters(Parameters):
    batch_size=512
    num_flow_steps=10
    hidden_features=256
    num_bins=8
    dropout_probability=0.1