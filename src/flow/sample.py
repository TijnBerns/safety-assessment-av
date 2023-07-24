import sys
sys.path.append('src')

import click
import parameters
from evaluate import get_pl_checkpoint, get_best_checkpoint, sample_normal, sample_event
from flow_module import FlowModule
from utils import set_device
from scipy.stats import uniform
import torch
import utils
import numpy as np
from typing import Tuple, List
from pathlib import Path
from data.base import CustomDataset


@click.command()
@click.option('--dataset', type=str, default='gas')
@click.option('--version', type=str, default='253784')
@click.option('--num_samples', type=int, default=None)
def sample(dataset:str, version: str, num_samples: int) -> Tuple[np.ndarray, np.ndarray]: 
    """_summary_

    Args:
        dataset (str): _description_
        version (str): _description_
        num_samples (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

    # Set device dataset and parameters for given dataset
    
    num_samples = None
    utils.seed_all(2023)
    device, _ = set_device()
    args = parameters.get_parameters(dataset)
    dataset: CustomDataset = parameters.get_dataset(dataset)(split='_train')
    
    # Get dataset stats
    threshold = dataset._threshold
    xi = dataset.xi
    features = dataset.data.shape[1]
    
    # Load checkpoint
    best, _ = get_pl_checkpoint(version)
    best = get_best_checkpoint(best)
    flow_module = FlowModule.load_from_checkpoint(best, features=features, device=device, args=args, dataset=dataset, map_location=torch.device('cpu')).eval()
    
    # flow_module = flow_module.to(device)
    if num_samples is None:
        num_normal_samples =  dataset.stats['normal_train.npy']
        num_event_samples = dataset.stats['event_train.npy']
    else:
        num_normal_samples = dataset.stats['normal_train.npy']
        num_event_samples = dataset.stats['normal_train.npy']
    num_test_samples = dataset.stats['test.npy']

    # Generate normal train data
    print(f"Generating {num_normal_samples} normal train and validation data") 
    normal = sample_normal(flow_module, num_normal_samples, save=dataset.root / 'normal_sampled.npy')
    val = sample_normal(flow_module, dataset.stats['val.npy'], save=dataset.root / 'val_sampled.npy')

    # Generate event train data
    print(f"Generating {num_event_samples} event train data")
    event = sample_event(flow_module, num_event_samples, threshold, xi, save=dataset.root / 'event_sampled.npy')
    
    # Generate test data
    print(f"Generating {num_test_samples} test samples")
    test = sample_normal(flow_module, num_samples=num_test_samples,  save=dataset.root / 'test_sampled.npy' )
    return normal, event, val, test

        

if __name__ == "__main__":
    sample()
