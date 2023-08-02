import sys
sys.path.append('src')

import click
import parameters
from evaluate import get_pl_checkpoint, get_best_checkpoint
from flow_module import FlowModule
from utils import set_device
from scipy.stats import uniform
import torch
import utils
import numpy as np
from typing import Tuple, List
from pathlib import Path
from data.base import CustomDataset

def sample_normal(flow_module: FlowModule, num_samples: int, save: str= None):
    """Sample normal data from a flow module, and saves result to provided save path as numpy array.

    Args:
        flow_module (FlowModule): The module from which sampels are drawn.
        num_samples (int): Number of samples to draw.
        save (str, optional): The path to which sampled data is save. If None, samples are not saved. Defaults to None.

    Returns:
        np.ndarray: The sampled data.
    """
    sampled_data = flow_module.sample(num_samples).numpy()
    
    if save is not None:
        utils.save_np(save, sampled_data)
    
    return sampled_data

def sample_event(flow_module: FlowModule, num_samples: int, threshold:float, xi: int, save: str=None) -> np.ndarray:
    """Sample event data from a flow module, and saves result to provided save path as numpy array.

    Args:
        flow_module (FlowModule): The module from which samples are drawn.
        num_samples (int): Number of samples to draw.
        threshold (float): The threshold determining whether a sample is an event or not.
        xi (int): The variable on which the threshold is set.
        save (str, optional): The path to which sampled data is save. If None, samples are not saved. Defaults to None.

    Returns:
        np.ndarray: The sampled data.
    """
    sampled_data = torch.empty((0, flow_module.features))
    while sampled_data.shape[0] < num_samples:
        temp = flow_module.sample(num_samples)
        sampled_data = torch.cat((sampled_data, temp[temp[:,xi] > threshold]), 0)
        print(f'{sampled_data.shape[0]} / {num_samples}')
    sampled_data = sampled_data[:num_samples].numpy()
    
    if save is not None:
        utils.save_np(save, sampled_data)
    
    return sampled_data


@click.command()
@click.option('--dataset', type=str)
@click.option('--version', type=str)
@click.option('--num_normal', type=int, default=None)
@click.option('--num_event', type=int, default=None)
def sample(dataset:str, version: str, num_normal: int, num_event: int) -> Tuple[np.ndarray, np.ndarray]: 
    """Sample data from a flow module, and save results to DATAROOT/DATASET/*_sampled.npy.

    Args:
        dataset (str): The dataset that is used to train the flow module. Choices: gas, miniboone, hepmass, power.
        version (str): The pytorch lightning version of the trained flow module. Checkpoint is loaded from lightning_logs/version_$VERSION.
        num_normal (int): The number of normal samples to draw. If None, the same amount of samples is drawn as in the normal train split.
        num_event (int): The number of event samples to draw. If None, the same amount of samples is drawn as in the event train split.

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    # Set device dataset and parameters for given dataset
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
    
    # Set the number of samples to draw
    num_normal = dataset.stats['normal_train.npy'] if num_normal is None else num_normal
    num_event = dataset.stats['event_train.npy'] if num_event is None else num_event
    num_val = dataset.stats['val.npy']
    num_test = dataset.stats['test.npy']

    # Generate normal train data
    print(f"Generating {num_normal} normal train and validation data") 
    normal = sample_normal(flow_module, num_normal, save=dataset.root / 'normal_sampled.npy')

    # Generate event train data
    print(f"Generating {num_event} event train data")
    event = sample_event(flow_module, num_event, threshold, xi, save=dataset.root / 'event_sampled.npy')
    
    # Generate validation data
    print(f"Generating {num_val} event val samples")
    val = sample_normal(flow_module, num_val, save=dataset.root / 'val_sampled.npy')
    
    # Generate test data
    print(f"Generating {num_test} test samples")
    test = sample_normal(flow_module, num_samples=num_test,  save=dataset.root / 'test_sampled.npy' )
    return normal, event, val, test

if __name__ == "__main__":
    sample()
