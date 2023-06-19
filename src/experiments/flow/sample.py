import sys
sys.path.append('src')

import click
import parameters
from evaluate import get_checkpoint
from flow_module import FlowModule
from utils import set_device
from scipy.stats import uniform
import torch
from utils import save_np
import numpy as np
from typing import Tuple, List
from pathlib import Path
from data.base import CustomDataset

def sample_normal(flow_module: FlowModule, num_samples: int, save: str= None):
    sampled_data = flow_module.sample(num_samples).numpy()
    
    if save is not None:
        save_np(save, sampled_data)
    
    return sampled_data

def sample_event(flow_module: FlowModule, num_samples: int, threshold:float, xi: int, save: str=None):
    sampled_data = torch.empty((0, flow_module.features))
    while sampled_data.shape[0] < num_samples:
        temp = flow_module.sample(num_samples)
        sampled_data = torch.cat((sampled_data, temp[temp[:,xi] > threshold]), 0)
        print(f'{sampled_data.shape[0]} / {num_samples}')
    sampled_data = sampled_data[:num_samples].numpy()
    
    if save is not None:
        save_np(save, sampled_data)
    
    return sampled_data


def get_best_checkpoint(checkpoints: List[Path]):
    def sort_fn(path: Path) -> float:
        name = path.name
        log_density = float(name.split('log_density_')[-1].split('.best.ckpt')[0])
        return log_density
    
    checkpoints.sort(key=sort_fn)
    return checkpoints[-1]
    
    
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
    device, _ = set_device()
    args = parameters.get_parameters(dataset)
    dataset: CustomDataset = parameters.get_dataset(dataset)(split='_train')
    
    # Get dataset stats
    threshold = dataset.threshold
    xi = dataset.xi
    features = dataset.data.shape[1]
    
    # Load checkpoint
    best, _ = get_checkpoint(version)
    best = get_best_checkpoint(best)
    flow_module = FlowModule.load_from_checkpoint(best, features=features, device=device, args=args, dataset=dataset, map_location=torch.device('cpu')).eval()
    # flow_module = flow_module.to(device)
   
    # Generate normal data
    print(f"Generating {num_samples} normal train and validation data") 
    normal = sample_normal(flow_module, dataset.stats['normal_train.npy'], save=dataset.root / 'normal_sampled.npy')
    val = sample_normal(flow_module, dataset.stats['val.npy'], save=dataset.root / 'val_sampled.npy')

    # Generate event data
    print(f"Generating {num_samples} event train data")
    event = sample_event(flow_module, dataset.stats['event_train.npy'], threshold, xi, save=dataset.root / 'event_sampled.npy')
    
    return normal, event, val
        

if __name__ == "__main__":
    sample()
