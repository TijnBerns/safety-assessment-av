import sys
sys.path.append('src')

import data.data_utils as data_utils
import utils
from pathlib import Path

from typing import Tuple
import click
from torch.utils.data import DataLoader
from flow_module import FlowModule, FlowModuleWeighted
import matplotlib.pyplot as plt
import parameters
from utils import save_json
import pandas as pd
import numpy as np
from scipy.stats import uniform
from typing import List, Union
from tqdm import tqdm
import data.base
import torch

def sample_normal(flow_module: FlowModule, num_samples: int, save: str= None):
    """Sample normal data from a flow module

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
    """Sample event data from a flow module.

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


def get_checkpoint(version: str) -> Tuple[List[Path], None]:
    """Loads the checkpoints of a given version

    Args:
        version (str): The version that is used to load the checkpoints.

    Returns:
        Tuple[List[Path], None]: List of Paths to all checkpoints containing the keyword 'best'
    """
    path = Path(f'/home/tberns/safety-assessment-av/lightning_logs/version_{version}/checkpoints')
    best_checkpoint = list(path.rglob('*best.ckpt'))
    return best_checkpoint, None


def get_best_checkpoint(checkpoints: List[Path]):
    """Gets the best checkpoint based on the loglikelihood out of a list of checkpoints.

    Args:
        checkpoints (List[Path]): List of checkpoints.
    
    Returns:
        Path: Path to the best checkpoint.
    """
    def sort_fn(path: Path) -> float:
        name = path.name
        log_density = float(name.split('log_density_')[-1].split('.best.ckpt')[0])
        return log_density
    
    checkpoints.sort(key=sort_fn)
    return checkpoints[-1]


class Evaluator():
    def __init__(self, dataset: str, version: str, test_set: str='all') -> None:
        # Set device
        device, _ = utils.set_device()
        self.device = device
    
        # Retrieve arguments corresponding to dataset
        self.args = parameters.get_parameters(dataset)
  
        # Initialize dataset
        self._initialize_dataset(dataset, test_set)
        self.features = self.dataset.data.shape[1]
        
        # Initialize dataloaders
        self._initialize_dataloaders(self.dataset, self.args)
    
    
    def _initialize_dataset(self, dataset: data.base.CustomDataset, test_set: str) -> data.base.CustomDataset:
        dataset = parameters.get_dataset(dataset)
        if test_set == 'normal':
            self.dataset = dataset(split='test')
        elif test_set == 'all':
            self.dataset = dataset(split='_test')
        elif test_set == 'sampled':
            self.dataset = dataset(split='test_sampled')
        else:
            print(f'Got unexpected argument for test_set: {dataset}')
            raise ValueError
        
    def _initialize_dataloaders(self, dataset:data.base.CustomDataset, args: parameters.Parameters) -> DataLoader:
        self.threshold = dataset._threshold
        self.xi = dataset.xi
        
        event_data = dataset.data[dataset.data[:,self.xi] > self.threshold]
        normal_data = dataset.data[dataset.data[:,self.xi] <= self.threshold]
        self.test = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        self.normal = DataLoader(normal_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        self.event = DataLoader(event_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    def _load_checkpoint(self, checkpoint: Path) -> FlowModule:
        # Load model from checkpoint
        flow_module = FlowModule.load_from_checkpoint(checkpoint, features=self.features, 
                                                      device=self.device, args=self.args, 
                                                      dataset=self.dataset, map_location="cpu").eval()
        return flow_module.to(self.device)
    
    def _likelihood_ratio(self, true, other):
        return 2 * float(torch.mean(other - true))
    
    def _mse(self, true, other):
        return float(torch.mean(torch.square(true - other)))

    def _split_llh(self, llh):
        llh_all = llh
        llh_non_event = llh[self.test.dataset[:,self.xi] <= self.threshold]
        llh_event = llh[self.test.dataset[:,self.xi] > self.threshold]
        return llh_all, llh_non_event, llh_event
    
    def compute_llh_tensor(self, checkpoint: Path) -> torch.Tensor:
        # Load flow module from checkpoint
        flow_module = self._load_checkpoint(checkpoint)
        llh = flow_module.compute_llh(self.test)
        
        # Save log-likelihood tensor
        torch.save(llh, Path(checkpoint).parent / (checkpoint.name[:-4] + "llh.pt"))

        # Split in all, non-event, and event
        llh_all, llh_non_event, llh_event = self._split_llh(llh)
        
        # Return mean log-likelihood on non-event and event data
        return torch.mean(llh_all), torch.mean(llh_non_event), torch.mean(llh_event)
    
    def compute_llh(self, checkpoint: Path):
        # Load likelihood tensors
        llh_estimate = torch.load(checkpoint.parent / (checkpoint.name[:-4] + "llh.pt"))
        
        # Split in all, non-event, and event
        all, non_event, event = self._split_llh(llh_estimate)
        return float(torch.mean(all).item()), float(torch.mean(non_event).item()), float(torch.mean(event).item())
        
    def compute_lr(self, true: Path, checkpoint: Path) -> Tuple[float, float]: 
        # Load likelihood tensors
        llh_estimate = torch.load(checkpoint.parent / (checkpoint.name[:-4] + "llh.pt"))
        llh_true = torch.load(true.parent / (true.name[:-4] + "llh.pt"))
        
        # Split in all, non-event, and event
        all_estimate, non_event_estimate, event_estimate = self._split_llh(llh_estimate)
        all_true, non_event_true, event_true = self._split_llh(llh_true)
        
        # Compute llh-ratios
        ratio_all = self._likelihood_ratio(all_true, all_estimate)
        ratio_event = self._likelihood_ratio(event_true, event_estimate)
        ratio_non_event = self._likelihood_ratio(non_event_true, non_event_estimate)
        return ratio_all, ratio_non_event, ratio_event
    
    
    def compute_mse(self, true: Path, checkpoint: Path) -> Tuple[float, float]: 
        # Load likelihood tensors
        llh_estimate = torch.load(checkpoint.parent / (checkpoint.name[:-4] + "llh.pt"))
        llh_true = torch.load(true.parent / (true.name[:-4] + "llh.pt"))
        
        # Split in all, non-event, and event
        all_estimate, non_event_estimate, event_estimate = self._split_llh(llh_estimate)
        all_true, non_event_true, event_true = self._split_llh(llh_true)
        
        # Compute llh-ratios
        mse_all = self._mse(all_true, all_estimate)
        mse_non_event = self._mse(non_event_true, non_event_estimate)
        mse_event = self._mse(event_true, event_estimate)
        return mse_all, mse_non_event, mse_event
    
    
def evaluate(version: str, dataset:str, test_set: str):
    utils.seed_all(2023)
    evaluator = Evaluator(dataset=dataset, version=version, test_set=test_set)
    best, _ = get_checkpoint(version)

    for checkpoint in tqdm(best):
        print(f'Evaluating {checkpoint}')
        evaluator.compute_llh_tensor(checkpoint)


@click.command()
@click.option('--version', type=str)
@click.option('--dataset', default='hepmass')
@click.option('--test_set', default='all', help='Whether to evaluate on test data normalized using all data or normal data only. Choices: [normal, all]')
def main(version: str, dataset: str, test_set: str):
    evaluate(version=version,dataset=dataset, test_set=test_set)

           

if __name__ == "__main__":
    main()