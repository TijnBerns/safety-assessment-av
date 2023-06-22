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

def write_results(path: Union[str, Path], row: List[float]):
    """Append a row to a csv file specified by the given path.

    Args:
        path (Union[str, Path]): Path to csv file.
        row (List[float]): Row of length three, containing log-likelihood values.
    """
    results_file = Path(path)
    if not results_file.exists():
        results_file.touch()
        with open(results_file, 'w') as f:
            f.write('version, all, all_std, normal, normal_std, event, event_std')
    df = pd.read_csv(results_file, index_col=False)
    df.loc[-1] = row
    df.to_csv(results_file, index=False)


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
    def __init__(self, dataset: str, version: str, test_set: str) -> None:
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
            raise ValueError
        
    def _initialize_dataloaders(self, dataset:data.base.CustomDataset, args: parameters.Parameters) -> DataLoader:
        threshold = dataset.threshold
        xi = dataset.xi
        
        event_data = dataset.data[dataset.data[:,xi] > threshold]
        normal_data = dataset.data[dataset.data[:,xi] <= threshold]
        self.test = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        self.normal = DataLoader(normal_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        self.event = DataLoader(event_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    def _load_checkpoint(self, checkpoint):
        # Load model from checkpoint
        flow_module = FlowModule.load_from_checkpoint(checkpoint, features=self.features, 
                                                      device=self.device, args=self.args, 
                                                      dataset=self.dataset, map_location="cpu").eval()
        return flow_module.to(self.device)

    def evaluate(self, checkpoint):
        flow_module = self._load_checkpoint(checkpoint)
        
        llh_all = float(flow_module.compute_log_prob(self.test))
        print(f'log likelihood all {llh_all}')
        
        llh_normal = float(flow_module.compute_log_prob(self.normal))
        print(f'log likelihood normal {llh_normal}')
        
        llh_event = float(flow_module.compute_log_prob(self.event))
        print(f'log likelihood event {llh_event}')   
        
        return llh_all, llh_normal, llh_event
    
    
    def evaluate_sampled(self, checkpoint, true):
        flow_module = self._load_checkpoint(checkpoint)
        true = self._load_checkpoint(true) 
        
        mse_all = float(flow_module.compute_mse(true, self.test))
        print(f'MSE all {mse_all}')
        
        mse_normal = float(flow_module.compute_mse(true, self.normal))
        print(f'MSE normal {mse_normal}')
        
        mse_event = float(flow_module.compute_mse(true, self.event))
        print(f'MSE event {mse_event}')   
        
        return mse_all, mse_normal, mse_event
        
        
@click.command()
@click.option('--version', type=str)
@click.option('--dataset', default='hepmass')
@click.option('--test_set', default='normal', help='Whether to evaluate on test data normalized using all data or normal data only. Choices: [normal, all]')
@click.option('--true_model', type=str, default=None)
def main(version: str, dataset: str, test_set: str, true_model: str):
    utils.seed_all(2023)
    evaluator = Evaluator(dataset=dataset, version=version, test_set=test_set)
    
    best, _ = get_checkpoint(version)
    if test_set == 'sampled':
        true_model, _ = get_checkpoint(true_model)    
        true_model = get_best_checkpoint(true_model)
    
    all_ls = []
    normal_ls = []
    event_ls = []
    for checkpoint in tqdm(best):
        print(f'Evaluating {checkpoint}')
        if test_set != 'sampled':
            all, normal, event = evaluator.evaluate(checkpoint) 
        else: 
            all, normal, event = evaluator.evaluate_sampled(checkpoint, true_model)
        all_ls.append(all)
        normal_ls.append(normal)
        event_ls.append(event)
        
    row = [
        version, 
        np.mean(all_ls),
        np.std(all_ls),
        np.mean(normal_ls),
        np.std(normal_ls),
        np.mean(event_ls),
        np.std(event_ls)
    ]
    
    if test_set == 'sampled':
        write_results(path='results_sampled.csv', row=row)
    else: 
        write_results(path='results.csv', row=row)

           

if __name__ == "__main__":
    main()