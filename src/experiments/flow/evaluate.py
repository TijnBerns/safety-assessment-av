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
from typing import List
from tqdm import tqdm
import data.base
      

def write_results(row: List[float]):
    """Append a row to 'results.csv'

    Args:
        row (List[float]): Row of length three, containing log-likelihood values.
    """
    results_file = Path('results.csv')
    if not results_file.exists():
        results_file.touch()
        with open(results_file, 'w') as f:
            f.write('version, all, all_std, normal, normal_std, event, event_std')
    df = pd.read_csv(results_file, index_col=False)
    df.loc[-1] = row
    df.to_csv(results_file, index=False)


def get_checkpoint(version: str) -> Tuple[Path, Path]:
    """Loads the best checkpoint and last checkpoint

    Args:
        version (str): The version that is used to load the checkpoints.

    Returns:
        Tuple[Path, Path]: Path to best and last checkpoint respectively.
    """
    path = Path(f'/home/tberns/safety-assessment-av/lightning_logs/version_{version}/checkpoints')
    best_checkpoint = list(path.rglob('*best.ckpt'))
    # best_checkpoint.sort(key=sort_fn)

    # last_checkpoint = list(path.rglob('*last.ckpt'))[0]
    return best_checkpoint, None


def initialize_dataset(dataset: data.base.CustomDataset, normalize: str) -> data.base.CustomDataset:
    if normalize == 'normal':
        return dataset(split='test')

    elif normalize == 'all':
        return dataset(split='test')
    else:
        raise ValueError
    

def get_dataloaders(dataset:data.base.CustomDataset, args: parameters.Parameters) -> DataLoader:
    threshold = dataset.threshold
    xi = dataset.xi
    
    event_data = dataset.data[dataset.data[:,xi] > threshold]
    normal_data = dataset.data[dataset.data[:,xi] <= threshold]
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    normal_loader = DataLoader(normal_data, batch_size=args.batch_size, num_workers=args.num_workers)
    event_loader = DataLoader(event_data, batch_size=args.batch_size, num_workers=args.num_workers)
    
    return test_loader, normal_loader, event_loader        


def eval(checkpoint, version, dataset, normalize='all'):
    # Set device
    device, _ = utils.set_device()
    
    # Retrieve arguments corresponding to dataset
    args = parameters.get_parameters(dataset)
       
    # Inititialize datasets
    dataset = parameters.get_dataset(dataset)
    dataset = initialize_dataset(dataset, normalize)
    features = dataset.data.shape[1]
    
    # Initialize dataloaders
    test, normal, event = get_dataloaders(dataset, args)
       
    # Load model from checkpoint
    flow_module = FlowModule.load_from_checkpoint(checkpoint, features=features, device=device, args=args, dataset=dataset, map_location="cpu").eval()
    flow_module = flow_module.to(device)
    
    # Print log likelihood for normal and event data
    # print(float(flow_module.compute_log_prob(_test)))
    
    llh_all = float(flow_module.compute_log_prob(test))
    print(f'log likelihood all {llh_all}')
    
    llh_normal = float(flow_module.compute_log_prob(normal))
    print(f'log likelihood normal {llh_normal}')
    
    llh_event = float(flow_module.compute_log_prob(event))
    print(f'log likelihood event {llh_event}')   
    
    return version, llh_all, llh_normal, llh_event
    


@click.command()
@click.option('--version', type=str)
@click.option('--dataset', default='hepmass')
@click.option('--normalize', default='normal', help='')
def main(version: str, dataset: str, normalize: str):
    best, _ = get_checkpoint(version)
    
    llh_all_ls = []
    llh_normal_ls = []
    llh_event_ls = []
    for checkpoint in tqdm(best):
        print(f'Evaluating {version} best')
        _,  llh_all, llh_normal, llh_event = eval(checkpoint, version + ' best', dataset)
        llh_all_ls.append(llh_all)
        llh_normal_ls.append(llh_normal)
        llh_event_ls.append(llh_event)
        
    row = [
        version, 
        np.mean(llh_all_ls),
        np.std(llh_all_ls),
        np.mean(llh_normal_ls),
        np.std(llh_normal_ls),
        np.mean(llh_event_ls),
        np.std(llh_event_ls)
    ]
    
    write_results(row)
           

if __name__ == "__main__":
    main()