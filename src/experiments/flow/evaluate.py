import sys
sys.path.append('src')

import data.data_utils as data_utils
import utils
from pathlib import Path

from typing import Tuple
import click
from torch.utils.data import DataLoader
from flow_module import FlowModule
import matplotlib.pyplot as plt
import parameters
from utils import save_json
import pandas as pd
import numpy as np
from scipy.stats import uniform
from typing import List
from tqdm import tqdm


def sample_test(dataset: str, num_samples:int = int(1e6)):
    custom_dataset = parameters.get_dataset(dataset)
    train = custom_dataset(split='_train')   

    variable_min = np.min(train.data, axis=0)
    variable_max = np.max(train.data, axis=0)
    test = np.zeros((num_samples, len(variable_min)))
    
    for i, (vmin, vmax) in enumerate(zip(variable_min, variable_max)):
        da = uniform(loc=vmin, scale=vmax).rvs(size=num_samples)
        test[:,i] = da
       
    return test 
        

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


# def sort_fn(path: Path) -> float:
#     """Utility function used to sort checkpoints files

#     Args:
#         path (Path): Checkpoint path

#     Returns:
#         float: Log-likelihood of saved checkpoint
#     """
#     s = path.name.split('.')
#     r = s[-3]
#     m = s[-4][-3:]
#     if m[0] == '_':
#         m = m[1:]
#     return float(f'{m}.{r}')


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


def eval(checkpoint, version, dataset):
    # Set device
    device, _ = utils.set_device()
    
    # Retrieve arguments correspondnig to dataset
    args = parameters.get_parameters(dataset)
    
    # Inititialize datasets
    dataset = parameters.get_dataset(dataset)
    normal  = dataset(split='test_normal')
    event = dataset(split='test_event')
    test = dataset(split='test')
    _test = dataset(split='_test')
    features = normal.data.shape[1]
    
    # Initialize dataloaders
    normal = DataLoader(normal, batch_size=args.batch_size, num_workers=args.num_workers)
    event = DataLoader(event, batch_size=args.batch_size, num_workers=args.num_workers)
    test = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)
    _test = DataLoader(_test, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load model from checkpoint
    flow_module = FlowModule.load_from_checkpoint(checkpoint, features=features, device=device, args=args, dataset=dataset()).eval()
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
def main(version: str, dataset: str):
    best, _ = get_checkpoint(version)
    
    llh_all_ls = []
    llh_normal_ls = []
    llh_event_ls = []
    for checkpoint in tqdm(best):
        # evaluate best checkpoint
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