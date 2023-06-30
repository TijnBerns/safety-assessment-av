import sys
sys.path.append('src')

import numpy as np
import scipy.stats
from pprint import pprint
from datetime import datetime
from typing import List
import click
import pandas as pd

from flow import compute_llh
import matplotlib.pyplot as plt
# from flow_module import create_flow
import flow.parameters as parameters

import utils
from pathlib import Path

@click.command()
@click.option('--true', type=str)
@click.option('--weights', type=List[str])
@click.option('--versions',type=List[str])
@click.option('--dataset',type=str)
def main(versions, weights, true, dataset):
    assert len(version) == len(weights), f"expected versions and weights to be of same length but got: {len(versions)} and {len(weights)}"
    
    df = pd.DataFrame()
    for weight, version in zip(weights, versions):
        row = compute_llh.compute_llh(version, true, dataset)
        row['weight'] = weight
        df = pd.concat([df, pd.DataFrame(row)])
        
    _, axs = plt.subplots(1,1)
    axs.plot(df['weight'], df['mse_all'], label='all')
    axs.plot(df['weight'], df['mse_non_event'], label='non-event')
    axs.plot(df['weight'], df['mse_event'], label='event')
    axs.set_xlabel('weight')
    axs.set_ylabel('MSE')
    
    plt.savefig('img/weight_vs_mse.png')


if __name__ =='__main__': 
    device, _ = utils.set_device()
    
    # Get arguments and dataset
    dataset_str = 'gas'
    dataset = parameters.get_dataset(dataset_str)
    args = parameters.get_parameters(dataset_str)
    
    # Initialize data
    train_full = dataset(split='_train').data
    test_full = dataset(split='_test').data
    results = []
    

    