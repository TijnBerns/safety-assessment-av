import sys
sys.path.append('src')

import numpy as np
from pprint import pprint
from utils import save_json, save_np
from base import CustomDataset, split_data
from miniboone import MiniBoone
from power import Power
from gas import Gas
from hepmass import Hepmass
from bsds300 import BSDS300Dataset

import click

def select_variable(data):
    corr = np.abs(np.corrcoef(data.T))
    xi = np.argmax(np.mean(corr, axis=0))
    return xi, corr

def set_threshold(data, xi):
    p_event = 0.08
    threshold = np.percentile(data[:, xi], 100 - (100 * p_event))
    return threshold
   
def normalize(data: np.ndarray, mu=None, std=None):
    if mu is None or std is None:
        mu = data.mean(axis=0)
        std = data.std(axis=0)
    
    normalized_data = (data - mu) / std
    return normalized_data, mu, std

def compute_event_weight(normal, event, xi, threshold):
    p_event = 0.08
    n_norm = np.sum(normal[:,xi] <= threshold)
    n_event = len(event) + len(normal) - n_norm
    return ((p_event * n_norm) / (n_event)) / (1 - p_event)

def load_data(dataset):
    if type(dataset) == BSDS300Dataset:
        train, val, test = dataset.load_data() 
    else:
        train, test = dataset.load_data()
        train, val = split_data(train)
        
    return train, val, test
    
    
def save_splits(dataset: CustomDataset):
    # 1. Split data
    train, val, test = load_data(dataset)
    test_copy = np.array(test)
    val_copy = np.array(val)
    all_train = np.vstack((train, val))
    all_data = np.vstack((all_train, test))
    
    # 2. Select variable on which threshold is set
    # This is set on the attribute with the highest mean correlation wrt all other variables
    xi, corr = select_variable(all_data)

    # 3. Set the threshold for p_event
    threshold = set_threshold(all_data, xi)
    
    # 4. Split train data into normal and event
    # TODO: Check what fraction we want to use here
    train_normal, train_event = split_data(train, frac=0.2)
    train_event = train_event[train_event[:, xi] > threshold]
    test_normal = test[test[:, xi] <= threshold]
    test_event = test[test[:, xi] > threshold]
    save_np(dataset.root / 'normal_train_unnormalized.npy', train_normal)
    save_np(dataset.root / 'event_train_unnormalized.npy', train_event)

    # 5. Normalize data
    train_normal, mu, std = normalize(train_normal)
    train_event = normalize(train_event, mu, std)[0]
    val = normalize(val, mu, std)[0]
    test = normalize(test, mu, std)[0]
    test_normal = normalize(test_normal, mu, std)[0]
    test_event = normalize(test_event, mu, std)[0]
    threshold = (threshold - mu[xi]) / std[xi]
    
    # To reproduce results in normal spline flow paper also store entire train set
    _, mu, std = normalize(all_train)
    train_all = normalize(train, mu, std)[0]
    val_all = normalize(val_copy, mu, std)[0]
    test_all = normalize(test_copy, mu, std)[0]
    
    # 7. Save all datasplits
    # Train splits
    save_np(dataset.root / '_train.npy', train_all)
    save_np(dataset.root / 'normal_train.npy', train_normal)
    save_np(dataset.root / 'event_train.npy', train_event)
    
    # Validation splits
    save_np(dataset.root / '_val.npy', val_all)
    save_np(dataset.root / 'val.npy', val)
    # save_np(dataset.root / 'normal_val.npy', normal_val)
    # save_np(dataset.root / 'event_val.npy', event_val)
    
    # Test splits
    save_np(dataset.root / '_test.npy', test_all)
    save_np(dataset.root / 'test.npy', test)
    save_np(dataset.root / 'test_normal.npy', test_normal)
    save_np(dataset.root / 'test_event.npy', test_event)

    # 7. Save stats of splits
    stats = {
        'root': str(dataset.root),
        '_train.npy': len(train_all),
        'normal_train.npy':len(train_normal),
        'event_train.npy':len(train_event),
        'val.npy':len(val),
        '_test.npy': len(test_all),
        'test.npy': len(test),
        'test_normal.npy': len(test_normal),
        'test_event.npy': len(test_event),
        'attributes': train_all.shape,
        'mu': mu.tolist(),
        'std': std.tolist(),
        'Xi': int(xi), 
        'corr': np.max(np.mean(corr, axis=0)), 
        'threshold': threshold,
        'weight': compute_event_weight(train_normal, train_event, xi, threshold)
    }
    save_json(dataset.root / 'stats.json', stats)
    

def main():
    print('Preprocessing: MiniBoone')
    save_splits(MiniBoone())
    
    print('Preprocessing: Power')
    save_splits(Power())
    
    print('Preprocessing: Gas')
    save_splits(Gas())
    
    print('Preprocessing: Hepmass')
    save_splits(Hepmass())
    
    # print('preprocessing: BSDS300')
    # save_splits(BSDS300Dataset())

    
if __name__ == "__main__":
    main()