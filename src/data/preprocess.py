import sys
sys.path.append('src')

import numpy as np
from pprint import pprint
from utils import save_json, save_np
from config import FlowParameters
from base import CustomDataset, split_data
from miniboone import MiniBoone
from power import Power
from gas import Gas
from hepmass import Hepmass

def save_splits(dataset: CustomDataset):
    # 1. Split data
    train, test = dataset.load_data()
    data = np.vstack((train, test))

    # 2. Select variable on which threshold is set
    # TODO: Decide on selecting random variable, alsways the last attribute, or the one that satisfies a certain correlation
    # For now I selected the last attribute, as is done in previous experiments
    xi = train.shape[1] - 1

    # 3. Set the threshold for p_event
    p_event = FlowParameters.p_event
    threshold = np.percentile(data[:, xi], 100 - (100 * p_event))

    # 4. Split train data into normal and event
    # TODO: Check what fraction we want to use here
    normal_train, event_train = split_data(train, frac=0.2)
    event_train = event_train[event_train[:, xi] > threshold]
    test_normal = test[test[:, xi] <= threshold]
    test_event = test[test[:, xi] > threshold]

    # 5. Normalize data
    mu = normal_train.mean(axis=0)
    sigma = normal_train.mean(axis=0)
    normal_train = (normal_train - mu) / sigma
    event_train = (event_train - mu) / sigma
    test = (test - mu) / sigma
    test_normal = (test_normal - mu) / sigma
    test_event = (test_event - mu) / sigma
    
    # 6. Split validation set into event and normal
    normal_train, normal_val = split_data(normal_train)
    event_train, event_val = split_data(event_train)
    
    # 6. Save all datasplits
    save_np(dataset.root / '_train.npy', train)
    save_np(dataset.root / 'normal_train.npy', normal_train)
    save_np(dataset.root / 'event_train.npy', event_train)
    save_np(dataset.root / 'normal_val.npy', normal_val)
    save_np(dataset.root / 'event_val.npy', event_val)
    save_np(dataset.root / 'test.npy', test)
    save_np(dataset.root / 'test_normal.npy', test_normal)
    save_np(dataset.root / 'test_event.npy', test_event)

    # # 7. Save stats of splits
    # stats = {
    #     'root': str(dataset.root),
    #     'mu': mu.tolist(),
    #     'std': sigma.tolist(),
    #     '_train': len(train),
    #     'normal':len(normal_train),
    #     'event':len(event_train),
    #     'val':len(val),
    #     'test': len(test),
    #     'attributes': train.shape[1],
    # }
    # pprint(stats)
    # save_json(dataset.root / 'stats.json', stats)
    
    
def main():
    save_splits(MiniBoone())
    save_splits(Power())
    save_splits(Gas())
    save_splits(Hepmass())
    
if __name__ == "__main__":
    main()