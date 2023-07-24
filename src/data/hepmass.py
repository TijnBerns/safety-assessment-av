import sys
sys.path.append('src')

import os
from pathlib import Path
import numpy as np
from data.base import CustomDataset
import pandas as pd
from collections import Counter

class Hepmass(CustomDataset):
    def __init__(self, split=None) -> None:
        super().__init__(Path(os.environ['DATAROOT']) / 'hepmass', split)
        self.dataset_str = "hepmass"
    
    def load_data(self):
        # Following https://github.com/bayesiains/nsf/blob/master/data/hepmass.py
        train = pd.read_csv(self.root / '1000_train.csv', index_col=False)
        test =  pd.read_csv(self.root / '1000_test.csv', index_col=False)
     
        # Gets rid of any background noise examples i.e. class label 0.
        train = train[train[train.columns[0]] == 1]
        train = train.drop(train.columns[0], axis=1)
        test = test[test[test.columns[0]] == 1]
        test = test.drop(test.columns[0], axis=1)
        
        # Because the data set is messed up!
        test = test.drop(test.columns[-1], axis=1)
        
        train, test = train.values, test.values
        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in train.T:
            c = Counter(feature)
           
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        train = train[:, np.array(
            [i for i in range(train.shape[1]) if i not in features_to_remove])]
        test = test[:, np.array(
            [i for i in range(test.shape[1]) if i not in features_to_remove])]
        
        return train, test

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

    
