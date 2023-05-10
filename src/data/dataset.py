sys.path.append(
    '/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/src')

import pandas as pd
import scipy.stats
from config import MVParameters as args
import data_utils
from torch.utils import data
import sys

batch_size = 64


def train(loader, flow):
    pass


def main():
    # Generate data
    distributions, _, distribution = args.get_distributions()

    threshold = data_utils.determine_threshold(
        args.p_event[0], distributions[-1])
    num_normal = int(10e3)
    num_event = int(10e3)

    normal_data, event_data = data_utils.generate_data(
        distribution, num_normal, num_event, threshold)
    # train_normal_data = normal_data

    pd.DataFrame(normal_data).to_csv('data/normal_data.csv', index=False)
    pd.DataFrame(event_data).to_csv('data/event_data.csv', index=False)


if __name__ == "__main__":
    main()
