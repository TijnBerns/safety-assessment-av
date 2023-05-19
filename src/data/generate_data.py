import sys
sys.path.append('src')

import data_utils
from config import MVParameters
import pandas as pd

import click
@click.command()
@click.option("--num_normal", default=10_000)
@click.option("--num_event",default=10_000)
@click.option("--p_event", default=0.08)
@click.option("--random_state", default=2023)
def main(num_normal, num_event, p_event, random_state):
    distributions_, _, distribution = MVParameters.get_distributions()
    threshold = data_utils.determine_threshold(p_event, distributions_[-1])
    normal_data, event_data = data_utils.generate_data(distribution, num_normal, num_event, threshold, random_state=random_state)
    
    pd.DataFrame(normal_data).to_csv('data/normal.csv', index=False)
    pd.DataFrame(event_data).to_csv('data/event.csv', index=False)
    
if __name__ == "__main__":
    main()