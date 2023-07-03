import sys
sys.path.append('src')
sys.path.append('src/flow')


import click
import pandas as pd
from flow import compute_llh
from utils import FIGSIZE_1_2 as FIGSIZE

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

@click.command()
@click.option('--true', type=str)
@click.option('--weights', multiple=True)
@click.option('--versions',multiple=True)
@click.option('--dataset',type=str)
def main(versions, weights, true, dataset) -> None:
    assert len(versions) == len(weights), f"expected versions and weights to be of same length but got: {len(versions)} and {len(weights)}"
    print(list(zip(versions, weights)))
    
    df = None
    for weight, version in zip(weights, versions):
        # Compute llh and mse
        row = compute_llh.compute_llh(version=version, true=true, dataset=dataset)
        row[0]['weight'] = weight
 
        # Add row to dataframe
        if df is None:
            df = pd.DataFrame(row)
        else:
            df = pd.concat([df, pd.DataFrame(row)])
    
    for score in ['mse', 'llh']:
        for label in ['all', 'non-event', 'event']:    
            col = label.replace('-', '_')
            _, axs = plt.subplots(1,1, figsize=FIGSIZE)
            axs.plot(df['weight'], df[f'{score}_{col}'], label=label)
            axs.set_xlabel('$w$')
            axs.set_ylabel('MSE')
            plt.tight_layout()
            plt.savefig(f'img/weight_vs_{score}_{col}.pgf')


if __name__ =='__main__': 
    main()
    