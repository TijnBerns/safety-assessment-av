import sys
sys.path.append('src')
sys.path.append('src/flow')


import click
import pandas as pd
from flow import compute_llh
from utils import FIGSIZE_1_1 as FIGSIZE

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"


versions = ['566848', '566849', '566850', '566852', '566853', '566854', '566855', '566856', '566858']
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
true = '253784'
dataset = 'gas'


def format(x, pos):
    return ("%.1f"%(x)).lstrip('0')

# @click.command()
# @click.option('--true', type=str)
# @click.option('--weights', multiple=True)
# @click.option('--versions',multiple=True)
# @click.option('--dataset',type=str)
def main(versions, weights, true, dataset) -> None:
    assert len(versions) == len(weights), f"expected versions and weights to be of same length but got: {len(versions)} and {len(weights)}"
    print(list(zip(versions, weights)))
    
    df = None
    for weight, version in zip(weights, versions):
        # Compute llh and mse
        row = compute_llh.main(version=version, true=true, dataset=dataset)
        row[0]['weight'] = weight
 
        # Add row to dataframe
        if df is None:
            df = pd.DataFrame(row)
        else:
            df = pd.concat([df, pd.DataFrame(row)])
    
    for score in ['mse', 'llh']:
        # 
        _, axs = plt.subplots(1,3, figsize=(6.2,1.8))
        axs[0].set_ylabel('MSE')
        for ax, label in zip(axs, ['all', 'non-event', 'event']):    
            col = label.replace('-', '_')
            # weight_list = list(df['weight'])
            # weight_list.append(1.0)
            # score_list = list(df[f'{score}_{col}'])
            # score_list.append('3.77291005611419')
            ax.plot(df['weight'], df[f'{score}_{col}'], label=label)
            ax.set_title(label.capitalize())
            ax.set_xlabel('$w$')
            
            # ax.xaxis.set_major_formatter(FuncFormatter(format))
            
        plt.tight_layout()
        plt.savefig(f'img/weight_vs_{score}_2.pgf')


if __name__ =='__main__': 
    main(versions, weights, true, dataset)
    