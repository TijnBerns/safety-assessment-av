import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory namepip install shapely
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import json
import click
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import evaluate
import numpy as np 
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

import pygeos 

FIGSIZE = (6.6, 4.0)
labels = {
    'num_norm': '$N_\\textrm{norm}$',
    'num_event': '$N_\\textrm{event}$',
    'p_event': '$p_\\textrm{event}$',
    'correlation': '$\\rho$',
    'jaccard_mean': 'Jaccard',
    'jaccard_std': 'Jaccard (std)',
    'delta_jaccard_mean': '$\\Delta$ Jaccard',
    'delta_jaccard_std': '$\\Delta$ Jaccard (std)'
}

def extend_dict(f: Path, suffix: str):
    estimates = utils.load_json(f / (f.name + suffix))
    p_event, n_normal, n_edge, corr = utils.variables_from_filename(f.name)
    estimates['p_event'] = p_event
    estimates['num_norm'] = n_normal
    estimates['num_event'] = n_edge
    estimates['correlation'] =  corr
    
    _, jaccard_mean, jaccard_std = evaluate.evaluate_epsilon_support(estimates)
    estimates['jaccard_mean'] = jaccard_mean
    estimates['jaccard_std'] = jaccard_std
    # estimates['mean'] = mean
    # estimates['std'] = std
    # estimates['mae'] = mae
    # estimates['std_error'] =  std_error
    return estimates

def plot(df: pd.DataFrame, col: str, groupby:str):
    mean = df.groupby(groupby)[col].mean()
    std = df.groupby(groupby)[col].std()
    lower = df.groupby(groupby)[col].quantile(0.025)
    upper = df.groupby(groupby)[col].quantile(0.975)
    x_ticks = df[groupby].unique()
 
    # Create plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(mean, label='mean')
    plt.fill_between(mean.keys(), lower, upper, alpha=0.25, label='95 confidence interval')
    plt.xticks(x_ticks, x_ticks)
    plt.xlabel(labels[groupby])
    plt.ylabel(labels[col])

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'img/epsilon_{col}_{groupby}.png')
    
    # Print correlation
    corrcoef = np.corrcoef(df[col], df[groupby])
    print(f'Correlation ({col} vs. {groupby}): {corrcoef[0][1]}')
    
def plot_diff(baseline_df: pd.DataFrame, improved_df:pd.DataFrame, col: str, groupby:str):
    # Create plot
    plt.figure(figsize=FIGSIZE)
    x_ticks = baseline_df[groupby].unique()
    
    for df, label in (baseline_df, 'baseline'), (improved_df, 'improved'):
        mean = df.groupby(groupby)[col].mean()
        # std = df.groupby(groupby)['jaccard_std'].mean()
        # plt.fill_between(mean.keys(), mean-std, mean+std, alpha=0.25)
        plt.plot(mean, label=label)

    plt.xticks(x_ticks, x_ticks)
    plt.xlabel(labels[groupby])
    plt.ylabel(labels[col])
    plt.legend()

    plt.tight_layout()
    # plt.show()
    print(groupby)
    plt.savefig(f'img/epsilon_{col}_{groupby}_2.png')
    
def plot_gaussian(epsilon=0.1):
    x_values = np.linspace(-6,6, 400)
    pdf = scipy.stats.norm().pdf(x_values)

    x_values_l  = x_values[:200][pdf[:200]<epsilon]
    epsilon_l =  np.ones_like(x_values_l) * epsilon
    x_values_r =  x_values[200:][pdf[200:]<epsilon]
    epsilon_r = np.ones_like(x_values_r) * epsilon
    
    epsilon_supp = x_values[pdf>=epsilon]
    xy = (epsilon_supp[0], epsilon)
    dxdy = (epsilon_supp[-1], epsilon)
    
    _, axs = plt.subplots(1,1, figsize=FIGSIZE)
    axs.plot(x_values_l, epsilon_l, color='black', linestyle='dotted', label='$y=\\epsilon$')
    axs.plot(x_values_r, epsilon_r, color='black', linestyle='dotted')
    axs.plot(x_values, pdf, label='$f(x)$')
    axs.annotate("", xy=xy, xytext=dxdy, arrowprops=dict(arrowstyle='<->'), label='test')
    # axs.annotate("$\\epsilon$-support", xy=xy, xytext=dxdy)
    axs.set_xlabel('$x$')
    axs.set_ylabel('density')
    # axs.legend()
    plt.tight_layout()
    plt.savefig(f'img/epsilon_support_example.pgf')

@click.command()
@click.option('--path', '-p',type=Path)
def main(path: Path):
    baseline_ls = []
    improved_ls = []
    
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir() or f.name.find('results') != -1:
            continue
        baseline = extend_dict(f, '.baseline_eps.json')
        baseline_ls.append(baseline)
        
        improved = extend_dict(f, '.improved_eps.json')
        improved_ls.append(improved)
    
    baseline_df = pd.DataFrame(baseline_ls)
    improved_df = pd.DataFrame(improved_ls)
    improved_df['delta_jaccard_mean'] = baseline_df['jaccard_mean'] - improved_df['jaccard_mean']
    improved_df['delta_jaccard_std'] = baseline_df['jaccard_std'] - improved_df['jaccard_std']
    
    # Create plots for each of the variables
    for col in ('delta_jaccard_mean', 'delta_jaccard_std'):
        for groupby in ('correlation', 'p_event', 'num_norm', 'num_event'):
            plot(improved_df, col, groupby)
    
    for col in ('jaccard_mean', 'jaccard_std'):
        for groupby in ('correlation', 'p_event', 'num_norm', 'num_event'):
            plot_diff(baseline_df, improved_df,col, groupby)
    
    # # Print maximum and minimum parameter configurations
    # cols = ['delta_mae', 'delta_std', 'p_event', 'num_norm', 'num_event', 'correlation']
    # print(improved_df[improved_df['delta_mae'] == improved_df['delta_mae'].min()][cols])
    # print(improved_df[improved_df['delta_mae'] == improved_df['delta_mae'].max()][cols])
    # print(improved_df[improved_df['delta_std'] == improved_df['delta_std'].min()][cols])
    # print(improved_df[improved_df['delta_std'] == improved_df['delta_std'].max()][cols])
    
    

if __name__ == "__main__":
    main()
    # plot_gaussian()