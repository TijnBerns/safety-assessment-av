from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re 
import json
from config import Config as cfg

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0,1, len(cfg.p_edge)))
p_edge_colors = {str(k): v for k,v in zip(cfg.p_edge, colors)}

def rec_dd():
    return defaultdict(rec_dd)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def variables_from_filename(f: str):
    # Extract variables from file name
    f_split = f.split('.')
    p_edge = f_split[0][7:] + '.' + f_split[1]
    n_normal = f_split[2][9:]
    n_edge = f_split[3][7:]
    return p_edge, n_normal, n_edge
    

def evaluate(dataframe: pd.DataFrame):
    # Collect all estimates in single ndarray
    run_cols = [col for col in dataframe if col.startswith('run')]
    estimates = dataframe[run_cols].to_numpy()
    
    # The true pdf values
    true = dataframe['true'].to_numpy()
    true_matrix = np.tile(true, (estimates.shape[-1], 1))
    
    # Compute various metrics
    sse = np.square(estimates.T - true_matrix).sum(axis=1)
    mean = np.mean(estimates.T, axis=0)
    var = np.var(estimates.T, axis=0)
    
    plot_estimate(true, estimates, dataframe['x'])

    return sse, var


def plot_estimate(true, estimates, x_values):
    _, axs = plt.subplots(1, 1)

    # Plot predictions
    axs.plot(x_values, np.mean(estimates.T, axis=0))
    upper = np.percentile(estimates.T, 97.5, axis=0)
    lower = np.percentile(estimates.T, 2.5, axis=0)
    axs.plot(x_values, upper, color='tab:blue')
    axs.plot(x_values, lower, color='tab:blue')
    axs.fill_between(x_values, upper, lower, color='tab:blue', alpha=0.5)
    axs.plot(x_values, true, color='tab:orange')
    plt.savefig('estimate')
    pass


def plot_var(baseline_var, improved_var, x_values, save=None):    
    _, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(x_values, baseline_var, label='baseline variance')
    axs[0].plot(x_values, improved_var, label='improved variance')
    axs[0].legend()
    axs[1].plot(x_values, improved_var / baseline_var, label='improved / baseline')
    axs[1].legend()
    axs[1].set_ylim(bottom=-0.5, top=2)
    plt.title(save.name[:-4])
    plt.savefig(save)
    pass
       

def evaluation_pipeline(path: Path):
    results_dict = rec_dd()
    # _, ax = plt.subplots(1, 1, figsize=(5,5))
    # ax.set_ylim(bottom=-0.5, top=2) 
    # ax.set_xlim(left=-6, right=6) 
    for f in path.glob('**/*'):
        try:
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
            baseline_sse, baseline_var  = evaluate(baseline_df)
            improved_sse, improved_var = evaluate(improved_df)
        except(Exception):
            continue
        
        # Extract variables from file name
        p_edge, n_normal, n_edge = variables_from_filename(f.name)

        # plot_var(baseline_var, improved_var, baseline_df['x'], save=Path('img')/ (f.name + '_var.png'))
        # ax.plot(baseline_df['x'], improved_var/baseline_var, alpha =0.5, color=p_edge_colors[p_edge])
        
        # Compute mean var and delta sse
        delta_sse = improved_sse.mean() - baseline_sse.mean()
        improved_var = improved_var.mean()
        baseline_var = baseline_var.mean()
    
        results_dict[p_edge][n_normal][n_edge] = {'delta_sse': delta_sse, 'baseline_var': baseline_var, "improved_var": improved_var}
    plt.savefig('test')
    with open( path / 'results.json', 'w') as fp:
        json.dump(results_dict, fp, indent=2)
            
    
        
    # res = {}
    # for n in [100, 1000, 10_000]:
    #     for f in path.rglob(f"*{n}.*{n}.*.csv"):
    #         try:
    #             baseline_df = pd.read_csv(f.parent / (f.parent.name + '.baseline.csv'))
    #             improved_df = pd.read_csv(f.parent / (f.parent.name + '.improved.csv'))
    #             baseline_sse, improved_var  = evaluate(baseline_df)
    #             improved_sse, improved_var = evaluate(improved_df)
    #         except(Exception):
    #             continue
            
    #         main(f.parent)
        
    #         delta_sse = improved_sse.mean() - baseline_sse.mean()
    #         improved_var = improved_var.mean()
    #         baseline_var = baseline_var.mean()
            
    #         p_edge = f.name.split('.n')[0][7:]
            
    #         res[p_edge] = (improved_sse - baseline_sse)    
            
    #     res_a[n] = {key: val for key, val in sorted(res.items(), key = lambda ele: ele[0])}
        
    # plt.figure()
    # for n in res_a.keys():
    #     plt.plot(res_a[n].keys(), res_a[n].values(), label=f'$n=m={n}$')
    # plt.legend()
    # plt.xlabel('$P$ edge')
    # plt.ylabel('$\Delta$ SSE')
    
    # plt.title('$\Delta$ SSE against the probability of observing edge scenarios')
    # plt.savefig('img/delta_SSE')
    


if __name__ == "__main__":
    path = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates/nn_approach/bivariate_guassian_a')
    evaluation_pipeline(path)
    

    

    
        
        
    