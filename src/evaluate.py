from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re 
import json

def rec_dd():
    return defaultdict(rec_dd)
    

def evaluate(dataframe: pd.DataFrame):
    # Collect all estimates in single ndarray
    run_cols = [col for col in dataframe if col.startswith('run')]
    estimates = dataframe[run_cols].to_numpy()
    
    # The true pdf values
    true = dataframe['true'].to_numpy()
    true_matrix = np.tile(true, (100, 1))

    # Compute various metrics
    sse = np.square(estimates.T - true_matrix).sum(axis=1)
    mean = np.mean(estimates.T, axis=0)
    var = np.var(estimates.T, axis=0)

    return sse, var


def plot_estimate(true, mean, upper, lower, x_values):
    _, axs = plt.subplots(1, 1)

    # Plot predictions
    axs.plot(x_values, mean)
    axs.plot(x_values, upper, color='tab:blue')
    axs.plot(x_values, lower, color='tab:blue')
    axs.fill_between(x_values, upper, lower, color='tab:blue', alpha=0.5)
    axs.plot(x_values, true, color='tab:orange')
    plt.savefig('estimate')


def plot_var(baseline_var, improved_var, x_values, save=None):
    _, axs = plt.subplots(1, 1)
    axs.plot(x_values, baseline_var, label='baseline variance')
    axs.plot(x_values, improved_var, label='improved variance')
    axs.legend()
    
    plt.savefig(save)


def evaluation_pipeline(path: Path):
    results_dict = rec_dd()
    for f in path.glob('**/*'):
        try:
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
            baseline_sse, baseline_var  = evaluate(baseline_df)
            improved_sse, improved_var = evaluate(improved_df)
        except(Exception):
            continue
        
        plot_var(baseline_var, improved_var, baseline_df['x'], save=Path('img')/ (f.name + '_var.png'))
        
        # Compute mean var and delta sse
        delta_sse = improved_sse.mean() - baseline_sse.mean()
        improved_var = improved_var.mean()
        baseline_var = baseline_var.mean()
        
        # Extract variables from file name
        f_split = f.name.split('.')
        p_edge = f_split[0][7:] + '.' + f_split[1]
        n_normal = f_split[2][9:]
        n_edge = f_split[3][7:]
        
        
        results_dict[p_edge][n_normal][n_edge] = {'delta_sse': delta_sse, 'baseline_var': baseline_var, "improved_var": improved_var}
    
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
    path = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates/bivariate_guassian_a')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    evaluation_pipeline(path)
    

    

    
        
        
    
