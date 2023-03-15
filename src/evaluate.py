from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def evaluate(dataframe):
    run_cols = [col for col in dataframe if col.startswith('run')]
    estimates = dataframe[run_cols].to_numpy()
    true = dataframe['true'].to_numpy()
    x_values = dataframe['x'].to_numpy()

    true_matrix = np.tile(true, (100, 1))
    # TODO: Check for alternative metric
    sse = np.square(estimates.T - true_matrix).sum(axis=1).mean()
    mean = np.mean(estimates.T, axis=0)
    upper = np.percentile(estimates.T, 95, axis=0)
    lower = np.percentile(estimates.T, 5, axis=0)
    var = np.var(estimates.T, axis=0)

    # plot(true, mean, upper, lower, x_values)

    return true, sse, mean, var, x_values


def plot(true, mean, upper, lower, x_values):
    fig, axs = plt.subplots(1, 1)

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


def main(path: Path):
    baseline_path = path / (path.name + '.baseline.csv')
    baseline_df = pd.read_csv(baseline_path)
    true, baseline_mse, baseline_mean, baseline_var, x_values = evaluate(
        baseline_df)

    improved_path = path / (path.name + '.improved.csv')
    improved_df = pd.read_csv(improved_path)
    _, improved_mse, improved_mean, improved_var, _ = evaluate(improved_df)

    plot_var(baseline_var, improved_var, x_values, save=Path('img')/ (path.name + '_var.png'))


if __name__ == "__main__":
    path = Path('/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates/bivariate_guassian_a')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    for n in [100, 1000]:
        res = {}
        for f in path.rglob(f"*{n}.*{n}.*.csv"):
            try:
                baseline_df = pd.read_csv(f.parent / (f.parent.name + '.baseline.csv'))
                improved_df = pd.read_csv(f.parent / (f.parent.name + '.improved.csv'))
                _,  baseline_sse, _, _,_ = evaluate(baseline_df)
                _,  improved_sse, _, _,_ = evaluate(improved_df)
            except(Exception):
                continue
            
            main(f.parent)
        
            print(f"{f.name}:\t delta sse: {improved_sse - baseline_sse}")
            p_edge = f.name.split('.n')[0][7:]
            
            res[p_edge] = (improved_sse - baseline_sse)    
        res = {key: val for key, val in sorted(res.items(), key = lambda ele: ele[0])}
    
        plt.plot(res.keys(), res.values(), label=f'$n=m={n}$')
    plt.legend()
    plt.xlabel('$P$ edge')
    plt.ylabel('$\Delta$ SSE')
    
    plt.title('$\Delta$ SSE against the probability of observing edge scenarios')
    plt.savefig('img/delta_SSE')
    
    

    
        
        
    
