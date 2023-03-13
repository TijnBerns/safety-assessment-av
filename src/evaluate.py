from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate(dataframe):
    run_cols = [col for col in dataframe if col.startswith('run')]
    estimates = dataframe[run_cols].to_numpy()
    true = dataframe['true'].to_numpy()
    x_values = dataframe['x'].to_numpy()
    
    true_matrix  = np.tile(true, (100,1))
    # TODO: Check for alternative metric
    mse = np.square(estimates.T - true_matrix).mean(axis=1)
    mean = np.mean(estimates.T, axis=0)
    upper = np.percentile(estimates.T, 95, axis=0)
    lower = np.percentile(estimates.T, 5, axis=0)
    var = np.var(estimates.T, axis=0)

    plot(true, mean, upper, lower, x_values)
    
    return true, mse, mean, var, x_values

def plot(true, mean, upper, lower, x_values):
    fig, axs = plt.subplots(1,1)
    
    # Plot predictions
    axs.plot(x_values, mean)
    axs.plot(x_values , upper, color='tab:blue')
    axs.plot(x_values, lower,color='tab:blue')
    axs.fill_between(x_values, upper, lower, color='tab:blue', alpha=0.5)
    plt.savefig('estimate')


    
def plot_var(baseline_var, improved_var, x_values):
    fig, axs = plt.subplots(1,1)
    axs.plot(x_values, baseline_var, label='baseline var')
    axs.plot(x_values, improved_var,label='improved var')
    axs.legend()
    plt.savefig('variance')
    
        
    

def main(path: Path):
    baseline_path = path / (path.name + '.baseline.csv')
    baseline_df = pd.read_csv(baseline_path)
    true, baseline_mse, baseline_mean, baseline_var, x_values = evaluate(baseline_df)
    
    improved_path = path / (path.name + '.improved.csv')
    improved_df = pd.read_csv(improved_path)
    _, improved_mse, improved_mean, improved_var, _ = evaluate(improved_df)
    
    plot_var(baseline_var, improved_var, x_values)
    
    

    
    


if __name__ == "__main__":
    path = Path('/home/tberns/safety-assessment-av/estimates/bivariate_guassian_a/p_edge_0.04.n_normal_100.n_edge_100')
    main(path)