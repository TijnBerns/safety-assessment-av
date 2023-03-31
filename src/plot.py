import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from evaluate import evaluate
from config import Config as cfg 
from utils import variables_from_filename
from evaluate import evaluation_pipeline

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(cfg.p_edge)))
bounds = cfg.p_edge
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
p_edge_colors = {str(k): v for k, v in zip(cfg.p_edge, colors)}


def plot_diff_grouped_by_n(results_df: pd.DataFrame, save:str):
    _, axs = plt.subplots(1,2, figsize=(10,5) )
    axs[0].set_ylabel('improved var / baseline var')
    axs[1].set_ylabel('improved error / baseline error')
    
    for ax in axs:
        ax.set_xlabel('p_edge')
    
    for n in pd.unique(results_df['n_edge']):
        p = results_df.where((results_df['n_edge'] == n) & (results_df['n_normal'] == n)).dropna()
        p = p.sort_values(by='p_edge')

        axs[0].plot(p['p_edge'], p['improved_var'] - p['baseline_var'], label=f'n={n}')
        axs[1].plot(p['p_edge'], p['improved_sse'] - p['baseline_sse'], label=f'n={n}')
    plt.tight_layout()
    plt.savefig(save, bbox_inches = "tight")


def plot_diff_grouped_by_p_edge(results_df: pd.DataFrame, save:str):
    _, axs = plt.subplots(1,2, figsize=(10,5) )
    axs[0].set_ylabel('improved var - baseline var')
    axs[1].set_ylabel('improved error - baseline error')
    
    for ax in axs:
        ax.set_xlabel('n')
    
    for p_edge in pd.unique(results_df['p_edge']):
        p = results_df.where((results_df['n_edge'] == results_df['n_normal']) & (results_df['p_edge'] == p_edge)).dropna()
        p = p.sort_values(by='n_normal')

        axs[0].plot(p['n_normal'], p['improved_var'] - p['baseline_var'], label=f'p_edge={p_edge}')
        axs[1].plot(p['n_normal'], p['improved_sse'] - p['baseline_sse'], label=f'p_edge={p_edge}')
    plt.tight_layout()
    plt.savefig(save, bbox_inches = "tight")

    
def plot_diff(path: Path, save): 
    _, axs = plt.subplots(1,2, figsize=(10,5))
    plt.tight_layout()
    
    # Set limits and x-labels
    for ax in axs:
        ax.set_ylim(bottom=-0.5, top=2)
        # ax.set_xlim(left=-5, right=5)
        ax.set_xlabel('x')
    
    axs[0].set_ylabel('improved var / baseline var')
    axs[1].set_ylabel('improved error / baseline error')
    
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir():
            continue
        try:
            # This fails if not all columns in the cvs have equal rows
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
            baseline_sse, baseline_var = evaluate(baseline_df)
            improved_sse, improved_var = evaluate(improved_df)
        except Exception as e:
            print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
            continue
        
        # Extract variables from file name
        p_edge, _, _ = variables_from_filename(f.name)
        
        axs[0].plot(baseline_df['x'], improved_var/baseline_var, alpha =0.5, color=p_edge_colors[p_edge])
        axs[1].plot(baseline_df['x'], improved_sse/baseline_sse, alpha =0.5, color=p_edge_colors[p_edge])
        
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='p_edge')
    plt.tight_layout()
    plt.savefig(save, bbox_inches = "tight")
        
        
def plot_pdf(path: Path, save):
    _, ax = plt.subplots(1,2, figsize=(10,5))
    plt.tight_layout()
    
    # Set limits and x-labels
    ax[0].set_xlabel('x') 
    ax[1].set_xlabel('x') 
    ax[0].set_title('baseline')
    ax[1].set_title('improved')

    
    for f in tqdm(list(path.glob('**/*'))):
        if not f.is_dir():
            continue
        try:
            # This fails if not all columns in the cvs have equal rows
            baseline_df = pd.read_csv(f / (f.name + '.baseline.csv'))
            improved_df = pd.read_csv(f / (f.name + '.improved.csv'))
        except Exception as e:
            print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
            continue
        
        # Extract variables from file name
        p_edge, _, _ = variables_from_filename(f.name)
        
        run_cols = [col for col in baseline_df if col.startswith('run')]
        baseline_estimates = baseline_df[run_cols].to_numpy()
        
        run_cols = [col for col in improved_df if col.startswith('run')]
        improved_estimates = improved_df[run_cols].to_numpy()
        

        ax[0].plot(baseline_df['x'], baseline_estimates.mean(axis=1), alpha =0.5, color=p_edge_colors[p_edge])
        ax[1].plot(baseline_df['x'], improved_estimates.mean(axis=1), alpha =0.5, color=p_edge_colors[p_edge])
        
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='p_edge')
    plt.tight_layout()
    plt.savefig(save, bbox_inches = "tight")
    
            
if __name__ == "__main__":
    for estimates_path in Path("/home/tijn/CS/Master/SA_Automated_Vehicles/safety-assessment-av/estimates/kde_combined_estimator").glob('*/'):
        if not estimates_path.is_dir():
            continue

        evaluation_pipeline(estimates_path)
        results_path = estimates_path / 'results.csv'
        results_df = pd.read_csv(results_path)
        
        plot_diff(estimates_path, save=Path('img') / (estimates_path.parent.name + '.' + estimates_path.name + '.results.png'))
        plot_diff_grouped_by_n(results_df, save=Path('img') / (estimates_path.parent.name + '.' + estimates_path.name + '.grouped_by_n.png'))
        plot_pdf(estimates_path, save=Path('img') / (estimates_path.parent.name + '.' + estimates_path.name + '.pdf_estimate.png'))
    
    # plot_diff_grouped_by_p_edge(results_df, save=Path('img') / (estimates_path.name + '.grouped_by_p_edge.png'))
    
    
    
    