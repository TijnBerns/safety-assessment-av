import sys
sys.path.append('src')
sys.path.append('src/kde')

import utils
from kde.config import SharedParameters as params
import click
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

LABELS = params.LABELS
FIGSIZE = params.FIGSIZE


def plot_overall_improvement_uv(path: Path, save: str):
    """Creates a plot summarizing all results: mean improvement and 95% confidence interval of improvement

    Args:
        path (Path): The root where all results.csv files are located 
        save (str): Base name of saved images (img/{save}.mse.pgf and img/{save}.std.pgf )
    """
    files = list(path.rglob('*results.csv'))
    num_eval = len(pd.read_csv(files[0])['x'])

    num_files = 0
    for i, f in enumerate(files):
        p_event, n_norm,n_event,corr = utils.variables_from_filename(f.name)
        if not(corr == 0.5 and n_norm == n_event):
            continue
        num_files += 1
        
    results = np.zeros((6, num_files, num_eval))
    i = 0
    for f in files:
        p_event, n_norm,n_event,corr = utils.variables_from_filename(f.name)
        if not(corr == 0.5 and n_norm == n_event):
            continue


        df = pd.read_csv(f)
        x_values = df["x"]
        results[0][i] = df["baseline_mse"]
        results[1][i] = df["baseline_mean"]
        results[2][i] = df["baseline_std"]
        results[3][i] = df["improved_mse"]
        results[4][i] = df["improved_mean"]
        results[5][i] = df["improved_std"]
        i+=1

    # Plot std improved / std baseline
    _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    std = results[5] / results[2]
    axs.plot(x_values, std.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(std, 2.5, axis=0),
        np.percentile(std, 97.5, axis=0),
        alpha=0.25,
        label='$2.5-97.5$ percentile'
    )
    axs.legend()
    axs.set_xlabel('$x$')
    axs.set_ylabel('improved std / baseline std')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'img/{save}.std.pgf')

    # Plot mse improved / mse baseline
    _, axs = plt.subplots(1, 1, figsize=FIGSIZE)
    mse = results[3] / results[0]
    axs.plot(x_values, mse.mean(axis=0), label='mean')
    axs.fill_between(
        x_values,
        np.percentile(mse, 2.5, axis=0),
        np.percentile(mse, 97.5, axis=0),
        alpha=0.25,
        label='$2.5-97.5$ percentile'
    )
    axs.set_xlabel('$x$')
    axs.set_ylabel('improved mse / baseline mse')
    axs.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'img/{save}.mse.pgf')
    
def plot_pdf_bv(path: Path):
    """Temporary method used to plot 2d estimates

    Args:
        path (Path): _description_
    """
    # for f in tqdm(list(path.glob('**/*'))):
    #     if not f.is_dir() or f.name.find('results') != -1:
    #         continue
    #     try:
    #         # This fails if not all columns in the cvs have equal rows
    #         baseline_df = load_json_as_df(f / (f.name + '.baseline.json'))
    #         improved_df = load_json_as_df(f / (f.name + '.improved.json'))
    #         print(baseline_df)
    #         # baseline_mse, baseline_mean, baseline_std, _, _ = evaluate(baseline_df)
    #         # improved_mse, improved_mean, improved_std, _ ,_ = evaluate(improved_df)
    #     except Exception as e:
    #         print(f"WARNING: Could not evaluate for {str(f)}\n{e}")
    #         continue
    df = utils.load_json_as_df(path)

    x_values = df['x'].to_list()

    a = int(np.sqrt(len(x_values)))
    x_values = np.array(x_values)
    x_values = x_values.reshape((a, a, 2))
    X = x_values[:, :, 0]
    Y = x_values[:, :, 1]
    true = df['true'].to_list()
    true = np.array(true)
    true = true.reshape(a, a)
    estimate = df['estimate'].to_list()
    estimate = np.array(estimate)
    estimate= estimate.reshape(a,a)

    # # run_cols = [col for col in df if col.startswith('run')]
    # # estimates = improved_df[run_cols].to_numpy()
    # # mean = np.mean(estimates.T, axis=0)
    # # mean = mean.reshape(a, a)

    # # error = np.log(np.square(true - estimate))
    # error = np.abs(true - mean)

    fig, axs = plt.subplots(1, 2)
    axs[1].contourf(X, Y, estimate)
    axs[0].contourf(X, Y, true)

    plt.show()
    
    
@click.command()
@click.option('--path', '-p', type=Path)
@click.option('--save', '-s', type=str, default='PLACEHOLDER')
@click.option('--dim', '-d', type=int, default=1)
def main(path: Path, save: str, dim: int):
    if dim == 1:
        plot_overall_improvement_uv(path, save)
    elif dim == 2:
        plot_pdf_bv(path)
    

if __name__ == "__main__":
    main()
