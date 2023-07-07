import sys
sys.path.append('src')


from pathlib import Path
from typing import Dict
import evaluate
import pandas as pd
import numpy as np
import click
import utils


def write_results(path: Path, row) -> None:
    if not path.exists():
        df = pd.DataFrame(row)
    else:
        df = pd.concat([pd.read_csv(path), pd.DataFrame(row)])
    utils.save_csv(path, df)
    return


def compute_llh(version: str, true: str, dataset: str) -> Dict[str, float]:
    utils.seed_all(2023)
    evaluator = evaluate.Evaluator(dataset=dataset, version=version)

    # Load checkpoints
    best = evaluate.get_llh(version)
    true, _ = evaluate.get_checkpoint(true)
    true = evaluate.get_best_checkpoint(true)
    
    # Initialize list for storing results
    mse = np.zeros((len(best), 3))
    llh = np.zeros_like(mse)
    llr = np.zeros_like(mse)

    # Compute log likelihood ratios
    for i, llh_tensor in enumerate(best):
        try:
            print(f'Evaluating {llh_tensor}')
            llh[i] = evaluator.compute_llh(llh_tensor)
            mse[i] = evaluator.compute_mse(true, llh_tensor)
            llr[i] = evaluator.compute_lr(true, llh_tensor)
        except:
            print('could not evaluate probably due to correpted file')

    llh =  np.array(llh)[np.unique(np.nonzero(llh)[0])]
    mse = np.array(mse)[np.unique(np.nonzero(mse)[0])]
    llr = np.array(llr)[np.unique(np.nonzero(llr)[0])]
            
    # Aggregate results
    mse_mean = np.mean(mse, axis=0)
    mse_std = np.std(mse, axis=0)
    llh_mean = np.mean(llh, axis=0)
    llh_std = np.std(llh, axis=0)
    llr_mean = np.mean(llr, axis=0)
    llr_std = np.std(llr, axis=0)
    row = [{
        'version': version,
        'mse_all': mse_mean[0],
        'mse_all_std': mse_std[0],
        'mse_non_event': mse_mean[1],
        'mse_non_event_std': mse_std[1],
        'mse_event': mse_mean[2],
        'mse_event_std': mse_std[2],
        'llh_all': llh_mean[0],
        'llh_all_std': llh_std[0],
        'llh_non_event': llh_mean[1],
        'llh_non_event_std': llh_std[1],
        'llh_event': llh_mean[2],
        'llh_event_std': llh_std[2],
    }]

    print(row)

    # Write results to file
    write_results(path=Path('results.csv'), row=row)

    return row


@click.command()
@click.option('--version', type=str)
@click.option('--true', type=str)
@click.option('--dataset', default='hepmass')
def main(version: str, true: str, dataset: str):
    compute_llh(version, true, dataset)


if __name__ == "__main__":
    main()
