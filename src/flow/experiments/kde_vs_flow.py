import sys
sys.path.append('src')
sys.path.append('src/flow')

import numpy as np
import scipy.stats
from pprint import pprint
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import click

from flow.flow_module import FlowModule

# from flow_module import create_flow
import flow.parameters as parameters
import matplotlib.pyplot as plt
import utils
from pathlib import Path

N_DIM = list(range(2,8))
N_SAMPLES = [100, 1000, 10_000, 100_000]
N_MODELS = 5



def mse(true, estimate):
    return np.mean(np.square(true - estimate))

def plot(path: str= 'kde_vs_flow_old_2.json'):
    path = Path(path)
    data = utils.load_json(path)['data']
    df = pd.DataFrame(data)
    group = df.groupby(['num_samples'])
    
    
    num_dim = np.sort(df['num_dim'].unique())
    num_samples = np.sort(df['num_samples'].unique())
    
    fig, axs = plt.subplots(len(num_samples),2, sharex=True)
    fig.subplots_adjust(hspace=0)
    for i, n in enumerate(num_samples):
        gd = group.get_group(n)
        mean = gd.groupby('num_dim').mean()
        
        axs[i][0].plot(mean['kde_log_prob'])
        axs[i][0].plot(mean['flow_log_prob'])
        axs[i][1].plot(mean['kde_eval_time'])
        axs[i][1].plot(mean['flow_eval_time'])
    plt.savefig('test')
     
def remove_outliers(pdf, data):
    Q1 = np.percentile(pdf, 25, method='midpoint')
    Q3 = np.percentile(pdf, 75, method='midpoint')
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = np.logical_and(pdf >= lower, pdf <= upper)
    return pdf[mask], data[mask]
    
    
def run_exp(true, dataset):
    device, _ = utils.set_device()
    
    # Get arguments and dataset
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    results = []
    
    # Initialize data
    train_full = dataset(split='_train').data
       
    for num_dim in N_DIM:    
        true = scipy.stats.gaussian_kde(train_full[:,:num_dim].T)
        test = true.resample(10_000).T
        true_pdf = true.logpdf(test.T)
        true_pdf, test = remove_outliers(true_pdf, test)

        for num_samples in N_SAMPLES:
            
            for _ in range(N_MODELS):
                res = {
                    'num_dim': num_dim,
                    'num_samples': num_samples
                }
                
                # Normalize data
                train = true.resample(num_samples).T
                
                # Fit KDE
                start = datetime.now()
                kde = scipy.stats.gaussian_kde(train.T)
                res['kde_train_time'] = (datetime.now() - start).total_seconds()
                
                # Evaluate KDE
                start =  datetime.now()
                kde_pdf = kde.logpdf(test.T)
                res['kde_eval_time'] = (datetime.now() - start).total_seconds()
                res['kde_log_prob'] = np.mean(kde_pdf)
                res['kde_mse'] = mse(true_pdf, kde_pdf)

                # Fit Flow
                flow = FlowModule(features=num_dim, dataset=dataset(), stage=2, args=args)
                dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
                trainer = pl.Trainer(max_steps=args.training_steps,
                                     inference_mode=False,
                                     callbacks=[EarlyStopping(monitor="log_density", mode="max",  min_delta=0.00, patience=3,)],
                                     enable_checkpointing=False,
                                     log_every_n_steps=args.logging_interval,
                                     accelerator=device)
                
                # Fit flow
                start = datetime.now()
                trainer.fit(flow, dataloader)
                res['flow_train_time'] = (datetime.now() - start).total_seconds()
                
                # Eval flow
                with torch.no_grad():
                    test = torch.Tensor(test)
                    start = datetime.now()
                    flow_pdf = flow.flow.log_prob(torch.tensor(test, dtype=torch.float32))
                    res['flow_eval_time'] = (datetime.now() - start).total_seconds()
                
                flow_pdf = flow_pdf.detach().numpy()
                res['flow_log_prob'] = float(np.mean(flow_pdf))
                res['flow_mse'] = mse(true_pdf, flow_pdf)
                    
                results.append(res)
                utils.save_json(path=Path('kde_vs_flow.json'), data={'data':results})
                
                print(res)
    

@click.command()
@click.option('--true',type=str, default='')
@click.option('--dataset',type=str, default='gas')
@click.option('--todo', default='run')
def main(true: str, dataset: str, todo: str):
    if todo == 'run':
        run_exp(true, dataset)
    else:
        plot()
 
                
            

if __name__ =='__main__': 
    main()
    

    