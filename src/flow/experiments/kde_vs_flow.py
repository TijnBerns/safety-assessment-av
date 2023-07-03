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

import flow.evaluate as evaluate

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import click

from flow.flow_module import FlowModule

# from flow_module import create_flow
import flow.parameters as parameters

import utils
from pathlib import Path

N_DIM = list(range(2,8))
N_SAMPLES = [100, 1000, 10_000, 100_000]
N_MODELS = 5



def eval_kde():
    pass

def eval_flow():
    pass



@click.command()
@click.option('--true',type=str, default='')
@click.option('--dataset',type=str, default='gas')
def main(true: str, dataset: str):
    device, _ = utils.set_device()
    
    # Get arguments and dataset
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    results = []
    
    # Initialize data
    train_full = dataset(split='_train').data
    test_full = dataset(split='_test')
        

    for num_dim in N_DIM:        
        for num_samples in N_SAMPLES:
            
            for _ in range(N_MODELS):
                res = {
                    'num_dim': num_dim,
                    'num_samples': num_samples
                }
                
                # Normalize data
                train = train_full[:,:num_dim]
                train = train[np.random.choice(list(range(len(train))), num_samples, replace=False)]
                test = test_full[:,:num_dim]
                if num_dim == 1:
                    train = train.reshape((-1, 1))
                    test = test.reshape((-1, 1))

                # Fit KDE
                start = datetime.now()
                kde = scipy.stats.gaussian_kde(train.T)
                res['kde_train_time'] = (datetime.now() - start).total_seconds()
                
                # Evaluate KDE
                start =  datetime.now()
                kde_pdf = kde.logpdf(test.T)
                res['kde_eval_time'] = (datetime.now() - start).total_seconds()
                res['kde_log_prob'] = np.mean(kde_pdf)
                
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
                    start = datetime.now()
                    flow_pdf = flow.flow.log_prob(torch.tensor(test, dtype=torch.float32))
                    res['flow_eval_time'] = (datetime.now() - start).total_seconds()
                    res['flow_log_prob'] = float(np.mean(flow_pdf.detach().numpy()))
                    
                results.append(res)
                utils.save_json(path=Path('kde_vs_flow.json'), data={'data':results})
                
            

if __name__ =='__main__': 
    main()
    

    