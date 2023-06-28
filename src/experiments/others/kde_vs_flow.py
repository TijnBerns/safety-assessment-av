import sys
sys.path.append('src')
sys.path.append('src/data')
sys.path.append('src/experiments')
sys.path.append('src/experiments/flow')

import numpy as np
import scipy.stats
from pprint import pprint
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from nflows import flows, distributions, transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from nflows import transforms
from nflows.flows.base import Flow
from nflows.nn.nets import ResidualNet
import pytorch_lightning as pl
from torch.utils.data import DataLoader



from flow.flow_module import FlowModule

# from flow_module import create_flow
from parameters import get_parameters
import utils
import json
from pathlib import Path
import pandas as pd
from power import Power

N_DIM = list(range(2,6))
N_SAMPLES = [100, 1000, 10_000, 100_000]
N_MODELS = 5

def create_flow(num_features, num_layers: int=5):
    # Base distribution
    base_dist = distributions.StandardNormal((num_features,))
    
    # Transform
    f = []
    for _ in range(num_layers):
        f.append(transforms.RandomPermutation(features=num_features))
        f.append(transforms.LULinear(features=num_features, identity_init=True))
        f.append(transforms.AffineCouplingTransform(
            mask=torch.zeros(num_features), 
            transform_net_create_fn = lambda in_features, out_features: 
                ResidualNet(hidden_features=16,
                            in_features=in_features,
                            out_features=out_features)
                ))
    transform = transforms.CompositeTransform(f)    
    
    # Create and return flow
    return Flow(transform, base_dist)

def fit_flow(flow: Flow, data, num_steps):
    optimizer = optim.Adam(flow.parameters())
    prev_loss = -10_000
    alpha = 0.001
    beta = 1
    patience = 0
    for i in range(num_steps):
        x = torch.tensor(data, dtype=torch.float32)
        optimizer.zero_grad()
        
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'iteration {i}: {loss, beta}')
    
        beta = np.abs(prev_loss - loss.item())
        if  beta <= alpha:
            patience += 1
            if patience == 5:
                return flow
        else: 
            patience = 0
        
        prev_loss = loss.item()
        
        
    return flow

def mse(true, estimate):
    return np.mean(np.square(true - estimate))
        

def get_true_distribution(num_dim):
    if num_dim == 1:
        return scipy.stats.norm()
    mean = np.zeros(num_dim)
    cov = np.diag(np.ones(num_dim))
    return scipy.stats.multivariate_normal(mean=mean, cov=cov)  


if __name__ =='__main__': 
    dataset = Power
    train_full = dataset(split='_train').data
    test_full = dataset(split='_test').data
    args = get_parameters('power')
    results = []
    device, _ = utils.set_device()
    
    for num_dim in N_DIM:
        # Generate test samples
        distribution = get_true_distribution(num_dim)
        
        for num_samples in N_SAMPLES:
            
            for m in range(50):
                # Normalize data
                train = train_full[:,:num_dim]
                train = train[np.random.choice(list(range(len(train))), num_samples, replace=False)]
                test = test_full[:,:num_dim]
                if num_dim == 1:
                    train = train.reshape((-1, 1))
                    test = test.reshape((-1, 1))

                # Fit KDE
                kde_start = datetime.now()
                kde = scipy.stats.gaussian_kde(train.T)
                kde_time = (datetime.now() - kde_start).total_seconds()
                kde_pdf = np.mean(kde.logpdf(test.T))
            
                # Fit Flow
                flow = FlowModule(features=num_dim, dataset=dataset(), stage=3, args=args)
                dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
                trainer = pl.Trainer(max_steps=args.training_steps_stage_2,
                                     inference_mode=False,
                                     callbacks=[EarlyStopping(monitor="log_density", mode="max",  min_delta=0.01, patience=3,)],
                                     enable_checkpointing=False,
                                     log_every_n_steps=args.logging_interval,
                                     accelerator=device)
                
                flow_start = datetime.now()
                trainer.fit(flow, dataloader)
                flow_time = (datetime.now() - flow_start).total_seconds()
                with torch.no_grad():
                    flow_pdf = flow.flow.log_prob(torch.tensor(test, dtype=torch.float32))
                    flow_pdf = float(np.mean(flow_pdf.detach().numpy()))
                    
                results.append(
                    {
                        'dim': num_dim,
                        'samples': num_samples,
                        'kde': kde_pdf,
                        'kde_time': kde_time,
                        'flow': flow_pdf,
                        'flow_time': flow_time
                    }
                )   
                
                utils.save_json(path=Path('kde_vs_flow.json'), data={'data':results})
                
    