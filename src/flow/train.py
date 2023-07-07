import sys
sys.path.append('src')
sys.path.append('src/data')

import utils
import parameters
from flow_module import FlowModule, FlowModuleWeighted

import click
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from data.base import CustomDataset
import data.preprocess
import evaluate

def create_checkpointer(prefix:str=None):
    if prefix is None:
        pattern = "epoch_{epoch:04d}.step_{step:09d}.log_density_{val_log_density:.2f}"
    else:
        pattern = prefix + ".epoch_{epoch:04d}.step_{step:09d}.log_density_{val_log_density:.2f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=50,
        every_n_train_steps=100,
        monitor="val_log_density",
        mode='max',
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )
    return checkpointer

def train_val_split(data, frac):
    split_idx = int(frac * len(data))
    return  data[:split_idx], data[split_idx:]

def create_data_loaders(dataset:CustomDataset, batch_size:int, dataset_type:str):
    val = DataLoader(dataset(split='val'), shuffle=False, batch_size=batch_size, num_workers=2)
    
    if dataset_type == 'all':
        train = DataLoader(dataset(split='_train'), shuffle=True, batch_size=batch_size, num_workers=2)
        val = DataLoader(dataset(split='_val'), shuffle=False, batch_size=batch_size, num_workers=2)
        return train, None, val
    
    elif dataset_type == 'split':
        train_normal = DataLoader(dataset(split='normal_train'), shuffle=True, batch_size=batch_size, num_workers=2)
        train_event = DataLoader(dataset(split='event_train'), shuffle=True, batch_size=batch_size, num_workers=2)
        return train_normal, train_event, val
    
    elif dataset_type == 'weighted' or dataset_type=='zero_weight':
        normal = dataset(split='normal_train').data 
        event = dataset(split='event_train').data  
        train = DataLoader(np.concatenate((normal, event)), shuffle=True, batch_size=batch_size, num_workers=2)
        return train, None, val 
    
    elif dataset_type == 'sampled_split':
        train_normal = DataLoader(dataset(split='normal_sampled'), shuffle=True, batch_size=batch_size, num_workers=2)
        train_event = DataLoader(dataset(split='event_sampled'), shuffle=True, batch_size=batch_size, num_workers=2)
        val = DataLoader(dataset(split='val_sampled'), shuffle=False, batch_size=batch_size, num_workers=2)
        return train_normal, train_event, val
    
    elif dataset_type == 'sampled_zero_weight' or dataset_type == 'sampled_weighted':
        normal = dataset(split='normal_sampled').data 
        event = dataset(split='event_sampled').data  
        train = DataLoader(np.concatenate((normal, event)), shuffle=True, batch_size=batch_size, num_workers=2)
        val = DataLoader(dataset(split='val_sampled'), shuffle=False, batch_size=batch_size, num_workers=2)
        return train, None, val 
    else: 
        raise ValueError

def create_module(dataset: CustomDataset, features: int, dataset_type: str, weight:float, args: parameters.Parameters, stage: int):
    if dataset_type in ['weighted', 'sampled_weighted']:
        return FlowModuleWeighted(features=features, dataset=dataset, args=args, stage=stage, weight=weight)
    else:
        return FlowModule(features=features, dataset=dataset, args=args, stage=stage, weight=weight)
   


@click.command()
@click.option('--dataset', type=str, default='gas')
@click.option('--dataset_type', type=str, default='sampled_weighted') 
@click.option('--weight', type=float, default=None) 
def train(dataset:str, dataset_type: str, weight: float):    
    # Set seeds for reproducibility 
    utils.seed_all(2023)
    dataset_str = dataset
    
    # Construct data loaders
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    normal_train, _, val = create_data_loaders(dataset, args.batch_size, dataset_type)
    print(f"\n\n!!!!!!!!!!!{normal_train.dataset.data.shape}!!!!!!!!!!!!!!!!\n\n")
    
    # Get device
    device, version = utils.set_device()
    
    # create model
    if weight is None:
        weight = data.preprocess.compute_event_weight_np(data=normal_train.dataset, xi=dataset().xi, threshold=dataset().threshold)
        
    features = normal_train.dataset.data.shape[1]
    flow_module = create_module(features=features, dataset=dataset(), dataset_type=dataset_type, args=args, stage=1, weight=weight)

    
    # Initialize checkpointers
    checkpointer = create_checkpointer()

    # Fine-tune on normal data
    trainer = pl.Trainer(max_steps=args.training_steps,
                                 inference_mode=False,
                                 callbacks=[checkpointer],
                                 log_every_n_steps=args.logging_interval,
                                 accelerator=device)
    trainer.fit(flow_module, normal_train, val)
    
    # Evaluate models (computes log likelihood tensor)
    test_set = 'all' if 'sampled' in dataset_type else 'normal'     
    evaluate.evaluate(dataset=dataset_str, version=version, test_set=test_set)


if __name__ == "__main__":
    train()
