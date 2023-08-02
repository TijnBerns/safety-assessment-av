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
from typing import Tuple

def create_checkpointer():
    """Creates a checkpointer. That saves the top 50 models every 100 training steps.

    Returns:
        pytorch_lightning.callbacks.ModelCheckpoint: Checkpointer.
    """
    pattern = "epoch_{epoch:04d}.step_{step:09d}.log_density_{val_log_density:.2f}"
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

def create_data_loaders(dataset:CustomDataset, batch_size:int, dataset_type:str) -> Tuple[DataLoader, DataLoader]:
    """Creates train and val dataloaders corresponding to provedided dataset and dataset type

    Args:
        dataset (CustomDataset): Dataset
        batch_size (int): Batch size of train an val dataloader
        dataset_type (str): What data to use. Choices: all, normal, weighted, sampled_normal, or sampled_weighted.

    Raises:
        ValueError: Provided dataset type does not correspond to the choices listed above.

    Returns:
        Tuple[DataLoader, DataLoader]: _description_
    """
    # Use validation set normalized using all data
    if dataset_type == 'all':
        train = DataLoader(dataset(split='_train'), shuffle=True, batch_size=batch_size, num_workers=2)
        val = DataLoader(dataset(split='_val'), shuffle=False, batch_size=batch_size, num_workers=2)
        return train, val
    
    # Next two options use validation set normalized using normal data
    val = DataLoader(dataset(split='val'), shuffle=False, batch_size=batch_size, num_workers=2)
    if dataset_type == 'normal':
        train_normal = DataLoader(dataset(split='normal_train'), shuffle=True, batch_size=batch_size, num_workers=2)
        return train_normal, val
    
    elif dataset_type == 'weighted':
        normal = dataset(split='normal_train').data 
        event = dataset(split='event_train').data  
        train = DataLoader(np.concatenate((normal, event)), shuffle=True, batch_size=batch_size, num_workers=2)
        return train, val 
    
    # Remaining options use sampled validation set
    val = DataLoader(dataset(split='val_sampled'), shuffle=False, batch_size=batch_size, num_workers=2)
    if dataset_type == 'sampled_normal':
        train_normal = DataLoader(dataset(split='normal_sampled'), shuffle=True, batch_size=batch_size, num_workers=2)
        return train_normal, val
    
    elif dataset_type == 'sampled_weighted':
        normal = dataset(split='normal_sampled').data 
        event = dataset(split='event_sampled').data  
        train = DataLoader(np.concatenate((normal, event)), shuffle=True, batch_size=batch_size, num_workers=2)
        return train, val 
    
    raise ValueError(f"expected dataset_type one of: all, normal, weighted, sampled_normal, or sampled_weighted but got {dataset_type}")

def create_module(dataset: CustomDataset, features: int, dataset_type: str, weight:float, args: parameters.Parameters, stage: int) -> FlowModule:
    """Creates flow module.

    Args:
        dataset (CustomDataset): Dataset
        features (int): Number of variables/features of data.
        dataset_type (str): The dataset type that is used.
        weight (float): Weight to use during training in case datasetype is either 'weighted' or 'sampled_weighted'.
        args (parameters.Parameters): Parameter dataclass containing hyperparmaters.
        stage (int): Stage of the flow module. If 1 the model trains with a cosine anealing lr scheduler, otherwise not.

    Returns:
        FlowModule: Pytorch lighting flow module
    """
    if dataset_type in ['weighted', 'sampled_weighted']:
        return FlowModuleWeighted(features=features, dataset=dataset, args=args, stage=stage, weight=weight)
    else:
        return FlowModule(features=features, dataset=dataset, args=args, stage=stage, weight=weight)
   


@click.command()
@click.option('--dataset', type=str, default='gas')
@click.option('--dataset_type', type=str, default='sampled_weighted') 
@click.option('--weight', type=float, default=None) 
def train(dataset:str, dataset_type: str, weight: float):    
    """Train a normalizing flow on provided dataset and dataset type.

    Args:
        dataset (str): Dataset to train on. Default: gas. Choices: power, gas, miniboone, hepmass.
        dataset_type (str): The type of flow network and dataset that is used. Default: sampled_weighted
        Choices: all (train on the entire UCI dataset), split (train on event and normal data),
        
        weight (float): The weight that is used in case of weighted training. Default: None.
        If none the weight is computed using Equation 2.4.10 of the report.
    """
    # Set seeds for reproducibility 
    utils.seed_all(2023)
    dataset_str = dataset
    
    # Construct data loaders
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    train_loader, val_loader = create_data_loaders(dataset, args.batch_size, dataset_type)
    
    # Get device
    device, version = utils.set_device()
    
    # create model
    if weight is None:
        normal_data = dataset(split='normal_train')
        event_data = dataset(split='event_train')
        weight = data.preprocess.compute_event_weight(normal=normal_data, event=event_data, xi=dataset().xi, threshold=dataset()._threshold)

    features = train_loader.dataset.data.shape[1]
    flow_module = create_module(features=features, dataset=dataset(), dataset_type=dataset_type, args=args, stage=1, weight=weight)

    
    # Initialize checkpointers
    checkpointer = create_checkpointer()

    # Fine-tune on normal data
    trainer = pl.Trainer(max_steps=args.training_steps,
                                 inference_mode=False,
                                 callbacks=[checkpointer],
                                 log_every_n_steps=args.logging_interval,
                                 accelerator=device)
    trainer.fit(flow_module, train_loader, val_loader)
    
    # Evaluate models (computes log likelihood tensor)
    test_set = 'all' if 'sampled' in dataset_type else 'normal'     
    evaluate.evaluate(dataset=dataset_str, version=version, test_set=test_set)


if __name__ == "__main__":
    train()
