import sys
sys.path.append('src')

import utils
import parameters
from flow_module import FlowModule, FlowModuleWeighted, FlowModuleSampled

import click
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from data.base import CustomDataset



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

def create_module(dataset, features, dataset_type, args, stage):
    if dataset_type in ['weighted', 'sampled_weighted']:
        return FlowModuleWeighted(features=features, dataset=dataset, args=args, stage=stage)
    else:
        return FlowModule(features=features, dataset=dataset, args=args, stage=stage)
    

@click.command()
@click.option('--pretrain', type=bool, default=False)
@click.option('--dataset', type=str, default='gas')
@click.option('--dataset_type', type=str, default='split') #choices=['weighted','all','split', 'zero_weight']
def train(pretrain: bool, dataset:str, dataset_type: str):    
    # Construct data loaders
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    normal_train, event_train, val = create_data_loaders(dataset, args.batch_size, dataset_type)
    event_data = event_train.dataset.data
    xi = dataset().xi
    threshold = dataset().threshold
    event_from_normal = normal_train.dataset.data[normal_train.dataset.data[:, xi] > threshold]
    event_train = DataLoader(np.concatenate((event_data, event_from_normal)), shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Get device
    device, _ = utils.set_device()
    
    # create model
    features = normal_train.dataset.data.shape[1]
    n_event = np.sum(normal_train.dataset.data[:,normal_train.dataset.xi] > normal_train.dataset.threshold)
    flow_module = FlowModuleSampled(features=features, dataset=dataset(), args=args, stage=1, n_event=n_event, event_loader=event_train)

    # Initialize checkpointers
    checkpointer = create_checkpointer()

    if pretrain:
        # Pre-train on event data
        trainer_stage_1 = pl.Trainer(max_steps=args.training_steps_stage_1,
                                     inference_mode=False,
                                     enable_checkpointing=False,
                                     log_every_n_steps=args.logging_interval,
                                     accelerator=device)
        trainer_stage_1.fit(flow_module, event_train, val)
        # flow_module.freeze_partially()
    else:
        args.training_steps_stage_2 = args.training_steps_stage_1 + args.training_steps_stage_2
        args.learning_rate_stage_2 = args.learning_rate_stage_1
        flow_module.max_steps_stage_two = args.training_steps_stage_2
        flow_module.lr_stage_two = args.learning_rate_stage_2

    # Fine-tune on normal data
    trainer_stage_2 = pl.Trainer(max_steps=args.training_steps_stage_2,
                                 inference_mode=False,
                                 callbacks=[checkpointer],
                                 log_every_n_steps=args.logging_interval,
                                 accelerator=device)
    flow_module.set_stage(2)
    trainer_stage_2.fit(flow_module, normal_train, val)


if __name__ == "__main__":
    train()
