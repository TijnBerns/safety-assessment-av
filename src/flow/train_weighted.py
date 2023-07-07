import sys
sys.path.append('src')
sys.path.append('src/data')

import utils
import parameters
from flow_module import FlowModule, FlowModuleWeighted, FlowModuleTrainableWeight

import click
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from data.base import CustomDataset
import data.preprocess
import evaluate


# Finetune weight
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from ray.train.lightning import LightningTrainer, LightningConfigBuilder


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
    device = 'cpu'
    
    # create model
    if weight is None:
        weight = data.preprocess.compute_event_weight_np(data=normal_train.dataset, xi=dataset().xi, threshold=dataset().threshold)
        
    features = normal_train.dataset.data.shape[1]
    flow_module = create_module(features=features, dataset=dataset(), dataset_type=dataset_type, args=args, stage=1, weight=weight)

    
    # Initialize checkpointers
    checkpointer = create_checkpointer()

    
    class DataModule(pl.LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            
        def train_dataloader(self):
            return normal_train

        def val_dataloader(self):
            return val

        def test_dataloader(self):
            return None
        
    dm = DataModule()

    
    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 10
    config = {
        "features":features, 
        "dataset":dataset,
        "args": args, 
        "stage":1,
        "weight": tune.uniform(0.0,1.0)
    }
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=FlowModuleTrainableWeight, config=config)
        .trainer(
            max_steps=args.training_steps,
            inference_mode=False,
            callbacks=[checkpointer],
            log_every_n_steps=args.logging_interval,
            accelerator=device
            )
        .fit_params(datamodule=dm)
        .checkpointing(monitor="val_log_density", save_top_k=2, mode="max")
        .build()
    )

    # Make sure to also define an AIR CheckpointConfig here
    # to properly save checkpoints in AIR format.
    run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_log_density",
        checkpoint_score_order="max",
        ),
    )
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    # Define a base LightningTrainer without hyper-parameters for Tuner
    lightning_trainer = LightningTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_log_density",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_asha",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_log_density", mode="max")
    best_result



    # trainer.fit(flow_module, normal_train, val)
    
    # Evaluate models (computes log likelihood tensor)
    test_set = 'all' if 'sampled' in dataset_type else 'normal'     
    evaluate.evaluate(dataset=dataset_str, version=version, test_set=test_set)




if __name__ == "__main__":
    train()
