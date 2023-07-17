import sys
sys.path.append('src')
sys.path.append('src/data')
import os

import utils
import parameters
from flow_module import FlowModule, FlowModuleTrainableWeight

import click
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import train
import data.preprocess
import evaluate
from pathlib import Path

# Finetune weight
import ray
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


from ray.train.lightning import LightningTrainer, LightningConfigBuilder

tmp_dir = '/ceph/csedu-scratch/other/tberns/tmp'
os.environ["RAY_TMPDIR"] = tmp_dir
ray.init(num_cpus=4, num_gpus=2, _temp_dir=tmp_dir)


@click.command()
@click.option('--dataset', type=str, default='gas')
@click.option('--dataset_type', type=str, default='sampled_weighted') 
def train_pb(dataset:str, dataset_type: str):    
    # Set seeds for reproducibility 
    utils.seed_all(2023)
    dataset_str = dataset
    
    # Construct data loaders
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    normal_train, _, val = train.create_data_loaders(dataset, args.batch_size, dataset_type)
    features = normal_train.dataset.data.shape[1]
    print(f"\n\n!!!!!!!!!!!{normal_train.dataset.data.shape}!!!!!!!!!!!!!!!!\n\n")
    
    # Get device
    device, version = utils.set_device()
    
    # Create datamodule 
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

    # Number of samples from parameter space
    num_samples = 4
    config = {
        "features": features, 
        "dataset": dataset(split="normal_sampled"),
        "args": args, 
        "stage": 1,
        "weight": tune.uniform(0,1)
    }
    
    pattern = "epoch_{epoch:04d}.step_{step:09d}.log_density_{val_log_density:.2f}.best"
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=FlowModuleTrainableWeight, config=config)
        .trainer(
            max_steps=args.training_steps,
            inference_mode=False,
            enable_progress_bar=False,
            log_every_n_steps=args.logging_interval,
            accelerator=device
            )
        .fit_params(datamodule=dm)
        .checkpointing(monitor="val_log_density", 
                       save_top_k=2, 
                       save_last=False,
                       mode="max", 
                       every_n_epochs=1, 
                       filename=pattern, 
                       auto_insert_metric_name=False)
        .build()
    )
    
    
    # Pobulation Based scheduler
    mutations_config = (
        LightningConfigBuilder()
        .module(
            config={"weight": tune.uniform(0,1)}
        )
        .build()
    )
    
    scheduler = PopulationBasedTraining(
        perturbation_interval=2,
        resample_probability=0.15,
        time_attr='training_iteration',
        hyperparam_mutations={"lightning_config": mutations_config},
    )

    # Define a base LightningTrainer without hyper-parameters for Tuner
    lightning_trainer = LightningTrainer(
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=device != 'cpu', resources_per_worker={"CPU": 1, "GPU": 0 if device == 'cpu' else 1}
        )
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
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_log_density",
                checkpoint_score_order="max",
            ),
            storage_path="/home/tberns/safety-assessment-av/ray_results",
            name="gas",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_log_density", mode="max")


    evaluator = evaluate.Evaluator(dataset=dataset_str, version=version, test_set='all')
    best_checkpoint = evaluate.get_ray_checkpoint(Path(best_result.path))
    evaluator.compute_llh_tensor(best_checkpoint)

if __name__ == "__main__":
    train_pb()
