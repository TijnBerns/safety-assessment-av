""" 
Module containing population based training pipeline for normalizing flows.
"""

import sys

sys.path.append("src")
sys.path.append("src/data")
import os

import utils
import parameters
from flow_module import FlowModule, FlowModuleTrainableWeight
import compute_llh

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


def get_true_version(dataset_str):
    if dataset_str == "gas":
        return "253784"
    if dataset_str == "power":
        return "270632"
    if dataset_str == "hepmass":
        return "270635"
    if dataset_str == "miniboone":
        return "320617"
    return ValueError


@click.command()
@click.option("--dataset", type=str)
@click.option("--dataset_type", type=str)
@click.option("--num_cpus", type=int)
@click.option("--num_gpus", type=int)
@click.option("--num_samples", type=int, default=4)
@click.option(
    "--storage_path", type=str, default=f"/scratch/{os.environ['USER']}/ray_results"
)
def train_pb(
    dataset: str,
    dataset_type: str,
    num_cpus: int,
    num_gpus: int,
    storage_path: str,
    num_samples: int,
):
    # Set environment variables
    tmp_dir = os.environ["TMPDIR"]
    os.environ["RAY_TMPDIR"] = tmp_dir

    # Initialize ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=tmp_dir)

    # Set seeds for reproducibility
    utils.seed_all(2023)
    dataset_str = dataset

    # Construct data loaders
    args = parameters.get_parameters(dataset)
    dataset = parameters.get_dataset(dataset)
    train_loader, val = train.create_data_loaders(
        dataset, args.batch_size, dataset_type
    )
    features = train_loader.dataset.data.shape[1]

    print(
        f"\n\n!!!!!!!!!!! DATASET SIZE {train_loader.dataset.data.shape}!!!!!!!!!!!!!!!!\n\n"
    )
    # Get device
    device, version = utils.set_device()

    # Create datamodule
    class DataModule(pl.LightningDataModule):
        def __init__(self) -> None:
            super().__init__()

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val

        def test_dataloader(self):
            return None

    dm = DataModule()

    config = {
        "features": features,
        "dataset": dataset(split="normal_sampled"),
        "args": args,
        "stage": 1,
        "weight": tune.uniform(0, 1),
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
            accelerator=device,
        )
        .fit_params(datamodule=dm)
        .checkpointing(
            monitor="val_log_density",
            save_top_k=2,
            save_last=False,
            mode="max",
            every_n_epochs=1,
            filename=pattern,
            auto_insert_metric_name=False,
        )
        .build()
    )

    # Pobulation Based scheduler
    mutations_config = (
        LightningConfigBuilder().module(config={"weight": tune.uniform(0, 1)}).build()
    )

    scheduler = PopulationBasedTraining(
        perturbation_interval=2,
        mode="max",
        metric="val_log_density",
        time_attr="training_iteration",
        hyperparam_mutations={"lightning_config": mutations_config},
    )

    # Define a base LightningTrainer without hyper-parameters for Tuner
    num_workers = max(1, int(num_samples / num_cpus))
    lightning_trainer = LightningTrainer(
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=device != "cpu",
            resources_per_worker={
                "CPU": num_workers,
                "GPU": 0 if device == "cpu" else 1,
            },
        )
    )

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_log_density",
                checkpoint_score_order="max",
            ),
            storage_path=storage_path,
            name=dataset_str,
        ),
    )
    results = tuner.fit()

    # Load best checkpoint
    best_result = results.get_best_result(metric="val_log_density", mode="max")
    evaluator = evaluate.Evaluator(dataset=dataset_str, version=version, test_set="all")
    best_checkpoint = evaluate.get_ray_checkpoint(Path(best_result.path))
    best_checkpoint = evaluate.get_best_checkpoint(best_checkpoint)

    # Compute LLH tensor
    _, llh_tensor_path = evaluator.compute_llh_tensor(best_checkpoint)
    llh_tensor_path = Path(llh_tensor_path)

    # Compute LLH and MSE
    true_version = get_true_version(dataset_str)
    true_path, _ = evaluate.get_pl_checkpoint(true_version)
    true_path = evaluate.get_best_checkpoint(true_path)
    compute_llh.compute_metrics(evaluator, true_path, [llh_tensor_path], version)

    # # Copy checkpoint and llh tensor to lightning_logs
    # dest = Path('lightning_logs', *best_checkpoint.parts[-3:]).parent
    # dest.mkdir(parents='True')
    # best_checkpoint.rename(dest / best_checkpoint.name)
    # llh_tensor_path.rename(dest / llh_tensor_path.name)


if __name__ == "__main__":
    train_pb()
