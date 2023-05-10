from argparse import ArgumentParser
from config import MVParameters as parameters
import data.data_utils as data_utils
from torch.utils.data import DataLoader
from model import FeedForward
from utils import set_device

import scipy
import scipy.stats
import scipy.integrate
from scipy.optimize import fsolve
import numpy as np
from nn_approach.model import FeedForward
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from config import UVParameters as uv_params
from config import MVParameters as mv_params
import torch
from torch.utils.data import DataLoader
import data.data_utils as data_utils
from itertools import product
from utils import save_csv, save_json
import pandas as pd
from typing import Tuple
from pathlib import Path
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
parser = ArgumentParser()


parser.add_argument('--num_normal', default=int(10e3))
parser.add_argument('--num_event', default=int(10e3))
parser.add_argument('--p_event', default=0.08)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--training_steps_stage_1', default=int(10e3))
parser.add_argument('--training_steps_stage_2', default=int(10e3))
parser.add_argument('--learning_rate_stage_1', default=1e-4)
parser.add_argument('--learning_rate_stage_2', default=1e-4)
parser.add_argument('--logging_interval', default=50)
args = parser.parse_args()

def main():
    # Generate/load data
    distributions, _, distribution = parameters.get_distributions()
    threshold = data_utils.determine_threshold(args.p_event, distributions[-1])
    normal_data, event_data = data_utils.generate_data(distribution, args.num_normal, args.num_event, threshold)
    
    # Construct data loaders
    normal_loader = DataLoader(normal_data, shuffle=True, batch_size=args.batch_size)
    event_loader = DataLoader(normal_data, shuffle=True, batch_size=args.batch_size)
    
    # Define model and device
    device, jobid = set_device()
    model = FeedForward()
    
    # Initialize checkpointer
    pattern = ''
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        every_n_train_steps=100,
        monitor="val_mse",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Pre-train on event data
    trainer_stage_1 = pl.Trainer(max_steps=args.training_steps_stage_1,
                            inference_mode=False,
                            callbacks=[checkpointer],
                            #  logger=False,
                            log_every_n_steps=args.logging_interval,
                            accelerator=device)
    trainer_stage_1.fit(model, event_loader)
    
    
    # Fine-tune on normal data
    trainer_stage_2 = pl.Trainer(max_steps=args.training_steps_stage_2,
                        inference_mode=False,
                        callbacks=[checkpointer],
                        # logger=False,
                        log_every_n_steps=args.logging_interval,
                        accelerator=device)
    trainer_stage_2.fit(model, normal_loader)

if __name__ == "__main__":
    main()