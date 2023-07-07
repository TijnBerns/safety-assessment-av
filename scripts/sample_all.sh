#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

# GAS DATASET
python "$project_dir"/src/flow/sample.py --version=253784 --dataset=gas --num_samples=1
python "$project_dir"/src/flow/sample.py --version=270632 --dataset=power --num_samples=1
python "$project_dir"/src/flow/sample.py --version=320617 --dataset=miniboone --num_samples=1
python "$project_dir"/src/flow/sample.py --version=270635 --dataset=hepmass --num_samples=1