#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

python src/experiments/flow/compute_llh.py --true=253784 --version=448275 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448277 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448278 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448279 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448280 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448281 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448283 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448284 --dataset=gas
python src/experiments/flow/compute_llh.py --true=253784 --version=448283 --dataset=gas



