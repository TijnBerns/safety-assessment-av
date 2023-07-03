#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

python src/flow/compute_llh.py --true=253784 --version=454675 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454676 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454677 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454678 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454679 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454680 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454681 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454682 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=454683 --dataset=gas



