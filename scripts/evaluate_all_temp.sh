#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

python src/flow/compute_llh.py --true=253784 --version=659061 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=659062 --dataset=gas
python src/flow/compute_llh.py --true=253784 --version=669556 --dataset=gas

python src/flow/compute_llh.py --true=270632 --version=659064 --dataset=power
python src/flow/compute_llh.py --true=270632 --version=659065 --dataset=power
python src/flow/compute_llh.py --true=270632 --version=669557 --dataset=power

python src/flow/compute_llh.py --true=320617 --version=659067 --dataset=miniboone
python src/flow/compute_llh.py --true=320617 --version=659068 --dataset=miniboone
python src/flow/compute_llh.py --true=320617 --version=669558 --dataset=miniboone

python src/flow/compute_llh.py --true=270635 --version=669559 --dataset=hepmass
python src/flow/compute_llh.py --true=270635 --version=669560 --dataset=hepmass
python src/flow/compute_llh.py --true=270635 --version=669561 --dataset=hepmass














