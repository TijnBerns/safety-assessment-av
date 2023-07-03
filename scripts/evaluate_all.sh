#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

# python src/flow/compute_llh.py --true=253784 --version=422125 --dataset=gas
# python src/flow/compute_llh.py --true=253784 --version=422135 --dataset=gas
# python src/flow/compute_llh.py --true=253784 --version=422151 --dataset=gas

# python src/flow/compute_llh.py --true=270632 --version=422535 --dataset=power
# python src/flow/compute_llh.py --true=270632 --version=422536 --dataset=power
# python src/flow/compute_llh.py --true=270632 --version=422537 --dataset=power

# python src/flow/compute_llh.py --true=320617 --version=422538 --dataset=miniboone
# python src/flow/compute_llh.py --true=320617 --version=422539 --dataset=miniboone
# python src/flow/compute_llh.py --true=320617 --version=422540 --dataset=miniboone

# python src/flow/compute_llh.py --true=270635 --version=422541 --dataset=hepmass
# python src/flow/compute_llh.py --true=270635 --version=422542 --dataset=hepmass
# python src/flow/compute_llh.py --true=270635 --version=422543 --dataset=hepmass

# True models
python src/flow/compute_llh.py --true=253784 --version=253784 --dataset=gas
python src/flow/compute_llh.py --true=270632 --version=270632 --dataset=power
python src/flow/compute_llh.py --true=320617 --version=320617 --dataset=miniboone
python src/flow/compute_llh.py --true=270635 --version=270635 --dataset=hepmass
