#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

python src/flow/experiments/plot_weigth_vs_mse.py \
    --true=253784 \
    --versions=454675 --weights=0.1 \
    --versions=454676 --weights=0.2 \
    --versions=454677 --weights=0.3 \
    --versions=454678 --weights=0.4 \
    --versions=454679 --weights=0.5 \
    --versions=454680 --weights=0.6 \
    --versions=454681 --weights=0.7 \
    --versions=454682 --weights=0.8 \
    --versions=454683 --weights=0.9 \
    --versions=514633 --weights=1.0 \
    --dataset=gas



