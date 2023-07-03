#!/bin/bash

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'

# sbatch "$project_dir"/scripts/flow_normal_sampled.sh gas
# sbatch "$project_dir"/scripts/flow_zero_weight_sampled.sh gas
# sbatch "$project_dir"/scripts/flow_weighted_sampled.sh gas

# sbatch "$project_dir"/scripts/flow_normal_sampled.sh power
# sbatch "$project_dir"/scripts/flow_zero_weight_sampled.sh power
sbatch "$project_dir"/scripts/flow_weighted_sampled.sh power

# sbatch "$project_dir"/scripts/flow_normal_sampled.sh miniboone
# sbatch "$project_dir"/scripts/flow_zero_weight_sampled.sh miniboone
sbatch "$project_dir"/scripts/flow_weighted_sampled.sh miniboone

# sbatch "$project_dir"/scripts/flow_normal_sampled.sh hepmass
# sbatch "$project_dir"/scripts/flow_zero_weight_sampled.sh hepmass
sbatch "$project_dir"/scripts/flow_weighted_sampled.sh hepmass

