#!/usr/bin/env bash
#SBATCH --partition=cncz
#SBATCH --time=48:00:00


project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/main.py