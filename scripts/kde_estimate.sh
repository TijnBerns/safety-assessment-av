#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu-prio,csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out


project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/kde_approach/main.py