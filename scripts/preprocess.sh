#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out


project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir/venv/bin/activate"
./$project_dir/scripts/set_dataroot
python "$project_dir/src/data/preprocess.py"