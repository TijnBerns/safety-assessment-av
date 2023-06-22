#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'
python "$project_dir"/src/experiments/flow/evaluate.py --dataset=hepmass --version=$1