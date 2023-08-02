#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:0
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
mkdir /scratch/tberns/tmp


export DATAROOT='/scratch/tberns/data'
export TMPDIR='/scratch/tberns/tmp'
srun python "$project_dir"/src/flow/experiments/kde_vs_flow.py --seed=$1 