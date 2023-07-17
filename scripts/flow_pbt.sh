#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --nodelist=cn47
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=47:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
export DATAROOT='/scratch/tberns/data'
srun python "$project_dir"/src/flow/train_pb.py --dataset=$1 --dataset_type=weighted