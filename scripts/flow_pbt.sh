#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
##SBATCH --nodelist=cn47
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:0
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.out

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
mkdir /scratch/tberns/tmp


export DATAROOT='/scratch/tberns/data'
export TMPDIR='/scratch/tberns/tmp'

srun python "$project_dir"/src/flow/train_pb.py --dataset=$1 --dataset_type=sampled_weighted --num_cpus=8 --num_gpus=0 --storage_path=/scratch/tberns/ray_results

# python src/flow/train_pb.py --dataset=miniboone --dataset_type=sampled_weighted --num_cpus=1 --num_gpus=0 --storage_path=/scratch/tberns/ray_results

