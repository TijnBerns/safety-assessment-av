#!/bin/bash
# #SBATCH --account=cseduproject
# #SBATCH --partition=csedu
# #SBATCH --qos=csedu-normal
# #SBATCH --gres=gpu:1
# #SBATCH --mem=2G
# #SBATCH --cpus-per-task=1
# #SBATCH --time=11:00:00
# #SBATCH --output=./logs/slurm/%J.out
# #SBATCH --error=./logs/slurm/%J.out

# project_dir=.

# # Train on train-full with no speaker change symbols or ids
# source "$project_dir"/venv/bin/activate
# export DATAROOT='/scratch/tberns/data'

# # python src/flow/compute_llh.py --true=253784 --version=422125 --dataset=gas
# # python src/flow/compute_llh.py --true=253784 --version=422135 --dataset=gas
# # python src/flow/compute_llh.py --true=253784 --version=422151 --dataset=gas

# # python src/flow/compute_llh.py --true=270632 --version=422535 --dataset=power
# # python src/flow/compute_llh.py --true=270632 --version=422536 --dataset=power
# # python src/flow/compute_llh.py --true=270632 --version=422537 --dataset=power

# # python src/flow/compute_llh.py --true=320617 --version=422538 --dataset=miniboone
# # python src/flow/compute_llh.py --true=320617 --version=422539 --dataset=miniboone
# # python src/flow/compute_llh.py --true=320617 --version=422540 --dataset=miniboone

# # python src/flow/compute_llh.py --true=270635 --version=422541 --dataset=hepmass
# # python src/flow/compute_llh.py --true=270635 --version=422542 --dataset=hepmass
# # python src/flow/compute_llh.py --true=270635 --version=422543 --dataset=hepmass

# # # True models
# # python src/flow/compute_llh.py --true=253784 --version=253784 --dataset=gas
# # python src/flow/compute_llh.py --true=270632 --version=270632 --dataset=power
# # python src/flow/compute_llh.py --true=320617 --version=320617 --dataset=miniboone
# # python src/flow/compute_llh.py --true=270635 --version=270635 --dataset=hepmass

# python src/flow/evaluate.py --version=659061 --dataset=gas --test_set=all
# python src/flow/evaluate.py --version=659062 --dataset=gas --test_set=all
# python src/flow/evaluate.py --version=669556 --dataset=gas --test_set=all
# python src/flow/evaluate.py --version=659064 --dataset=power --test_set=all
# python src/flow/evaluate.py --version=659065 --dataset=power --test_set=all
# python src/flow/evaluate.py --version=669557 --dataset=power --test_set=all
# python src/flow/evaluate.py --version=659067 --dataset=miniboone --test_set=all
# python src/flow/evaluate.py --version=659068 --dataset=miniboone --test_set=all
# python src/flow/evaluate.py --version=669558 --dataset=miniboone --test_set=all
# python src/flow/evaluate.py --version=669559 --dataset=hepmass --test_set=all
# python src/flow/evaluate.py --version=669560 --dataset=hepmass --test_set=all
# python src/flow/evaluate.py --version=669551 --dataset=hepmass --test_set=all

project_dir=.

sbatch "$project_dir"/scripts/flow_eval.sh gas 659061
sbatch "$project_dir"/scripts/flow_eval.sh gas 659062
sbatch "$project_dir"/scripts/flow_eval.sh gas 669556

sbatch "$project_dir"/scripts/flow_eval.sh power 659064
sbatch "$project_dir"/scripts/flow_eval.sh power 659065
sbatch "$project_dir"/scripts/flow_eval.sh power 669557

sbatch "$project_dir"/scripts/flow_eval.sh miniboone 659067
sbatch "$project_dir"/scripts/flow_eval.sh miniboone 659068
sbatch "$project_dir"/scripts/flow_eval.sh miniboone 669558

sbatch "$project_dir"/scripts/flow_eval.sh hepmass 669559
sbatch "$project_dir"/scripts/flow_eval.sh hepmass 669560
sbatch "$project_dir"/scripts/flow_eval.sh hepmass 669561


