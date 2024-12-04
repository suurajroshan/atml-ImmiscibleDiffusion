#!/bin/bash -l

#SBATCH --job-name=ddim
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

conda activate sd

python3.9 train.py --chkpt-dir="./chkpts/ddim" \
    --use-ddim \
    --eval \
    --exp-name="ddim"
