# #!/bin/bash -l

# #SBATCH --job-name=ddim
# #SBATCH --gres=gpu:a40:1
# #SBATCH --time=24:00:00
# #SBATCH --output=slurm.out
# #SBATCH --error=slurm.err

# conda activate sd

python train.py \
    --chkpt-dir="./chkpts/ddim" \
    --use-ddim \
    --eval \
    --exp-name="ddim" \
    --dataset="cifar10" \
    --epochs=50 \
    --batch-size=256 \
    --num-samples=64 \
    --chkpt-intv=120 \
    --immiscibility \
    --dry-run \
    --root="/home/roshasu/atml/datasets"
