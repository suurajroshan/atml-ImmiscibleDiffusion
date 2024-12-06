# #!/bin/bash -l

# #SBATCH --job-name=ddim
# #SBATCH --partition=gpu
# #SBATCH --time=24:00:00
# #SBATCH --output=ddim.out
# #SBATCH --error=ddim.err

# conda activate sd

python3.9 train.py \
    --chkpt-dir="./chkpts/ddim" \
    --use-ddim \
    --eval \
    --exp-name="ddim" \
    --dataset="cifar10" \
    --batch-size=256 \
<<<<<<< Updated upstream
    --num-samples=128 \
    --root="/home/roshasu/atml/datasets" \
    --eval-total-size=4096 \
    --eval-batch-size=128 \
    --image-intv=5 \
    --chkpt-intv=3900 \
    --immiscibility 
=======
    --num-samples=64 \
    --chkpt-intv=120 \
    --immiscibility \
    --dry-run 
>>>>>>> Stashed changes
