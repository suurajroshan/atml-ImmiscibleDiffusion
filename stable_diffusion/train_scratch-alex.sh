#!/bin/bash -l

#SBATCH --job-name=immdiff
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=immdiff-true-%j.out
#SBATCH --error=immdiff-true-%j.err

conda activate sd

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
export WORLD_SIZE=1  # Assuming 1 GPU
export RANK=0       # Replace with the rank of the current process

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="cifar10"

# accelerate config
accelerate launch --main_process_port $MASTER_PORT --mixed_precision="fp16" conditional_scratch_train_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=32 --center_crop --random_flip \
  --train_batch_size=256 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=100000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="stablediff-immdiff" \
  --enable_xformers_memory_efficient_attention \
  --caption_column="label" \
  --dataloader_num_workers=1 \
  --seed=42 \
  --checkpointing_steps=10000 \
  --image_column=img \
  --calc-fid-epochs=128 \
  --fid-prompt-length=16 \
  --num-imgs-save=128 \
  --immdiff
