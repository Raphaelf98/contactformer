#!/bin/bash

# Set the CUDA deviceexport 
CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=6
# Define the checkpoint directory
CKPT_DIR="checkpoints/ptv2-revised-robo-eval-loss-1-100-200"
OVERWRITE_CKPT_DIR=1
MODEL="ptv2"
which python
# Run the training script
taskset -c 0,1,2,3,4,5,6,7,8,9 python /home/raphael/thesis/contact_former/train_robo_eval.py \
    --ckpt_dir "$CKPT_DIR" \
    --overwrite_ckpt_dir "$OVERWRITE_CKPT_DIR" \
    --model "$MODEL" \
    --config_file "transformer_config_surface_loss.yaml" \

