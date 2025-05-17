#!/bin/bash

# Set the CUDA deviceexport 
CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=0
# Define the checkpoint directory
CKPT_DIR="checkpoints/ptv2-final-grid"
OVERWRITE_CKPT_DIR=1
MODEL="ptv2"
which python
# Run the training script
taskset -c 60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89 python /home/raphael/thesis/contact_former/train_robo_eval.py \
    --ckpt_dir "$CKPT_DIR" \
    --overwrite_ckpt_dir "$OVERWRITE_CKPT_DIR" \
    --model "$MODEL"\
    --config_file "ptv2_final_grid.yaml" \
    --resume ptv2-final-grid_20250517181129
