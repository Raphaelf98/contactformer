#!/bin/bash

# Set the CUDA deviceexport 
CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=0
# Define the checkpoint directory
CKPT_DIR="checkpoints/ptv3-revised"
OVERWRITE_CKPT_DIR=1
MODEL="ptv3"
which python
# Run the training script
taskset -c 0,1,2,3,4,5,6,7,8,9 python /home/raphael/thesis/contact_former/contact_grasp_net/train.py \
    --ckpt_dir "$CKPT_DIR" \
    --overwrite_ckpt_dir "$OVERWRITE_CKPT_DIR" \
    --model "$MODEL"\
    --config_file "transformer_config.yaml" 
