#!/bin/bash

# Set fixed parameters
ckpt_dir="checkpoints/ptv2-revised-robo-eval"
config_file="transformer_config.yaml"
load_model="model_best.pt"
dump_dir="/home/raphael/thesis/contact_former/analysis/results/ptv2-revised-robo-eval"

# List of .npy input files
np_path="/home/raphael/thesis/contact_former/acronym_scenes/005251/*.npz"

python contact_grasp_net/inference.py \
        --ckpt_dir "$ckpt_dir" \
        --config_file "$config_file" \
        --load_model "$load_model" \
        --dump_dir "$dump_dir" \
        --np_path "$np_path"
