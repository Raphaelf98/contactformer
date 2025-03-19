import os
import sys
import argparse
import torch
import numpy as np
import pyrender
import trimesh
from torch.utils.data.dataloader import DataLoader

# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
ACRONYM_DIR = BASE_DIR+"/acronym/"
CONFIG_DIR = CONTACT_DIR+"/config_original.yaml"
sys.path.append(os.path.join(BASE_DIR))
from contact_grasp_net import acronym_dataset
import config_parser

# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # To get pyrender to work headless

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("CUDA is available! 🎉")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Please check your installation.")
print(CONFIG_DIR)
global_config = config_parser.load_config(CONFIG_DIR)
print(global_config)
train_dataset = acronym_dataset.AcronymDataset(global_config, train=True, device=device, use_saved_renders=False, debug=False)
dataloader= DataLoader(train_dataset,1,shuffle=False)
successfull_scenes = []
failed_scenes = []
for i in range(len(train_dataset)):
    try:
        data = train_dataset[i]
        successfull_scenes.append(i)
    except:
        print(f'scene #{i} not loading')
        failed_scenes.append(i)
print(f'successfull_scenes #{len(successfull_scenes)}')
print(f'failed_scenes #{len(train_dataset)-len(successfull_scenes)}')
