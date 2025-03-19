import os
import sys
import argparse
import torch
import numpy as np
import pyrender
import trimesh
# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
ACRONYM_DIR = BASE_DIR+"/acronym/"
CONFIG_DIR = CONTACT_DIR+"/config.yaml"
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

global_config = config_parser.load_config(CONFIG_DIR)
print(global_config)
train_dataset = acronym_dataset.AcryonymDataset(global_config, train=True, device=device, use_saved_renders=False, debug=True)

print('pc_cam shape:',train_dataset[0]['pc_cam'].shape)
print('camera_pose shape:',train_dataset[0]['camera_pose'].shape)
print('pos_contact_points shape:',train_dataset[0]['pos_contact_points'].shape)
print('pos_contact_dirs shape:',train_dataset[0]['pos_contact_dirs'].shape)
print('pos_finger_diffs shape:',train_dataset[0]['pos_finger_diffs'].shape)

def visualize_pointcloud_with_camera(pointcloud, camera_pose):
    # Convert the torch point cloud tensor to numpy
    if isinstance(pointcloud, torch.Tensor):
        pointcloud_np = pointcloud.cpu().numpy()
    else:
        pointcloud_np = np.array(pointcloud)
    
    # Create a scene
    scene = pyrender.Scene()

    # Create a Trimesh point cloud object
    pc_mesh = trimesh.points.PointCloud(pointcloud_np[:, :3])

    # Convert Trimesh point cloud to Pyrender Mesh
    pc_pyrender_mesh = pyrender.Mesh.from_points(pc_mesh.vertices)

    # Add the point cloud to the scene
    scene.add(pc_pyrender_mesh)

    # Add a camera with the specified camera pose
    _fx=616.36529541
    _fy=616.20294189 
    _cx=310.25881958 
    _cy=236.59980774 
    _znear=0.04
    _zfar=20 
    _height=480 
    _width=640
    camera = pyrender.IntrinsicsCamera(_fx, _fy, _cx, _cy, _znear, _zfar)    
    _camera_node = scene.add(camera, pose=np.eye(4), name='camera')

    # Set up a light for better visualization
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Render the scene
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

# Sample data to test the function
pointcloud = torch.rand(20000, 4)  # Replace with your actual point cloud data
camera_pose = torch.eye(4)  # Replace with your actual camera pose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud_matplotlib(pointcloud):
    # Convert the torch point cloud tensor to numpy if necessary
    if isinstance(pointcloud, torch.Tensor):
        pointcloud_np = pointcloud.cpu().numpy()
    else:
        pointcloud_np = np.array(pointcloud)
    
    # Use only the XYZ coordinates for the point cloud
    x, y, z = pointcloud_np[:, 0], pointcloud_np[:, 1], pointcloud_np[:, 2]

    # Plotting the point cloud
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', s=1)  # 's' is the size of the points

    # Setting the labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud Visualization")

    # Set equal scaling for all axes to keep the aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
# Visualize
import random
print(train_dataset[0]['pc_cam'], train_dataset[0]['camera_pose'])
for i in range(10):
    #visualize_pointcloud_matplotlib(train_dataset[random.randint(0,2000)]['pc_cam'])
    visualize_pointcloud_matplotlib(train_dataset[i]['pc_cam'])
#visualize_pointcloud_with_camera(train_dataset[0]['pc_cam'], train_dataset[0]['camera_pose'])
