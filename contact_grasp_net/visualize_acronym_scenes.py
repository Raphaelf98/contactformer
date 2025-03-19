import os
import numpy as np
from visualization_utils_o3d import visualize_pointcloud, visualize_pointcloud_with_camera
def collect_pc_cam_fields(directory):
    """
    Collects the 'pc_cam' field from all .npz files in the given directory.

    Args:
        directory (str): Path to the directory containing .npz files.

    Returns:
        list: A list of numpy arrays containing 'pc_cam' data.
    """
    pc_cam_list = []
    camera_pose_list = []
    for file in os.listdir(directory):
        if file.endswith(".npz"):
            file_path = os.path.join(directory, file)
            data = np.load(file_path)
            if 'pc_cam' in data:
                pc_cam_list.append(data['pc_cam'][:, :3])
            if 'camera_pose' in data:
                camera_pose_list.append(data['camera_pose'])
            else:
                print(f"Warning: 'pc_cam' not found in {file}")

    return pc_cam_list, camera_pose_list

scene0 ='/home/ssdArray/datasets/grasp_planning_datasets/acronym/acronym/renders/000000'
pc, pose = collect_pc_cam_fields(scene0)
visualize_pointcloud_with_camera(pc[4], pose[4])