import glob
import os
import argparse
import sys
# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))

import torch
import numpy as np
from contact_grasp_net.contact_graspnet_predictor import GraspPredictor
from contact_grasp_net import config_parser
from contact_grasp_net.visualization_utils_o3d import visualize_grasps, show_image
from contact_grasp_net.checkpoint_io import CheckpointIO
from contact_grasp_net.data import load_available_input_data

import importlib.util

def inference(model, global_config, 
              ckpt_dir,
              input_paths, 
              local_regions=True, 
              filter_grasps=True, 
              skip_border_objects=False,
              z_range = [0.2,0.6],
              forward_passes=1,
              K=None,):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    # Build the model
    grasp_estimator = GraspPredictor(global_config, ContactGraspNet=model)

    # Load the weights
    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    try:
        load_dict = checkpoint_io.load('model_best.pt')
    except FileExistsError:
        print(f'No model checkpoint found under {model_checkpoint_dir}')
        load_dict = {}

    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    for p in glob.glob(input_paths):
        print('Loading ', p)

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, 
                                                                                    z_range=z_range)
        
        print(pc_full.shape)

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(pc_full, 
                                                                                       pc_segments=pc_segments, 
                                                                                       local_regions=local_regions, 
                                                                                       filter_grasps=filter_grasps, 
                                                                                       forward_passes=forward_passes)  
    
        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                  pc_full=pc_full, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, pc_colors=pc_colors)

        # Visualize results          
        # show_image(rgb, segmap)
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, help='Log dir')
    parser.add_argument('--model', type=str, default='ptv2', help='ptv2, ptv3')

    # parser.add_argument('--np_path', default='/test_data/8.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--np_path', default='/acronym_scenes/005251/005.npz', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    FLAGS.np_path = BASE_DIR + FLAGS.np_path
    if torch.cuda.is_available():
       print("CUDA is available! 🎉")
       print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
       print(f"CUDA Device Count: {torch.cuda.device_count()}")
       print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Please check your installation.")

    global_config = config_parser.load_config_inference(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    model_file_path = os.path.join(FLAGS.ckpt_dir, 'conatact_graspnet_model.py')
    if os.path.exists(model_file_path):
        spec = importlib.util.spec_from_file_location("checkpoiont_model", model_file_path)
        conatact_graspnet_model = importlib.util.module_from_spec(spec)
        sys.modules["checkpoiont_model"] = conatact_graspnet_model
        spec.loader.exec_module(conatact_graspnet_model)
        
        if FLAGS.model == 'ptv2':
            ContactGraspNet = conatact_graspnet_model.ContactGraspNetPtV2
        elif FLAGS.model == 'ptv3':
            ContactGraspNet = conatact_graspnet_model.ContactGraspNetPtV3
    else:
        raise ValueError(f'No model file found under {model_file_path}')
    
    inference(ContactGraspNet, global_config, 
              FLAGS.ckpt_dir,
              FLAGS.np_path, 
              local_regions=FLAGS.local_regions,
              filter_grasps=FLAGS.filter_grasps,
              skip_border_objects=FLAGS.skip_border_objects,
              z_range=eval(str(FLAGS.z_range)),
              forward_passes=FLAGS.forward_passes,
              K=eval(str(FLAGS.K)))