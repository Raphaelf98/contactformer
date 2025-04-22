import torch
import random
import os
import glob
import copy
import numpy as np
import cv2
import provider
import torch.nn.functional as F
from torch.utils.data import Dataset
from contact_grasp_net import scene_renderer
from contact_grasp_net import utils
import trimesh.transformations as tra
from pathlib import Path
class AcronymDataset(Dataset):
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_scences {int} -- training scenes (default: {None})
        num_test_scences {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """
    def __init__(self, global_config, debug=False, train=True, device=None, use_saved_renders=False):
        self.train = train
        self.device = device
        self.global_config = global_config

        self.debug = debug

        self._acronym_root_dir = global_config['DATA']['data_path']
        
        self._raw_num_points = global_config['DATA']['raw_num_points']
        self._scene_contacts_path = global_config['DATA']['scene_contacts_path']
        
        
        _, self.contact_infos = self._load_scene_contacts(self._acronym_root_dir, self._scene_contacts_path, debug=self.debug)
        num_test_scenes = global_config['DATA']['num_test_scenes']
        num_train_scenes = len(self.contact_infos)- num_test_scenes # determine the amount of test scenes from total scenes

        if debug:
            num_train_scenes = 8
            num_test_scenes = 2

        self.num_test_scenes = num_test_scenes
        self.num_train_scenes = num_train_scenes
        
        self.intrinsics=global_config['DATA']['intrinsics']
        self.elevation=global_config['DATA']['view_sphere']['elevation']

        self._estimate_normals = global_config['DATA']['input_normals']
        self._return_segmap = False
        self._distance_range = global_config['DATA']['view_sphere']['distance_range']
        self._pc_augm_config = global_config['DATA']['pc_augm']
        self._depth_augm_config = global_config['DATA']['depth_augm']
        self._use_farthest_point = global_config['DATA']['use_farthest_point']
        self._renderer = scene_renderer.SceneRenderer(caching=True, intrinsics=self.intrinsics, viewing_mode=False) 

        self._cam_orientations = []
        self._elevation = np.array(self.elevation)/180.
        for az in np.linspace(0, np.pi * 2, 30):
            for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                self._cam_orientations.append(tra.euler_matrix(0, -el, az))
        self._coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

        self._num_renderable_cam_poses = len(self._cam_orientations)
        self._num_saved_cam_poses = 200  # TODO: Make this a config

    def __getitem__(self, index):
        """
        Loads a scene and (potentially) renders a point cloud.

        Data is indexed as follows:
            For each scene, we have a set of cpos_contact_dirshe scene index is
            index // num_cam_poses, the camera pose index is index % num_cam_poses

        Arguments:
            index {int} -- scene_pose_index.

        Returns:
            dict -- data dictionary"""
        
        if not self.train:
            index = self.num_train_scenes + index
        

        #cam_pose_index = random.randit(0, 200) # create random view
        if self._estimate_normals:
            raise NotImplementedError

        if self._return_segmap:
            raise NotImplementedError

        estimate_normals = False
        # -- Build and Render Scene -- #
        
        scene_contact_points = self.contact_infos[index]['scene_contact_points']
        obj_paths_long = self.contact_infos[index]['obj_paths']
        obj_transforms = self.contact_infos[index]['obj_transforms']
        obj_scales = self.contact_infos[index]['obj_scales']
        grasp_transforms = self.contact_infos[index]['grasp_transforms']
        scene_id = self.contact_infos[index]['scene_contact_id']
        if self.debug:
            obj_paths = []
            for path in obj_paths_long:
                parts = path.split(os.path.sep)
                obj_paths.append(os.path.join(self._acronym_root_dir, os.path.join(parts[0], *parts[2:])))
        else: 
            obj_paths = [os.path.join(self._acronym_root_dir, p) for p in obj_paths_long]

        try: 
            self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        except: 
            print('Error loading scene %s' % scene_id)
            return self.__getitem__(self._generate_new_index())
        pc_cam, pc_normals, camera_pose, depth = self._render_scene(estimate_normals=estimate_normals)

        # Convert from OpenGL to OpenCV Coordinates
        camera_pose, pc_cam = self._center_pc_convert_cam(camera_pose, pc_cam)
        pc_cam = torch.from_numpy(pc_cam).type(torch.float32)
        pc_normals = torch.tensor(pc_normals).type(torch.float32) # Rn this is [] #not utilized
        camera_pose = torch.from_numpy(camera_pose).type(torch.float32)
        depth = torch.from_numpy(depth).type(torch.float32) #not utilized
        try:
            pos_contact_points, pos_contact_dirs, pos_finger_diffs, \
                pos_approach_dirs = self._process_contacts(scene_contact_points, grasp_transforms)
        except ValueError:
            print('No positive contacts found')
            print('Scene id: %s' % scene_id)
            return self.__getitem__(self._generate_new_index()) #NOTE: idx are chosen randomly not in succession
        
        data = dict(
            pc_cam=pc_cam, # scene point cloud
            camera_pose=camera_pose, # camera pose 
            pos_contact_points=pos_contact_points, # positive of contact points (paper variable: c)
            pos_contact_dirs=pos_contact_dirs, # positive contact directions (paper variable: b)
            pos_finger_diffs=pos_finger_diffs, # positive finger differences (paper variable: w)
            pos_approach_dirs=pos_approach_dirs # positve approach directions (paper variable: a) 
            )
        print(f'loaded scene #{scene_id}')
        return data
    def _process_contacts(self, scene_contact_points, grasp_transforms):
        """
        Processes contact information for a given scene

        num_contacts may not be the same_contact_grasp_cfg
        Arguments:
            scene_contact_points {np.ndarray} -- (num_contacts, 2, 3) array of contact points
            grasp_transforms {np.ndarray} -- (num_contacts, 4, 4) array of grasp transforms

        Returns:
            Scene data
        """
        num_pos_contacts = self.global_config['DATA']['labels']['num_pos_contacts'] # get number of positive contacts to be retrieved from scene
        contact_directions_01 = scene_contact_points[:,0,:] - scene_contact_points[:,1,:] # get vector from contact point 0 to contact point 1 for all contact point pairs
        all_contact_points = scene_contact_points.reshape(-1, 3) # flat array where each row is a 3d point

        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps) # euclidean distance between contact point pairs (variable name: w)
        
        all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_directions[1::2] =  contact_directions_01 / all_finger_diffs[:,np.newaxis] # store both positve and negative normalized (by finger difference w) contact directions

        all_contact_suc = np.ones_like(all_contact_points[:,0]) # create array of ones of length of all contact points
        all_grasp_transform = grasp_transforms.reshape(-1,4,4) # Reshapes grasp_transforms to align with the contact points
        all_approach_directions = all_grasp_transform[:,:3,2] # extract z axis from each 4x4 grasp transformation matrix, represents approach direction for each grasp

        pos_idcs = np.where(all_contact_suc>0)[0]
        if len(pos_idcs) == 0:
            raise ValueError('No positive contacts found')
        
        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
        all_pos_contact_dirs = all_contact_directions[pos_idcs]
        all_pos_approach_dirs = all_approach_directions[pos_idcs//2]

         # -- Sample Positive Contacts -- #
        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points)/2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(
                np.arange(len(all_pos_contact_points)),
                num_pos_contacts*2 - len(all_pos_contact_points),
                replace=True)
            pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs,
                                                 pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(
                np.arange(len(all_pos_contact_points)),
                num_pos_contacts*2,
                replace=False)
        pos_contact_points = torch.from_numpy(all_pos_contact_points[pos_sampled_contact_idcs,:]).type(torch.float32)

        pos_contact_dirs = torch.from_numpy(all_pos_contact_dirs[pos_sampled_contact_idcs,:]).type(torch.float32)
        pos_contact_dirs = F.normalize(pos_contact_dirs, p=2, dim=1)

        pos_finger_diffs = torch.from_numpy(all_pos_finger_diffs[pos_sampled_contact_idcs]).type(torch.float32)

        pos_approach_dirs = torch.from_numpy(all_pos_approach_dirs[pos_sampled_contact_idcs]).type(torch.float32)
        pos_approach_dirs = F.normalize(pos_approach_dirs, p=2, dim=1)

        return pos_contact_points, pos_contact_dirs, pos_finger_diffs, pos_approach_dirs
    def _generate_new_index(self):
        """
        Randomly generates a new index for the dataset.

        Used if the current index is invalid (e.g. no positive contacts or failed to load)
        """
        if self.train:
            return torch.randint(self.num_train_scenes, (1,))[0]
        else:
            return torch.randint(self.num_test_scenes, (1,))[0]

    def _render_scene(self,estimate_normals=False, camera_pose=None, augment=False):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(in_camera_pose, render_pc=False)
        depth = self._augment_depth(depth)

        pc = self._renderer._to_pointcloud(depth)
        pc = utils.regularize_pc_point_count(pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        if augment:
            pc = self._augment_pc(pc)

        pc_normals = utils.estimate_normals_cam_from_pc(pc[:,:3], raw_num_points=self._raw_num_points) if estimate_normals else []

        return pc, pc_normals, camera_pose, depth
    def get_cam_pose(self,cam_orientation):
        """
        Samples camera pose on shell around table center

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """
        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2,3] += self._renderer._table_dims[2]
        cam_pose[:3,:2]= -cam_pose[:3,:2]
        return cam_pose
    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """

        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self._apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'],
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :],
                                            sigma=self._pc_augm_config['sigma'],
                                            clip=self._pc_augm_config['clip'])[0]


        return pc[:,:3]
    def _apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = utils.farthest_points(pc, occlusion_nclusters, utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            mask = depth != 0
            noise = np.zeros_like(depth)
            noise[mask] = np.clip(sigma * np.random.randn(*depth.shape)[mask], -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]

        return depth
    def _center_pc_convert_cam(self, cam_pose, point_clouds):
        """
        Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud

        :param cam_poses: (bx4x4) Camera poses in OpenGL format
        :param batch_data: (bxNx3) point clouds
        :returns: (cam_poses, batch_data) c_augment_pconverted
        """

        # OpenCV OpenGL conversion
        cam_pose[:3, 1] = -cam_pose[:3, 1]
        cam_pose[:3, 2] = -cam_pose[:3, 2]
        cam_pose = utils.inverse_transform(cam_pose)

        pc_mean = np.mean(point_clouds, axis=0, keepdims=True)
        point_clouds[:,:3] -= pc_mean[:,:3]
        cam_pose[:3,3] -= pc_mean[0,:3]

        return cam_pose, point_clouds
    
    def __len__(self):
        """
        Returns the number of rendered scenes in the dataset.

        Returns:
        int -- self._num_train_scenes * len(self._cam_orientations)
        """
        if self.train:
            return self.num_train_scenes
        else:
            return self.num_test_scenes
        

    def _load_scene_contacts(self,acronym_dir, scene_contacts_path, debug=False):
        """
        Load contact grasp annotations from acronym scenes 

        Arguments:
            dataset_folder {str} -- folder with acronym data and scene contacts

        Keyword Arguments:
            TODO test_split_only {bool} -- whether to only return test split scenes (default: {False})
            TODO  num_test {int} -- how many test scenes to use (default: {None})
            scene_contacts_path {str} -- name of folder with scene contact grasp annotations (default: {'scene_contacts_new'})

        Returns:
            list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
        """
       
        scene_contact_paths = sorted(glob.glob(os.path.join(acronym_dir, scene_contacts_path, '*')))
        
        valid_contact_infos =[]
        if not debug:
            for contact_path in scene_contact_paths:

                contact_id = contact_path.split('/')[-1].split('.')[0]

                try:
                    npz = np.load(contact_path, allow_pickle=False)

                    contact_info = {'scene_contact_id': contact_id,
                                'scene_contact_points':npz['scene_contact_points'],
                                'obj_paths':npz['obj_paths'],
                                'obj_transforms':npz['obj_transforms'],
                                'obj_scales':npz['obj_scales'],
                                'grasp_transforms':npz['grasp_transforms']}
                    contact_info['obj_paths'] = [Path(p).parent.parent / Path(p).name for p in contact_info['obj_paths']]
                    valid_contact_infos.append(contact_info)
                   
                except:

                    print(f'corrupt scene: {contact_path}')
        else: 
            print("****DEBUG****")
            print(" Loading first 10 scenes ...")
            for contact_path in scene_contact_paths[:10]:

                contact_id = contact_path.split('/')[-1].split('.')[0]

                try:
                    npz = np.load(contact_path, allow_pickle=False)

                    contact_info = {'scene_contact_id': contact_id,
                                'scene_contact_points':npz['scene_contact_points'],
                                'obj_paths':npz['obj_paths'],
                                'obj_transforms':npz['obj_transforms'],
                                'obj_scales':npz['obj_scales'],
                                'grasp_transforms':npz['grasp_transforms']}
                    valid_contact_infos.append(contact_info)
                except:

                    print(f'corrupt scene: {contact_path}')
        
        return scene_contact_paths, valid_contact_infos
    
    def __del__(self):
        print('********** terminating renderer **************')


