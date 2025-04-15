import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contact_grasp_net.mesh_utils import create_gripper
from contact_grasp_net.utils import *

class ContacGraspNetLoss(nn.Module):
    def __init__(self, global_config, device):
        super(ContacGraspNetLoss,self).__init__()
        self.global_config = global_config
        self.device = device 

        # --Process config--
        config_losses = [
            'pred_contact_base',
            'pred_contact_success', # True
            'pred_contact_offset',  # True
            'pred_contact_approach',
            'pred_grasps_adds',  # True
            'pred_grasps_adds_gt2pred',
            'pred_grasps_centrality'

        ]
        config_weights = [
            'dir_cosine_loss_weight',
            'score_ce_loss_weight',  # True
            'offset_loss_weight',  # True
            'approach_cosine_loss_weight',
            'adds_loss_weight',  # True
            'adds_gt2pred_loss_weight',
            'centrality_loss_weight' # True
        ]
        for config_loss, config_weight in zip(config_losses, config_weights):
            if global_config['MODEL'][config_loss]:
                setattr(self, config_weight, global_config['OPTIMIZER'][config_weight])
            else:
                setattr(self, config_weight, 0.0)
        bin_weights = global_config['DATA']['labels']['bin_weights']
        self.bin_weights = torch.tensor(bin_weights).to(self.device)
        self.bin_vals = self._get_bin_vals().to(self.device)
        # -- create gripper---
        self.gripper = create_gripper('panda')

        n_copies = 1  # We will repeat this according to the batch size
        gripper_control_points = self.gripper.get_control_point_tensor(n_copies) # b x 5 x 3
        sym_gripper_control_points = self.gripper.get_control_point_tensor(n_copies, symmetric=True)
        self.gripper_control_points_homog = torch.cat([gripper_control_points,
            torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
        self.sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points,
            torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

        self.gripper_control_points_homog = self.gripper_control_points_homog.to(self.device)
        self.sym_gripper_control_points_homog = self.sym_gripper_control_points_homog.to(self.device)

    def forward(self, prediction, target):
        """
        Computes loss terms from pointclouds, network predictions and labels.
        Note: computation happens batch-wise

        Arguments:
            pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
            end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
            dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
            offset_labels_pc {tf.variable} -- grasp width labels (bxNx1)
            grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1)
            approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
            global_config {dict} -- config dict

        Returns:
            [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss,
            adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
        """
        # -- Interpolate Labels -- #
        pos_contact_points = target['pos_contact_points']    # B x M x 3
        pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
        pos_finger_diffs = target['pos_finger_diffs']        # B x M
        pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
        camera_pose = target['camera_pose']                  # B x 4 x 4

        pred_grasps_cam = prediction['pred_grasps_cam']                            # B x N x 4 x 4
        pred_points = prediction['pred_points']                                    # B x N x 3
        pred_scores = prediction['pred_scores']                                    # B x N x 1
        grasp_offset_head = prediction['grasp_offset_head'].permute(0, 2, 1)       # B x N x 10

        dir_labels_pc_cam, \
        grasp_offset_labels_pc, \
        grasp_success_labels_pc, \
        approach_labels_pc_cam, \
        debug = self._compute_labels(pred_points, 
                                     camera_pose,
                                     pos_contact_points,
                                     pos_contact_dirs,
                                     pos_finger_diffs,
                                     pos_approach_dirs)
        # I think this is the number of positive grasps that are in view
        min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
        pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B
        # pos_grasps_in_view = torch.maximum(grasp_success_labels_pc.sum(dim=1), min_geom_loss_divisor)  # B

        total_loss = 0.0

        if self.dir_cosine_loss_weight > 0:
            raise NotImplementedError

        # -- Grasp Confidence Loss -- #
        if self.score_ce_loss_weight > 0:  # TODO (bin_ce_loss)
            bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_labels_pc, reduction='none')  # F.binary_cross_entropy_with_logits(pred_scores, grasp_success_labels_pc) # # B x N x 1
            if 'topk_confidence' in self.global_config['LOSS'] \
                and self.global_config['LOSS']['topk_confidence']:
                bin_ce_loss, _ = torch.topk(bin_ce_loss.squeeze(), k=self.global_config['LOSS']['topk_confidence'])
            bin_ce_loss = torch.mean(bin_ce_loss)

            total_loss += self.score_ce_loss_weight * bin_ce_loss

        # -- Grasp Offset / Thickness Loss -- #
        if self.offset_loss_weight > 0:  # TODO  (offset_loss)
            if self.global_config['MODEL']['bin_offsets']:
                # Convert labels to multihot
                bin_vals = self.global_config['DATA']['labels']['offset_bins']
                grasp_offset_labels_multihot = self._bin_label_to_multihot(grasp_offset_labels_pc, 
                                                                           bin_vals)

                if self.global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
                    raise NotImplementedError

                else:
                    offset_loss = F.binary_cross_entropy_with_logits(grasp_offset_head,
                                                                     grasp_offset_labels_multihot, reduction='none')  # B x N x 1
                    if 'too_small_offset_pred_bin_factor' in self.global_config['LOSS'] \
                        and self.global_config['LOSS']['too_small_offset_pred_bin_factor']:
                        raise NotImplementedError

                    # Weight loss for each bin
                    shaped_bin_weights = self.bin_weights[None, None, :]
                    offset_loss = (shaped_bin_weights * offset_loss).mean(axis=2)
            else:
                raise NotImplementedError
            masked_offset_loss = offset_loss * grasp_success_labels_pc.squeeze()
            # Divide each batch by the number of successful grasps in the batch
            offset_loss = torch.mean(torch.sum(masked_offset_loss, axis=1, keepdim=True) / pos_grasps_in_view)

            total_loss += self.offset_loss_weight * offset_loss

        if self.approach_cosine_loss_weight > 0:
            raise NotImplementedError

        # -- 6 Dof Pose Loss -- #
        if self.adds_loss_weight > 0:  # TODO  (adds_loss)
            # Build groudn truth grasps and compare distances to predicted grasps

            ### ADS Gripper PC Loss
            # Get 6 DoF pose of predicted grasp
            if self.global_config['MODEL']['bin_offsets']:
                thickness_gt = self.bin_vals[torch.argmax(grasp_offset_labels_pc, dim=2)]
            else:
                thickness_gt = grasp_offset_labels_pc[:, :, 0]

            # TODO: Move this to dataloader? 
            pred_grasps = pred_grasps_cam  # B x N x 4 x 4
            gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, 
                                                            dir_labels_pc_cam, 
                                                            pred_points, 
                                                            thickness_gt, 
                                                            use_torch=True,
                                                            device=self.device) # b x N x 4 x 4
            # Select positive grasps I think?
            success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
            success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
            pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4

            # Expand gripper control points to match number of points
            # only use per point pred grasps but not per point gt grasps
            control_points = self.gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
            control_points = control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

            sym_control_points = self.sym_gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
            sym_control_points = sym_control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

            pred_control_points = torch.matmul(control_points, pred_grasps.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

            # Transform control points to ground truth locations
            gt_control_points = torch.matmul(control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3
            sym_gt_control_points = torch.matmul(sym_control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

            # Compute distances between predicted and ground truth control points
            expanded_pred_control_points = pred_control_points.unsqueeze(2)         # B x N x 1 x 5 x 3
            expanded_gt_control_points = gt_control_points.unsqueeze(1)             # B x 1 x N' x 5 x 3  I think N' == N
            expanded_sym_gt_control_points = sym_gt_control_points.unsqueeze(1)     # B x 1 x N' x 5 x 3  I think N' == N

            # Sum of squared distances between all points
            # expanded_pred_control_points = expanded_pred_control_points.half()
            # expanded_gt_control_points = expanded_gt_control_points.half()
            squared_add = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2, dim=(3, 4))  # B x N x N'
            sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3, 4))  # B x N x N'

            # Combine distances between gt and symmetric gt grasps
            squared_adds = torch.concat([squared_add, sym_squared_add], dim=2)  # B x N x 2N'

            # Take min distance to gt grasp for each predicted grasp
            squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False)[0]  # B x N

            # Mask negative grasps
            # TODO: If there are bugs, its prob here.  The original code sums on axis=1
            # Which just determines if there is a successful grasp in the batch.  
            # I think we just want to select the positive grasps so the sum is redundant.
            sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
            binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1)
            min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k)  # B x N x 1
            adds_loss = torch.sum(pred_scores * min_adds, dim=(1), keepdim=True)  # B x 1
            adds_loss = adds_loss.squeeze() / pos_grasps_in_view.squeeze()  # B x 1
            adds_loss = torch.mean(adds_loss)
            total_loss += self.adds_loss_weight * adds_loss

        if self.adds_gt2pred_loss_weight > 0:
            raise NotImplementedError
        
        # if self.centrality_loss_weight > 0:
        #     alpha = 1
        #     xy_coords = self._project_to_image_plane(pred_points)  # B x N x 2
        #     xy_center = xy_coords.mean(dim=1, keepdim=True)  # B x 1 x 2
        #     distances = torch.norm(xy_coords - xy_center, dim=2)  # B x N
        #     max_distance = distances.max(dim=1, keepdim=True)[0]  # B x 1
        #     norm_distances = distances / (max_distance + 1e-6)  # B x N
        #     distances = torch.norm(xy_coords - xy_center, dim=2)  # B x N
        #     max_distance = distances.max(dim=1, keepdim=True)[0]  # B x 1
        #     norm_distances = distances / (max_distance + 1e-6)  # B x N
        #     penalty = torch.exp(norm_distances * alpha) - 1
        #     grasp_scores = pred_scores.squeeze(-1)
        #     edge_penalty_loss = ((1 - grasp_scores) * penalty).mean()
        #     total_loss += self.centrality_loss_weight * edge_penalty_loss
        loss_info = {
            'bin_ce_loss': bin_ce_loss,  # Grasp success loss
            'offset_loss': offset_loss,  # Grasp width loss
            'adds_loss': adds_loss,  # Pose loss
            # 'centrality_loss': edge_penalty_loss,  # Centrality loss
        }

        return total_loss, loss_info
    def _project_to_image_plane(self, points_3d):
        """
        points_3d: B x N x 3
        intrinsics: 3 x 3
        returns: B x N x 2 (xy coordinates in the image plane)
        """
        pc_mean = points_3d.mean(dim=1, keepdim=True)  # B x 1 x 3
        points_3d = points_3d + pc_mean  
        B, N, _ = points_3d.shape
        print("===> points_3d stats:")
        print(f"    x: min {points_3d[..., 0].min().item():.4f}, max {points_3d[..., 0].max().item():.4f}")
        print(f"    y: min {points_3d[..., 1].min().item():.4f}, max {points_3d[..., 1].max().item():.4f}")
        print(f"    z: min {points_3d[..., 2].min().item():.6f}, max {points_3d[..., 2].max().item():.4f}")

        K = torch.tensor([
            [616.3653, 0.0,     310.2588],
            [0.0,      616.2029, 236.5998],
            [0.0,      0.0,      1.0]
        ], dtype=torch.float32, device=points_3d.device).unsqueeze(0).repeat(B, 1, 1)
        ones = torch.ones((B, N, 1), device=points_3d.device)
        homo_points = torch.cat([points_3d, ones], dim=2)  # B x N x 4
        # Remove depth to avoid division by 0
        points_2d_homo = torch.bmm(points_3d, K.transpose(1, 2))  # (B, N, 3)
        xy_coords = points_2d_homo[:, :, :2] / points_2d_homo[:, :, 2:3].clamp(min=1e-6)  # B x N x 2
        print("===> xy_coords stats:")
        print(f"    x: min {xy_coords[..., 0].min().item():.4f}, max {xy_coords[..., 0].max().item():.4f}")
        print(f"    y: min {xy_coords[..., 1].min().item():.4f}, max {xy_coords[..., 1].max().item():.4f}")
        return xy_coords
    
    def _bin_label_to_multihot(self, cont_labels, bin_boundaries):
        """
        Computes binned grasp width labels from continuous labels and bin boundaries

        Arguments:
            cont_labels {torch.Tensor} -- continuous labels
            bin_boundaries {list} -- bin boundary values

        Returns:
            torch.Tensor -- one/multi hot bin labels
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(torch.float32)

        return multi_hot_labels
    
    def _compute_labels(self,processed_pc_cams: torch.Tensor, 
                        camera_poses: torch.Tensor, 
                        pos_contact_points: torch.Tensor,
                        pos_contact_dirs: torch.Tensor,
                        pos_finger_diffs: torch.Tensor, 
                        pos_approach_dirs: torch.Tensor):
        """
        Project grasp labels defined on meshes onto rendered point cloud 
        from a camera pose via nearest neighbor contacts within a maximum radius. 
        All points without nearby successful grasp contacts are considered 
        negative contact points.

        Here N is the number of points returned by the PointNet Encoder (2048) while
        M is the number of points in the ground truth data.  B is the batch size.
        We are trying to assign a label to each of the PointNet points by 
        sampling the nearest ground truth points.

        Arguments:
            pc_cam_pl (torch.Tensor): (B, N, 3) point cloud in camera frame
            camera_pose_pl (torch.Tensor): (B, 4, 4) homogenous camera pose
            pos_contact_points (torch.Tensor): (B, M, 3) contact points in world frame (3 DoF points)
            pos_contact_dirs (torch.Tensor): (B, M, 3) contact directions (origin centered vectors?)
            pos_finger_diffs (torch.Tensor): (B, M, ) finger diffs in world frame  (scalar distances)
            pos_approach_dirs (torch.Tensor): (B, M, 3) approach directions in world frame (origin centered vectors?)
        """
        label_config = self.global_config['DATA']['labels']

        nsample = label_config['k']  # Currently set to 1
        radius = label_config['max_radius']
        filter_z = label_config['filter_z']
        z_val = label_config['z_val']

        _, N, _ = processed_pc_cams.shape
        B, M, _ = pos_contact_points.shape

        # -- Make sure pcd is B x N x 3 -- #
        if processed_pc_cams.shape[2] != 3:
            xyz_cam = processed_pc_cams[:,:,:3]  # N x 3
        else:
            xyz_cam = processed_pc_cams

        # -- Transform Ground Truth to Camera Frame -- #
        # Transform contact points to camera frame  (This is a homogenous transform)
        # We use matmul to accommodate batch
        # pos_contact_points_cam = pos_contact_points @ (camera_poses[:3,:3].T) + camera_poses[:3,3][None,:]
        pos_contact_points_cam = torch.matmul(pos_contact_points, camera_poses[:, :3, :3].transpose(1, 2)) \
            + camera_poses[:,:3,3][:, None,:]

        # Transform contact directions to camera frame (Don't translate because its a direction vector)
        # pos_contact_dirs_cam = pos_contact_dirs @ camera_poses[:3,:3].T
        pos_contact_dirs_cam = torch.matmul(pos_contact_dirs, camera_poses[:, :3,:3].transpose(1, 2))
        
        # Make finger diffs B x M x 1
        pos_finger_diffs = pos_finger_diffs[:, :, None]

        # Transform approach directions to camera frame (Don't translate because its a direction vector)
        # pos_approach_dirs_cam = pos_approach_dirs @ camera_poses[:3,:3].T
        pos_approach_dirs_cam = torch.matmul(pos_approach_dirs, camera_poses[:, :3,:3].transpose(1, 2))

        # -- Filter Direction -- #
        # TODO: Figure out what is going on here
        if filter_z:
            # Filter out directions that are too far
            dir_filter_passed = (pos_contact_dirs_cam[:, :, 2:3] > z_val).repeat(1, 1, 3)
            pos_contact_points_cam = torch.where(dir_filter_passed, 
                                                 pos_contact_points_cam, 
                                                 torch.ones_like(pos_contact_points_cam) * 10000)
        
        # -- Compute Distances -- #
        # We want to compute the distance between each point in the point cloud and each contact point
        # We can do this by expanding the dimensions of the tensors and then summing the squared differences
        xyz_cam_expanded = torch.unsqueeze(xyz_cam, 2)  # B x N x 1 x 3
        pos_contact_points_cam_expanded = torch.unsqueeze(pos_contact_points_cam, 1)  # B x 1 x M x 3
        squared_dists_all = torch.sum((xyz_cam_expanded - pos_contact_points_cam_expanded)**2, dim=3)  # B x N x M

        # B x N x k, B x N x k
        squared_dists_k, close_contact_pt_idcs = torch.topk(squared_dists_all, 
            k=nsample, dim=2, largest=False, sorted=False)

        # -- Group labels -- #
        grouped_contact_dirs_cam = index_points(pos_contact_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3
        grouped_finger_diffs = index_points(pos_finger_diffs, close_contact_pt_idcs)  # B x N x k x 1
        grouped_approach_dirs_cam = index_points(pos_approach_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3

        # grouped_contact_dirs_cam = pos_contact_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3
        # grouped_finger_diffs = pos_finger_diffs[close_contact_pt_idcs]  # B x N x k x 1
        # grouped_approach_dirs_cam = pos_approach_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3

        # -- Compute Labels -- #
        # Take mean over k nearest neighbors and normalize
        dir_label = grouped_contact_dirs_cam.mean(dim=2)  # B x N x 3
        dir_label = F.normalize(dir_label, p=2, dim=2)  # B x N x 3

        diff_label = grouped_finger_diffs.mean(dim=2)# B x N x 1

        approach_label = grouped_approach_dirs_cam.mean(dim=2)  # B x N x 3
        approach_label = F.normalize(approach_label, p=2, dim=2)  # B x N x 3

        grasp_success_label = torch.mean(squared_dists_k, dim=2, keepdim=True) < radius**2  # B x N x 1 
        grasp_success_label = grasp_success_label.type(torch.float32)  

        # debug = dict(
        #     xyz_cam = xyz_cam,
        #     pos_contact_points_cam = pos_contact_points_cam,
        # )
        debug = {}


        return dir_label, diff_label, grasp_success_label, approach_label, debug

    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            tf.constant -- bin value tensor
        """
        bins_bounds = np.array(self.global_config['DATA']['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width']-self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        return bin_vals
    
    def _grasp_success_loss():
        """
        TODO 
        -implement ce loss
        """
        pass
    def _grasp_width_loss():
        """utils
utils
utils
        TODO 
        -implement offset loss
        """
        pass
    def _grasp_pose_loss():
        """
        TODO 
        -implement adds loss, five 3d points are used representing the 6DoF grasp pose v e R^(5x3) 
            vi_gt = v*Ri^(T) + ti,   vi_pred = v*Ri_est^(T) + ti_est
            l_adds = (1/n+) SUM(i)(n+) si*min(u) ||vi_gt - vi_pred||_2
        """
        pass
    #Additional Loss functions suggested in original implementation
    def _grasp_approach_loss():
        """
        TODO 
        -implement approach cosine loss
        """
        pass
    def _grasp_direction_loss():
        """
        TODO 
        -implement direction cosine loss
        """
        pass
