
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import logging
# Import pointnet library
CKPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(CKPT_DIR))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointcept'))
sys.path.append(os.path.join(CKPT_DIR, 'Pointcept'))
# from pointcept.models.structure import Point
# from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3 import PointTransformerV3
from pointcept.models.builder import MODELS  # Import the registry

# Create an instance of the model using the registry
PointTransformerV2 = MODELS.get("PT-v2m2")
from pointcept.models.utils import batch2offset
from pointcept.models.utils.structure import Point
import logging
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))

from Pointnet_Pointnet2_pytorch.models import pointnet2_utils



class ContactGraspNetPtV2(nn.Module):

    def __init__(self, global_config, device, verbose=False):
        super(ContactGraspNetPtV2, self).__init__()

        self.global_config = global_config
        self.model_config = global_config['MODEL']
        self.data_config = global_config['DATA']

        

        # Read parameters from model config
        set_abstraction = self.model_config.get('set_abstraction')
        in_channels = self.model_config.get('in_channels', 3)
        
     
        
        self.fps = self.model_config.get('farthest_point_sampling')
        in_channels = self.model_config.get('in_channels')  # PT-v2m2 requires 9 input channels
        grid_sizes = tuple(self.model_config.get('grid_sizes'))
        attn_qkv_bias = self.model_config.get('attn_qkv_bias')
        pe_multiplier = self.model_config.get('pe_multiplier')
        pe_bias = self.model_config.get('pe_bias')
        attn_drop_rate = self.model_config.get('attn_drop_rate')
        drop_path_rate = self.model_config.get('drop_path_rate')
        enable_checkpoint = self.model_config.get('enable_checkpoint')
        unpool_backend = self.model_config.get('unpool_backend')
          # Set for classification tasks (can be modified)
        patch_embed_cfg = self.model_config['PATCH_EMBEDDING']
        patch_embed_depth = patch_embed_cfg.get('patch_embed_depth')
        patch_embed_channels = patch_embed_cfg.get('patch_embed_channels')
        patch_embed_groups = patch_embed_cfg.get('patch_embed_groups')
        patch_embed_neighbours = patch_embed_cfg.get('patch_embed_neighbours')

        encoder_cfg = self.model_config['ENCODER']
        enc_depths = tuple(encoder_cfg.get('enc_depths'))
        enc_channels = tuple(encoder_cfg.get('enc_channels'))
        enc_groups = tuple(encoder_cfg.get('enc_groups'))
        enc_neighbours = tuple(encoder_cfg.get('enc_neighbours'))
        decoder_cfg = self.model_config['DECODER']
        dec_depths = tuple(decoder_cfg.get('dec_depths'))
        dec_channels = tuple(decoder_cfg.get('dec_channels'))
        dec_groups = tuple(decoder_cfg.get('dec_groups'))
        dec_neighbours = tuple(decoder_cfg.get('dec_neighbours'))

        if set_abstraction:
            self.set_abstraction = True
            npoint_0=self.model_config['SET_ABSTRACTION']['npoint']
            radius_list_0=tuple(self.model_config['SET_ABSTRACTION']['radius_list'])
            nsample_list_0=tuple(self.model_config['SET_ABSTRACTION']['num_sample_list'])
            mlp_list_0= tuple(self.model_config['SET_ABSTRACTION']['mlp_list'])
            print(f'Using set abstraction with {sum([mlp_list_0[i][-1] for i in range(len(mlp_list_0))])} feature channels ...')
            self.set_abstraction_1 = pointnet2_utils.PointNetSetAbstractionMsg(npoint=npoint_0,radius_list=radius_list_0,nsample_list=nsample_list_0,in_channel=3,mlp_list=mlp_list_0)
        elif self.fps:
            self.set_abstraction = False
            print(f'Using FPS with {in_channels} input channels ...')
        
        # Instantiate PointTransformerV2
        self.ptv2 = PointTransformerV2(
            in_channels=in_channels,
            num_classes=0,
            patch_embed_depth=patch_embed_depth,
            patch_embed_channels=patch_embed_channels,
            patch_embed_groups=patch_embed_groups,
            patch_embed_neighbours=patch_embed_neighbours,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_groups=enc_groups,
            enc_neighbours=enc_neighbours,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_groups=dec_groups,
            dec_neighbours=dec_neighbours,
            grid_sizes=grid_sizes,
            attn_qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
            unpool_backend=unpool_backend,
        )
        
        
        
     
        self.input_normals = self.data_config['input_normals']
        self.offset_bins = self.data_config['labels']['offset_bins']
        self.joint_heads = self.model_config['joint_heads']


        self.device = device

    
        # --- Network heads ----
        # Head for grasp direction
        print(f'Using {dec_channels[1]} decoder channels for grasp heads ...')
        self.grasp_dir_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), # p = 1 - keep_prob  (tf is inverse of torch)
            nn.Conv1d(128, 3, 1, padding=0)
        )

        # Remember to normalize the output of this head
        # Head for grasp approach
        self.grasp_approach_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0)
        )

        # Head for grasp width
        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, len(self.offset_bins) - 1, 1, padding=0)
        )

        # Head for contact points
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 0.5 in original code
            nn.Conv1d(128, 1, 1, padding=0)
        )
        

    
    def log_model_config(self, config=None, indent=0, prefix="MODEL"):
        """
        Recursively logs the model configuration dictionary.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary to log (default is self.model_config).
        indent : int
            Indentation level for nested entries.
        prefix : str
            Prefix to indicate where in the config the print is coming from.
        """
        if config is None:
            config = self.model_config
            print("========== Model Configuration ==========")

        for key, value in config.items():
            if isinstance(value, dict):
                print(" " * indent + f"{prefix}.{key}:")
                self.log_model_config(value, indent + 2, prefix=f"{prefix}.{key}")
            else:
                print(" " * indent + f"{prefix}.{key}: {value}")

        if indent == 0:
            print("=========================================")        
            
    
    def forward(self,point_cloud):
        
        point_cloud = point_cloud[:, :, :3] 
        batch_size, num_points, _ = point_cloud.shape

        if self.fps:
            centroid_idcs = pointnet2_utils.farthest_point_sample(point_cloud, 2048)
            point_cloud = pointnet2_utils.index_points(point_cloud, centroid_idcs)
            device = point_cloud.device
            pred_points = point_cloud
            # Reshape from (B, N, 3) -> (B*N, 3)
            coord = point_cloud.view(-1, 3).to(device).contiguous()   # Flatten all batch samples into one tensor
            pred_points = pred_points.permute(0, 2, 1) 
            # Use coordinates as features (can be replaced with real features)
            feat = coord.clone()  # Default: Use (x, y, z) as features
            num_points = point_cloud.shape[1]  # Number of points after abstraction
            # Compute offset: Cumulative sum of points per batch
            #offset = torch.arange(1, batch_size + 1, device=device) * num_points
            offset = batch2offset(torch.arange(batch_size, device=device).repeat_interleave(2048)).to(device).contiguous()   

            # feat = torch.zeros_like(coord).to(device).contiguous()  # [N, 3] assuming no additional features

        elif self.set_abstraction:
            point_cloud = torch.transpose(point_cloud, 1, 2)  # (B, C, N)

            device = point_cloud.device
            l0_xyz = point_cloud[:, :3, :]  # Extract XYZ coordinates (B, 3, N)
            l0_points = point_cloud[:, 3:6, :] if self.input_normals else l0_xyz.clone()  # (B, C, N)

            # -- PointNet Backbone -- #
            # Apply Set Abstraction
            l1_xyz, l1_points = self.set_abstraction_1(l0_xyz, l0_points)

            batch_size = l1_xyz.shape[0]
            num_points = l1_xyz.shape[2]  # Number of points after abstraction

            # Reshape coordinates and features to match transformer input
            coord = l1_xyz.permute(0, 2, 1).reshape(-1, 3).contiguous().to(device)  # Shape: (B*N, 3)
            feat = l1_points.permute(0, 2, 1).reshape(-1, l1_points.shape[1]).contiguous().to(device)  # Shape: (B*N, feature_dim)

            # Create batch indices tensor: [B*N] with repeating batch indices
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)

            # Compute offset from batch indices
            offset = batch2offset(batch_indices).to(device)           # Shape: (48B*N,)
            pred_points = l1_xyz
        else: 
            pred_points = torch.transpose(point_cloud, 1, 2)[:, :3, :]  # (B, C, N)
            device = point_cloud.device
       
            # Reshape from (B, N, 3) -> (B*N, 3)
            coord = point_cloud.view(-1, 3).to(device).contiguous()   # Flatten all batch samples into one tensor

            # Use coordinates as features (can be replaced with real features)
            #feat = coord.clone()  # Default: Use (x, y, z) as features

            # Compute offset: Cumulative sum of points per batch
            #offset = torch.arange(1, batch_size + 1, device=device) * num_points
            offset = batch2offset(torch.arange(batch_size, device=device).repeat_interleave(num_points)).to(device).contiguous()   

            feat = torch.zeros_like(coord).to(device).contiguous()  # [N, 3] assuming no additional features
        # point_cloud = torch.transpose(point_cloud, 1, 2)  # Now we have batch x channels (3 or 6) x num_points

        
        
        point_dict = {
            "coord": coord,    # Shape: (B*N, 3)
            "feat": feat,      # Shape: (B*N, 3) (or real features if available)
            "offset": offset   # Shape: (B,)
        }
 
         
        
        out = self.ptv2(point_dict)


        feat_dim = out.shape[1]  # Extract feature dimension

        # Reshape back to (batch_size, num_points, feature_dim)
        out = out.view(batch_size, -1, feat_dim)
        # out = out.view(batch_size, num_points, feat_dim)

        # Permute to get (batch_size, feature_dim, num_points)
        feat = out.permute(0, 2, 1)  # (B, F, N)
        # Store the feature tensor for later use or visualization
        # Save the feature tensor to a file
        
        # -- Heads -- #
        # Grasp Direction Head
        grasp_dir_head = self.grasp_dir_head(feat)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)  # normalize along channels

        # Grasp Approach Head
        approach_dir_head = self.grasp_approach_head(feat)

        # compute gram schmidt orthonormalization
        dot_product = torch.sum(grasp_dir_head_normed * approach_dir_head, dim=1, keepdim=True)
        projection = dot_product * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_head - projection, p=2, dim=1)

        # Grasp Width Head
        grasp_offset_head = self.grasp_offset_head(feat)

        # Binary Segmentation Head
        binary_seg_head = self.binary_seg_head(feat)

        # -- Construct Output -- #
        # Get 6 DoF grasp pose
        torch_bin_vals = self.get_bin_vals()

        # PyTorch equivalent of tf.gather_nd with conditional
        # I think the output should be B x N
        if self.model_config['bin_offsets']:
            argmax_indices = torch.argmax(grasp_offset_head, dim=1, keepdim=True)
            offset_bin_pred_vals = torch_bin_vals[argmax_indices]  # kinda sketch but works?
            # expand_dims_indices = argmax_indices.unsqueeze(1)
            # offset_bin_pred_vals = torch.gather(torch_bin_vals, 1, argmax_indices)
        else:
            offset_bin_pred_vals = grasp_offset_head[:, 0, :]

        
        pred_grasps_cam = self.build_6d_grasp(approach_dir_head_orthog.permute(0, 2, 1),
                                              grasp_dir_head_normed.permute(0, 2, 1),
                                              pred_points.permute(0, 2, 1),
                                              offset_bin_pred_vals.permute(0, 2, 1))  # B x N x 4 x 4

        # Get pred scores
        pred_scores = torch.sigmoid(binary_seg_head).permute(0, 2, 1)

        # Get pred points
        pred_points = pred_points.permute(0, 2, 1)

        # Get pred offsets
        offset_pred = offset_bin_pred_vals

        # -- Values to compute loss on -- #
        

        pred = dict(
            pred_grasps_cam=pred_grasps_cam,
            pred_scores=pred_scores,
            pred_points=pred_points,
            offset_pred=offset_pred,
            grasp_offset_head=grasp_offset_head, # For loss
            approach_dir= approach_dir_head_orthog,
            base_dir = grasp_dir_head_normed
        )
        assert pred["pred_grasps_cam"].shape == torch.Size([batch_size, num_points, 4, 4]), \
                f"Expected pred_grasps_cam to have shape ({batch_size}, {num_points}, 4, 4), but got {pred['pred_grasps_cam'].shape}"

        assert pred["pred_scores"].shape == torch.Size([batch_size, num_points, 1]), \
                f"Expected pred_scores to have shape ({batch_size}, {num_points}, 1), but got {pred['pred_scores'].shape}"

        assert pred["offset_pred"].shape == torch.Size([batch_size, 1, num_points]), \
                f"Expected offset_pred to have shape ({batch_size}, 1, {num_points}), but got {pred['offset_pred'].shape}"

        assert pred["grasp_offset_head"].shape == torch.Size([batch_size, 10, num_points]), \
            f"Expected grasp_offset_head to have shape ({batch_size}, 10, {num_points}), but got {pred['grasp_offset_head'].shape}"
        
        assert pred["approach_dir"].shape == torch.Size([batch_size, 3, num_points]), \
            f"Expected approach_dir to have shape ({batch_size}, 3 {num_points}), but got {pred['approach_dir'].shape}"
        
        assert pred["base_dir"].shape == torch.Size([batch_size, 3, num_points]), \
            f"Expected base_dir to have shape ({batch_size}, 3 {num_points}), but got {pred['base_dir'].shape}"
        # return pred_grasps_cam, pred_scores, pred_points, offset_pred, intermediates
        return pred

    def get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            torch.tensor -- bin value tensor
        """
        bins_bounds = np.array(self.data_config['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width'] \
                                  -self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32).to(self.device)
        return bin_vals



    # def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness, use_tf=False, gripper_depth = 0.1034):
    def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness,  gripper_depth = 0.1034):
        """
        Build 6-DoF grasps + width from point-wise network predictions

        Arguments:
            approach_dirs {np.ndarray/tf.tensor} -- Nx3 approach direction vectors
            base_dirs {np.ndarray/tf.tensor} -- Nx3 base direction vectors
            contact_pts {np.ndarray/tf.tensor} -- Nx3 contact points
            thickness {np.ndarray/tf.tensor} -- Nx1 grasp width

        Keyword Arguments:
            use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
            gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

        Returns:
            np.ndarray -- Nx4x4 grasp poses in camera coordinates
        """
        # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
        # To do so, we calculate the rotation and translation portions according to the paper.
        # This gives us positions as shown:
        # [ R R R T ]
        # [ R R R T ]
        # [ R R R T ]
        # [ 0 0 0 1 ]                    Note that the ^ dim is 2 and the --> dim is 3
        # We need to pad with zeros and ones to get the final shape so we generate
        # ones and zeros and stack them.
       
        grasp_R = torch.stack([base_dirs, torch.cross(approach_dirs,base_dirs),approach_dirs], dim=3)  # B x N x 3 x 3
        grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs  # B x N x 3
        grasp_t = grasp_t.unsqueeze(3)  # B x N x 3 x 1
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).to(self.device)  # B x N x 1 x 1
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).to(self.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4

        

        return grasps

class ContactGraspNetPtV3(nn.Module):

    def __init__(self, global_config, device, verbose=False):
        super(ContactGraspNetPtV3, self).__init__()

        self.global_config = global_config
        self.model_config = global_config['MODEL']
        self.data_config = global_config['DATA']

        

        # Read parameters from model config
        set_abstraction = self.model_config.get('set_abstraction')
        in_channels = self.model_config.get('in_channels', 3)   
        self.fps = self.model_config.get('farthest_point_sampling')


         
        order = tuple(self.model_config.get('order', ("z", "z-trans")))
        stride = tuple(self.model_config.get('stride', (2, 2, 2, 2)))

        mlp_ratio = self.model_config.get('mlp_ratio', 4)
        qkv_bias = self.model_config.get('qkv_bias', True)
        qk_scale = self.model_config.get('qk_scale', None)
        attn_drop = self.model_config.get('attn_drop', 0.0)
        proj_drop = self.model_config.get('proj_drop', 0.0)
        drop_path = self.model_config.get('drop_path', 0.3)
        pre_norm = self.model_config.get('pre_norm', True)
        shuffle_orders = self.model_config.get('shuffle_orders', True)
        enable_rpe = self.model_config.get('enable_rpe', False)
        enable_flash = self.model_config.get('enable_flash', True)
        upcast_attention = self.model_config.get('upcast_attention', False)
        upcast_softmax = self.model_config.get('upcast_softmax', False)

        

        encoder_cfg = self.model_config['ENCODER']
        enc_depths = tuple(encoder_cfg.get('enc_depths'))
        enc_channels = tuple(encoder_cfg.get('enc_channels'))
        enc_num_head = tuple(encoder_cfg.get('enc_num_head'))
        enc_patch_size = tuple(encoder_cfg.get('enc_patch_size'))
        decoder_cfg = self.model_config['DECODER']
        dec_depths = tuple(decoder_cfg.get('dec_depths'))
        dec_channels = tuple(decoder_cfg.get('dec_channels'))
        dec_num_head = tuple(decoder_cfg.get('dec_num_head'))
        dec_patch_size = tuple(decoder_cfg.get('dec_patch_size'))


        if set_abstraction:
            self.set_abstraction = True
            npoint_0=self.model_config['SET_ABSTRACTION']['npoint']
            radius_list_0=tuple(self.model_config['SET_ABSTRACTION']['radius_list'])
            nsample_list_0=tuple(self.model_config['SET_ABSTRACTION']['num_sample_list'])
            mlp_list_0= tuple(self.model_config['SET_ABSTRACTION']['mlp_list'])
            print(f'Using set abstraction with {sum([mlp_list_0[i][-1] for i in range(len(mlp_list_0))])} feature channels ...')
            self.set_abstraction_1 = pointnet2_utils.PointNetSetAbstractionMsg(npoint=npoint_0,radius_list=radius_list_0,nsample_list=nsample_list_0,in_channel=3,mlp_list=mlp_list_0)
        elif self.fps:
            self.set_abstraction = False
            print(f'Using FPS with {in_channels} input channels ...')
        
        # Instantiate PointTransformerV2
        self.ptv3 = PointTransformerV3(
            in_channels, order, stride, enc_depths, enc_channels, enc_num_head, enc_patch_size, 
            dec_depths, dec_channels, dec_num_head, dec_patch_size, mlp_ratio, qkv_bias, 
            qk_scale, attn_drop, proj_drop, drop_path, pre_norm, shuffle_orders, enable_rpe, 
            enable_flash, upcast_attention, upcast_softmax
        )
        
        
        
     
        self.input_normals = self.data_config['input_normals']
        self.offset_bins = self.data_config['labels']['offset_bins']
        self.joint_heads = self.model_config['joint_heads']


        self.device = device

    
        # --- Network heads ----
        # Head for grasp direction
        print(f'Using {dec_channels[1]} decoder channels for grasp heads ...')
        self.grasp_dir_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), # p = 1 - keep_prob  (tf is inverse of torch)
            nn.Conv1d(128, 3, 1, padding=0)
        )

        # Remember to normalize the output of this head
        # Head for grasp approach
        self.grasp_approach_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0)
        )

        # Head for grasp width
        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, len(self.offset_bins) - 1, 1, padding=0)
        )

        # Head for contact points
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(dec_channels[1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 0.5 in original code
            nn.Conv1d(128, 1, 1, padding=0)
        )
        

    
    def log_model_config(self, config=None, indent=0, prefix="MODEL"):
        """
        Recursively logs the model configuration dictionary.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary to log (default is self.model_config).
        indent : int
            Indentation level for nested entries.
        prefix : str
            Prefix to indicate where in the config the print is coming from.
        """
        if config is None:
            config = self.model_config
            print("========== Model Configuration ==========")

        for key, value in config.items():
            if isinstance(value, dict):
                print(" " * indent + f"{prefix}.{key}:")
                self.log_model_config(value, indent + 2, prefix=f"{prefix}.{key}")
            else:
                print(" " * indent + f"{prefix}.{key}: {value}")

        if indent == 0:
            print("=========================================")        
            
    
    def forward(self,point_cloud):
        
        # Convert from tf to torch ordering
        if self.set_abstraction:
            point_cloud = torch.transpose(point_cloud, 1, 2) # Now we have batch x channels (3 or 6) x num_points

            device = point_cloud.device
            l0_xyz = point_cloud[:, :3, :]
            l0_points = point_cloud[:, 3:6, :] if self.input_normals else l0_xyz.clone()

            # -- PointNet Backbone -- #
            # Set Abstraction Layers
            
            l1_xyz, l1_points = self.set_abstraction_1(l0_xyz, l0_points) 

            batch_size=l1_xyz.shape[0]
            num_points=l1_xyz.shape[2]


            # Step 3: Reshape coordinates and features to match transformer input
            coord = l1_xyz.permute(0, 2, 1).reshape(-1, 3).to(device)  # Shape: [N, 3]
            feat = l1_points.permute(0, 2, 1).reshape(-1, l1_points.shape[1]).to(device)  # Shape: [N, feature_dim]
            # Convert from tf to torch ordering
            #point_cloud = torch.transpose(point_cloud, 1, 2) # Now we have batch x channels (3 or 6) x num_points

            # 2. Create batch index tensor: [batch_size * num_points] with repeating indices
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)        
        # 3. Create input dictionary for the Point class
        else:
            device = point_cloud.device
            point_cloud = point_cloud[:, :, :3] 
            centroid_idcs = pointnet2_utils.farthest_point_sample(point_cloud, 2048)
            point_cloud = pointnet2_utils.index_points(point_cloud, centroid_idcs)
            point_cloud = torch.transpose(point_cloud, 1, 2)  # Now we have batch x channels (3 or 6) x num_points

            
            batch_size = point_cloud.shape[0]
            num_points = point_cloud.shape[2]  # Keep the original number of points

            # Extract coordinates and features
            coord = point_cloud[:, :3, :].permute(0, 2, 1).reshape(-1, 3).to(device)  # Shape: [N, 3]
            feat = torch.zeros_like(coord).to(device).contiguous()  # [N, 3] assuming no additional features


            # Create batch index tensor: [batch_size * num_points] with repeating indices
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)

        point_dict = {
            "coord": coord,         # Coordinates
            "feat": feat,  # Use coordinates as features (change if you have real features)
            "batch": batch_indices, # Batch indices
        }
 
        point_dict['grid_size'] = 0.005  # 0.02 meters per cell   
        # point_dict['grid_size'] = torch.tensor(0.02, device=device)     #paper indoor setting 
        # -- Transformer Backbone -- #
        #point_cloud["offset"] = batch2offset(batch_indices)
        out = self.ptv3(point_dict)


        
        # Extract from out
        coord = out["coord"]  # (N, 3)
        feat = out["feat"]    # (N, feature_dim)
        batch = out["batch"]  # (N,)

        # Get batch size and num points
        batch_size = batch.max().item() + 1  # Get batch count
        num_points = coord.shape[0] // batch_size  # Divide total points by batch size

        # Reshape into (batch_size, num_points, feature_dim) for the heads
        feat = feat.view(batch_size, num_points, -1)  # (B, N, F)
        coord = coord.view(batch_size, num_points, 3) # (B, N, 3)

        # Transpose so it matches the format expected by network heads (B, F, N)
        feat = feat.permute(0, 2, 1)  # (batch_size, feature_dim, num_points)
        coord = coord.permute(0, 2, 1) # (batch_size, 3, num_points)

        
        # l1_xyz, l1_points = self.set_abstraction_1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.set_abstraction_2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.set_abstraction_3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.set_abstraction_4(l3_xyz, l3_points)

        # # Feature Propagation Layers
        # l3_points = self.feature_propagation_3(l3_xyz, l4_xyz, l3_points, l4_points)
        # l2_points = self.feature_propagation_2(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.feature_propagation_1(l1_xyz, l2_xyz, l1_points, l2_points)

        # l0_points = l1_points
        pred_points = coord

        
        # -- Heads -- #
        # Grasp Direction Head
        grasp_dir_head = self.grasp_dir_head(feat)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)  # normalize along channels

        # Grasp Approach Head
        approach_dir_head = self.grasp_approach_head(feat)

        # compute gram schmidt orthonormalization
        dot_product = torch.sum(grasp_dir_head_normed * approach_dir_head, dim=1, keepdim=True)
        projection = dot_product * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_head - projection, p=2, dim=1)

        # Grasp Width Head
        grasp_offset_head = self.grasp_offset_head(feat)

        # Binary Segmentation Head
        binary_seg_head = self.binary_seg_head(feat)

        # -- Construct Output -- #
        # Get 6 DoF grasp pose
        torch_bin_vals = self.get_bin_vals()

        # PyTorch equivalent of tf.gather_nd with conditional
        # I think the output should be B x N
        if self.model_config['bin_offsets']:
            argmax_indices = torch.argmax(grasp_offset_head, dim=1, keepdim=True)
            offset_bin_pred_vals = torch_bin_vals[argmax_indices]  # kinda sketch but works?
            # expand_dims_indices = argmax_indices.unsqueeze(1)
            # offset_bin_pred_vals = torch.gather(torch_bin_vals, 1, argmax_indices)
        else:
            offset_bin_pred_vals = grasp_offset_head[:, 0, :]

        
        pred_grasps_cam = self.build_6d_grasp(approach_dir_head_orthog.permute(0, 2, 1),
                                              grasp_dir_head_normed.permute(0, 2, 1),
                                              pred_points.permute(0, 2, 1),
                                              offset_bin_pred_vals.permute(0, 2, 1))  # B x N x 4 x 4

        # Get pred scores
        pred_scores = torch.sigmoid(binary_seg_head).permute(0, 2, 1)

        # Get pred points
        pred_points = pred_points.permute(0, 2, 1)

        # Get pred offsets
        offset_pred = offset_bin_pred_vals

        # -- Values to compute loss on -- #
        

        pred = dict(
            pred_grasps_cam=pred_grasps_cam,
            pred_scores=pred_scores,
            pred_points=pred_points,
            offset_pred=offset_pred,
            grasp_offset_head=grasp_offset_head # For loss
        )
        assert pred["pred_grasps_cam"].shape == torch.Size([batch_size, num_points, 4, 4]), \
                f"Expected pred_grasps_cam to have shape ({batch_size}, {num_points}, 4, 4), but got {pred['pred_grasps_cam'].shape}"

        assert pred["pred_scores"].shape == torch.Size([batch_size, num_points, 1]), \
                f"Expected pred_scores to have shape ({batch_size}, {num_points}, 1), but got {pred['pred_scores'].shape}"

        assert pred["offset_pred"].shape == torch.Size([batch_size, 1, num_points]), \
                f"Expected offset_pred to have shape ({batch_size}, 1, {num_points}), but got {pred['offset_pred'].shape}"

        assert pred["grasp_offset_head"].shape == torch.Size([batch_size, 10, num_points]), \
            f"Expected grasp_offset_head to have shape ({batch_size}, 10, {num_points}), but got {pred['grasp_offset_head'].shape}"
        # return pred_grasps_cam, pred_scores, pred_points, offset_pred, intermediates
        return pred

    def get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            torch.tensor -- bin value tensor
        """
        bins_bounds = np.array(self.data_config['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width'] \
                                  -self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32).to(self.device)
        return bin_vals



    # def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness, use_tf=False, gripper_depth = 0.1034):
    def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness,  gripper_depth = 0.1034):
        """
        Build 6-DoF grasps + width from point-wise network predictions

        Arguments:
            approach_dirs {np.ndarray/tf.tensor} -- Nx3 approach direction vectors
            base_dirs {np.ndarray/tf.tensor} -- Nx3 base direction vectors
            contact_pts {np.ndarray/tf.tensor} -- Nx3 contact points
            thickness {np.ndarray/tf.tensor} -- Nx1 grasp width

        Keyword Arguments:
            use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
            gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

        Returns:
            np.ndarray -- Nx4x4 grasp poses in camera coordinates
        """
        # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
        # To do so, we calculate the rotation and translation portions according to the paper.
        # This gives us positions as shown:
        # [ R R R T ]
        # [ R R R T ]
        # [ R R R T ]
        # [ 0 0 0 1 ]                    Note that the ^ dim is 2 and the --> dim is 3
        # We need to pad with zeros and ones to get the final shape so we generate
        # ones and zeros and stack them.
       
        grasp_R = torch.stack([base_dirs, torch.cross(approach_dirs,base_dirs),approach_dirs], dim=3)  # B x N x 3 x 3
        grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs  # B x N x 3
        grasp_t = grasp_t.unsqueeze(3)  # B x N x 3 x 1
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).to(self.device)  # B x N x 1 x 1
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).to(self.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4

        

        return grasps


if __name__ == "__main__":
    import contact_grasp_net.config_parser

    global_config = contact_grasp_net.config_parser.load_config('/home/raphael/thesis/contact_former/contact_grasp_net/transformer_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContactGraspNetPtV2(global_config, device)
