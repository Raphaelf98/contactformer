# ContactFormer

ContactFormer is a novel 6-DoF robotic grasp prediction network proposed in the Master's thesis *"Comparative Analysis of Deep Learning-Based 6DoF Grasp Pose Estimation Methods"* (Raphael Ullrich, TU Berlin, 2025). It replaces the PointNet++ backbone of [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) with a grouped vector attention (GVA) encoder-decoder derived from [Point Transformer V2](https://arxiv.org/abs/2203.02301), enabling more expressive feature learning for scene-level grasp planning.

> **Note:** This repository contains a PyTorch re-implementation of Contact-GraspNet (the original TensorFlow implementation is [here](https://github.com/NVlabs/contact_graspnet)) as well as the ContactFormer extension. Results have been evaluated empirically and may not match the original paper. This code is provided as-is.

---

## Overview

Given a raw point cloud of a cluttered tabletop scene captured from a single viewpoint, ContactFormer predicts collision-free, scene-level 6-DoF parallel-jaw gripper poses. Each predicted grasp pose *g = (R, t)* is fully described by:

- a contact point *c ∈ P*
- an approach vector *â*
- a grasp direction *b̂*
- a predicted grasp width *ŵ*

### Architecture

ContactFormer modifies the ContactGraspNet pipeline in the following way:

1. **MsgSA layer** — A multi-scale grouping set-abstraction (MsgSA) layer from PointNet++ downsamples the 20 000-point input to 2 048 points while aggregating multi-scale local features (radii: 0.02 m, 0.04 m, 0.08 m; 320 output channels). This bridges the input size to the transformer backbone.
2. **Symmetric Point Transformer V2 encoder-decoder** — A three-stage encoder-decoder with skip connections replaces the original PointNet++ backbone. The encoder uses 2, 6, and 2 transformer blocks across its stages with feature dimensions of 384 → 384 → 512, starting from 320 input channels. Each stage uses grouped vector attention (GVA) with grid pooling (grid sizes: 0.04 m, 0.08 m, 0.16 m, tuned for tabletop-scale scenes). The decoder reconstructs 2 048 points with 256 features using channel sizes 384 → 256 → 256.
3. **ContactGraspNet heads** — The original four 1D-CNN prediction heads (approach direction, grasp direction, grasp confidence, grasp width) are kept unchanged.

### Support Surface Loss (SSLoss)

A persistent challenge in scene-level grasp prediction is that networks produce high-confidence grasps on the edges of the point cloud — typically the tabletop surface. ContactFormer introduces a novel **support surface loss** to address this at training time.

The loss uses RANSAC to estimate the supporting plane in each scene point cloud and penalizes high-confidence contact predictions that fall close to or below that plane:

```
l(δ) = log(1 + α · exp(−β·δ))   for −0.02 ≤ δ ≤ 1.0
```

where *δ* is the signed distance from the contact point to the estimated plane, *α = 1*, and *β = 100*. The total support surface loss is weighted by the predicted confidence score and averaged over the top-512 predictions per batch. This conditions the network to assign lower confidence near the supporting surface without requiring post-processing filtering.

### Training

All ContactFormer variants are trained on the [Acronym dataset](https://github.com/NVlabs/acronym) (17.7 M simulated parallel-jaw grasps on 8 872 ShapeNetSem objects). The Acronym pipeline renders synthetic cluttered tabletop scenes with per-point grasp labels that are compatible with the ContactGraspNet training code.

The following model variants are provided:

| Model | Backbone | Support Surface Loss |
|---|---|---|
| ContactGraspNet (CGN) | PointNet++ | No |
| CGN + SSLoss | PointNet++ | Yes |
| **ContactFormer (CF)** | **Point Transformer V2** | **No** |
| **CF + SSLoss** | **Point Transformer V2** | **Yes** |

### Results

Experiments were conducted in physical simulation (PhysX via RAI) and on a real Franka Emika Panda robot across grasp-and-shake, table-clear, and pick-and-drop tasks.

| Model | Combined Success Rate |
|---|---|
| O-CGN (original TF checkpoint) | 68.5% |
| ContactFormer (CF) | 58% |
| ContactGraspNet (CGN) | 47% |

ContactFormer outperforms the PyTorch-trained ContactGraspNet baseline. The support surface loss was shown to meaningfully reduce boundary grasps during training (object grasp ratio improvement of ~262% for CF, from Or = 0.08 to Or = 0.29).

---

## Installation

### ContactFormer (Point Transformer V2)

```bash
conda create -n contact_former python=3.10
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_geometric
```

### ContactGraspNet (PyTorch baseline)

```bash
conda env create -f contact_graspnet_env.yml
conda activate pytorch_cuda_env
pip3 install -e .
```

### Hardware

- **Training:** 1× NVIDIA GPU with ≥ 24 GB VRAM (tested on RTX 3090). Reduce batch size if needed.
- **Inference:** 1× NVIDIA GPU with ≥ 8 GB VRAM.

---

## Inference

Model weights are included in the `checkpoints` directory. Test data can be found in `test_data/`.

ContactFormer can directly predict a 6-DoF grasp distribution from a raw scene point cloud. For object-wise grasps, removing background grasps, and denser proposals, it is recommended to use an object segmentation front-end. This repository was tested with [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) for unknown object segmentation (infrastructure not included).

**From a depth map (`.npy`/`.npz`) with camera matrix K and optional segmentation map:**

```bash
python contact_grasp_net/inference.py \
       --np_path="test_data/*.npy" \
       --local_regions --filter_grasps
```

**From a raw 3D point cloud:**

```bash
python contact_graspnet/inference.py --np_path=/path/to/your/pc.npy \
                                     --forward_passes=5 \
                                     --z_range=[0.2,1.1]
```

**Key inference flags:**

| Flag | Description |
|---|---|
| `--np_path` | Input `.npz`/`.npy` file(s). Keys: `depth`, `K`, optionally `segmap`, `rgb`; or `xyz` / `xyz_color` for a raw Nx3 point cloud. |
| `--ckpt_dir` | Path to checkpoint directory (default: `checkpoint/scene_test_2048_bs3_hor_sigma_001`). |
| `--local_regions` | Crop 3D local regions around object segments (requires `segmap`). |
| `--filter_grasps` | Filter contact points to object segment surfaces (requires `segmap`). |
| `--skip_border_objects` | Ignore segments touching the depth map boundary. |
| `--forward_passes` | Number of batched forward passes; increase to sample more grasp contacts. |
| `--z_range` | `[min, max]` depth range in metres to crop the input point cloud. |

---

## Training

### Set Up Acronym Dataset

Follow the instructions in [docs/acronym_setup.md](docs/acronym_setup.md).

### Set Environment Variables

On a headless server, set:

```bash
export PYOPENGL_PLATFORM='egl'
```

This is also handled automatically in the training script.

### Quickstart

```bash
python3 contact_graspnet_pytorch/train.py --data_path acronym/
```

### Additional Options

```bash
# Custom model name and data path
python contact_graspnet/train.py --ckpt_dir checkpoints/your_model_name \
                                 --data_path /path/to/acronym/data

# Resume a previous run
python contact_graspnet/train.py --ckpt_dir checkpoints/previous_model_name \
                                 --data_path /path/to/acronym/data
```

### Generate Scenes Yourself (optional)

See [docs/generate_scenes.md](docs/generate_scenes.md).

---

## Repository Structure

```
contactformer/
├── contact_grasp_net/          # ContactFormer and ContactGraspNet model code
├── Pointcept/                  # Point Transformer V2 (PTv3) components
├── Pointnet_Pointnet2_pytorch/ # PointNet++ components (MsgSA layer)
├── gripper_control_points/     # Franka Panda gripper geometry
├── gripper_models/             # Gripper mesh models
├── scripts/                    # Utility scripts
├── test_data/                  # Sample point clouds for inference
├── train.py                    # Training entry point
└── contact_former_env.yaml     # Conda environment
```

---

## Citation

If you use this work, please consider citing the original Contact-GraspNet paper and starring this repository:

```bibtex
@article{sundermeyer2021contact,
  title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

The Point Transformer V2 backbone:

```bibtex
@inproceedings{wu2022point,
  title={Point Transformer V2: Grouped Vector Attention and Partition-based Pooling},
  author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
  booktitle={NeurIPS},
  year={2022}
}
```
