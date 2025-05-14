import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from torch_ransac3d.plane import plane_fit

class PlaneFitter:
    def __init__(self, device='cpu'):
        self.device = device

    def batch_plane_ransac(self, pc_batch):
        def _plane_ransac(points):
            result = plane_fit(
                pts=points,
                thresh=0.005,
                max_iterations=1000,
                iterations_per_batch=100,
                epsilon=1e-8,
                device=self.device
            )
            return result.equation

        plane_eqs = []
        for i in range(pc_batch.shape[0]):
            plane = _plane_ransac(pc_batch[i])
            plane_eqs.append(plane)

        return torch.stack(plane_eqs, dim=0).to(self.device)

        return torch.stack(plane_eqs, dim=0).to(self.device)

def visualize_contact_points_o3d(point_cloud, contact_points, inlier_mask, plane_equation):
    # Scene point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Inlier contact points (green)
    inlier_pc = o3d.geometry.PointCloud()
    inlier_pc.points = o3d.utility.Vector3dVector(contact_points[inlier_mask])
    inlier_pc.paint_uniform_color([0.0, 1.0, 0.0])

    # Outlier contact points (red)
    outlier_pc = o3d.geometry.PointCloud()
    outlier_pc.points = o3d.utility.Vector3dVector(contact_points[~inlier_mask])
    outlier_pc.paint_uniform_color([1.0, 0.0, 0.0])

    # Plane visualization
    plane_center = point_cloud.mean(axis=0)
    plane_normal = plane_equation[:3].cpu().numpy()

    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.3, depth=0.001)
    plane_mesh.translate(-plane_mesh.get_center())

    # Align plane mesh with normal
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, plane_normal)
    c = np.dot(z_axis, plane_normal)
    skew_sym_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * (1 / (1 + c + 1e-8))
    plane_mesh.rotate(R_align, center=(0, 0, 0))
    plane_mesh.translate(plane_center)
    plane_mesh.paint_uniform_color([0.8, 0.1, 0.1])

    # Visualize all
    o3d.visualization.draw_geometries([pcd, inlier_pc, outlier_pc, plane_mesh])

def plot_contact_point_score_distribution(npy_file, distance_thresh=0.01, device='cpu'):
    # Load data
    data = np.load(npy_file, allow_pickle=True).item()
    point_cloud = data['point_cloud']  # (N, 3)
    contact_pts = data['contact_pts']
    scores_dict = data['scores']

    # Merge all contact points & scores
    all_contact_points = []
    all_scores = []
    for seg_id, points in contact_pts.items():
        if len(points) > 0:
            all_contact_points.append(points)
            all_scores.append(scores_dict[seg_id])

    if len(all_contact_points) == 0:
        print("No contact points found.")
        return

    all_contact_points = np.vstack(all_contact_points)  # (M, 3)
    all_scores = np.concatenate(all_scores)             # (M,)

    # Plane fitting on scene point cloud
    pc_batch = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)
    fitter = PlaneFitter(device=device)
    plane_equation = fitter.batch_plane_ransac(pc_batch)[0]

    normal = plane_equation[:3]
    d = plane_equation[3]
    normal_norm = torch.norm(normal) + 1e-8

    contact_points_torch = torch.tensor(all_contact_points, dtype=torch.float32).to(device)
    distances = torch.abs(torch.matmul(contact_points_torch, normal) + d) / normal_norm
    distances = distances.cpu().numpy()

    # Inliers / Outliers split
    inlier_mask = distances < distance_thresh

    print(f"Total Contact Points: {len(all_scores)} | Inliers: {np.sum(inlier_mask)} | Outliers: {np.sum(~inlier_mask)}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores[inlier_mask], bins=30, alpha=0.6, label='Inliers', color='green', edgecolor='black')
    plt.hist(all_scores[~inlier_mask], bins=30, alpha=0.6, label='Outliers', color='red', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Contacts')
    plt.title(f'Contact Score Distribution | Inliers vs Outliers (threshold={distance_thresh}m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Open3D Visualization
    visualize_contact_points_o3d(point_cloud, all_contact_points, inlier_mask, plane_equation)

# Example usage:
# file = 'grasp_predictions_ptv2-revised-robo-eval-loss-1_20250513_100529.npy' # early checkpint
# file = 'grasp_predictions_ptv2-revised-robo-eval_20250513_102322.npy' # fininshed checkpoint
file = 'results/ContactFormer/grasp_predictions_ptv2-revised-loss-1-10_20250514_132305.npy'
plot_contact_point_score_distribution(file, distance_thresh=0.01, device='cpu')
