# import torch
# import numpy as np
# from torch_ransac3d.plane import plane_fit
# import open3d as o3d
# import matplotlib.pyplot as plt
# # Mock class to hold your methods for debugging
# class PlaneLossDebugger:
#     def __init__(self, device='cpu'):
#         self.device = device


#     def plane_fit(self, pts, thresh=0.01, max_iterations=1000, iterations_per_batch=100, epsilon=1e-8):
#         # This is a simple mock for debugging (replace with actual fitting)
#         centroid = pts.mean(axis=0)
#         normal = np.array([0, 0, 1])  # Known plane normal (e.g. z-plane)
#         d = -centroid.dot(normal)
#         return torch.tensor(np.append(normal, d), dtype=torch.float32)

#     def _batch_plane_ransac(self, pc_batch):
#         # Works with both PyTorch tensors and NumPy arrays
        
#         # or np.random.rand(1000, 3)
#         def _plane_ransac(points):
#             # Fit a plane to the points using RANSAC
#             result = plane_fit(
#                 pts=points,
#                 thresh=0.02,
#                 max_iterations=1000,
#                 iterations_per_batch=100,
#                 epsilon=1e-8,
#                 device=self.device
#             )
#             return result.equation

#         plane_eqs = []
#         for i in range(pc_batch.shape[0]):
#             plane = _plane_ransac(pc_batch[i])
#             plane_eqs.append(plane)

#         return torch.stack(plane_eqs, dim=0).to(self.device) 

#     def _support_surface_loss(self,
#                           contact_points: torch.Tensor,
#                           pred_scores: torch.Tensor,
#                           plane_equation: torch.Tensor,
#                           alpha: float = 0.25,
#                           beta: float = 50.0) -> torch.Tensor:
#         """
#         Compute a normalized exponential distance loss from contact points to a plane.
    
#         Args:
#             contact_points (torch.Tensor): shape (B, N, 3) — batch of contact points.
#             plane_equation (torch.Tensor): shape (B, 4) — [a, b, c, d] for each batch.
#             alpha (float): scaling factor for exponential penalty.
#             beta (float): growth rate for exponential penalty.
    
#         Returns:
#             torch.Tensor: scalar normalized mean loss across batch.
#         """
#         # Extract normal vectors and offset (d)
#         normal = plane_equation[:, :3].to(self.device)      # (B, 3)
#         d = plane_equation[:, 3:].to(self.device)           # (B, 1)
#         normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-8  # (B, 1)
    
#         # Compute dot product between normals and contact points: (B, N)
#         dot = torch.einsum('bnd,bd->bn', contact_points, normal) + d  # (B, N)
    
#         # Compute distances
#         distances = torch.abs(dot) / normal_norm  # (B, N)
    
#         # Apply exponential penalty: alpha * exp(beta * distance)
#         penalized = pred_scores.squeeze(-1) *alpha * torch.exp(-1*beta * distances)  # (B, N)
    
#         # Normalize the total penalty per batch (max normalized value = 1)
#         max_penalty = torch.max(penalized, dim=1, keepdim=True)[0] * penalized.shape[1] + 1e-8  # (B, 1)
#         normalized_loss_per_batch = penalized.squeeze(1)# normalized_loss_per_batch = torch.sum(penalized, dim=1) / max_penalty.squeeze(1) #penalized.squeeze(1)  # (B,) torch.sum(penalized, dim=1) / max_penalty.squeeze(1)  # (B,)
    
#         # Final scalar loss across the batch
#         normalized_mean_loss = normalized_loss_per_batch.mean()  # scalar
    
#         return normalized_mean_loss
#     def visualize_plane_fit(self, points, plane_equation, pred_points):
#         points_np = points.cpu().numpy()
#         pred_points_np = pred_points.cpu().numpy()

#         plane_normal = plane_equation[:3].cpu().numpy()
#         plane_offset = plane_equation[3].item()

#         # Create Open3D point cloud
#         pc = o3d.geometry.PointCloud()
#         pc.points = o3d.utility.Vector3dVector(points_np)
#         pc.paint_uniform_color([0, 1.0, 0.0])

#         # Create predicted points cloud
#         pred_pc = o3d.geometry.PointCloud()
#         pred_pc.points = o3d.utility.Vector3dVector(pred_points_np)
#         pred_pc.paint_uniform_color([0.0, 0.0, 1])

#         # Create a plane mesh
#         plane_center = points_np.mean(axis=0)
#         plane_mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.001)
#         plane_mesh.translate(-plane_mesh.get_center())
#         plane_mesh.rotate(plane_mesh.get_rotation_matrix_from_xyz((0, 0, 0)))

#         # Align the plane mesh to plane normal
#         z_axis = np.array([0, 0, 1])
#         v = np.cross(z_axis, plane_normal)
#         c = np.dot(z_axis, plane_normal)
#         skew_sym_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
#         R = np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * (1 / (1 + c + 1e-8))

#         plane_mesh.rotate(R, center=(0, 0, 0))
#         plane_mesh.translate(plane_center)
#         plane_mesh.paint_uniform_color([0.9, 0.1, 0.1])

#         # Visualize
#         o3d.visualization.draw_geometries([pc, pred_pc, plane_mesh])

# # Testing function
# def debug_plane_loss():
#     # debugger = PlaneLossDebugger()

#     # # Generate synthetic point clouds batch (2 batches, points around z=0 plane)
#     # BATCH_SIZE = 2
#     # NUM_POINTS = 1000
#     # np.random.seed(42)

#     # pc_batch = np.random.rand(BATCH_SIZE, NUM_POINTS, 3) * 0.1
#     # pc_batch[..., 2] = pc_batch[..., 2] * 0.01  # Close to z-plane

#     # pc_batch_torch = torch.tensor(pc_batch, dtype=torch.float32)

#     # # Offset predicted points by +0.05 in z-direction to simulate bad predictions
#     # pred_points = pc_batch_torch.clone()
#     # pred_points[..., 2] += .01  # Shift away from plane by +5cm

#     # # Pred scores (uniform high confidence)
#     # pred_scores = torch.ones((BATCH_SIZE, NUM_POINTS, 1), dtype=torch.float32)

#     # # Fit plane equations
#     # plane_equations = debugger._batch_plane_ransac(pc_batch_torch)
#     # print("Plane equations fitted:")
#     # print(plane_equations)

#     # # Compute loss using the offset predicted points
#     # loss = debugger._support_surface_loss(pred_points, pred_scores, plane_equations)
#     # print(f"Computed loss: {loss.item()}")

#     # # Visualize: green = original points, blue = offset predicted points, red = fitted plane
#     # debugger.visualize_plane_fit(pc_batch_torch[0], plane_equations[0], pred_points[0])
#     # debugger = PlaneLossDebugger()

#     # # Generate synthetic point clouds (near z=0 plane)
#     # BATCH_SIZE = 2
#     # NUM_POINTS = 10000
#     # np.random.seed(42)

#     # pc_batch = np.random.rand(BATCH_SIZE, NUM_POINTS, 3) * 0.1
#     # pc_batch[..., 2] = pc_batch[..., 2] * 0.01  # Close to z-plane
#     # pc_batch_torch = torch.tensor(pc_batch, dtype=torch.float32)

#     # # Plane fitting
#     # plane_equations = debugger._batch_plane_ransac(pc_batch_torch)

#     # # Compute losses for 20 z-offsets from 0.001 to 1.0
#     # offsets = np.linspace(0.001, 1.0, 100)
#     # pred_scores = torch.ones((BATCH_SIZE, NUM_POINTS, 1), dtype=torch.float32)
#     # losses = []

#     # for offset in offsets:
#     #     pred_points = pc_batch_torch.clone()
#     #     pred_points[..., 2] += offset  # Offset predicted points in z-direction

#     #     loss = debugger._support_surface_loss(pred_points, pred_scores, plane_equations)
#     #     losses.append(loss.item())

#     # # Plot
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(offsets, losses, marker='o')
#     # plt.xlabel('Z Offset [m]')
#     # plt.ylabel('Support Surface Loss')
#     # plt.title('Loss vs. Z Offset of Predicted Points')
#     # plt.grid(True)
#     # plt.show()
    
#     debugger = PlaneLossDebugger()

#     # Generate synthetic flat point cloud near z=0 plane
#     BATCH_SIZE = 2
#     NUM_POINTS = 10000
#     np.random.seed(42)

#     pc_batch = np.random.rand(BATCH_SIZE, NUM_POINTS, 3) * 0.1
#     pc_batch[..., 2] = pc_batch[..., 2] * 0.01  # Close to z-plane
#     pc_batch_torch = torch.tensor(pc_batch, dtype=torch.float32)

#     # Plane fitting based on flat points
#     plane_equations = debugger._batch_plane_ransac(pc_batch_torch)

#     # Fixed spherical radius
#     sphere_radius = 0.05  # 5 cm sphere

#     # Shift sphere center in z-direction (simulate error)
#     shifts = np.linspace(0.0, 0.1, 20)  # shifts from 0 to 10 cm upwards
#     pred_scores = torch.ones((BATCH_SIZE, NUM_POINTS, 1), dtype=torch.float32)
#     losses = []

#     for shift in shifts:
#         # Generate spherical point cloud at origin
#         directions = np.random.normal(size=(BATCH_SIZE, NUM_POINTS, 3))
#         directions /= np.linalg.norm(directions, axis=2, keepdims=True)  # Normalize to unit sphere
#         sampled_radii = np.cbrt(np.random.rand(BATCH_SIZE, NUM_POINTS, 1)) * sphere_radius  # Uniform sphere volume
#         pred_points = directions * sampled_radii

#         # Shift sphere upwards in z-direction
#         pred_points[..., 2] += shift

#         pred_points_torch = torch.tensor(pred_points, dtype=torch.float32)

#         # Compute loss
#         loss = debugger._support_surface_loss(pred_points_torch, pred_scores, plane_equations)
#         losses.append(loss.item())

#     # Plot Loss vs Sphere Z-Shift
#     plt.figure(figsize=(10, 6))
#     plt.plot(shifts, losses, marker='o')
#     plt.xlabel('Z Shift of Sphere Center [m]')
#     plt.ylabel('Support Surface Loss')
#     plt.title('Loss vs Z-Shift of Spherical Predicted Points')
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     debug_plane_loss()


import torch
import matplotlib.pyplot as plt

B, N = 2, 2048
torch.manual_seed(0)

contact_points = torch.rand(B, N, 3) * 0.2
plane_equation = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(B, 1)
pred_scores = torch.rand(B, N, 1)

def support_surface_loss(contact_points, pred_scores, plane_equation, alpha=1.0, beta=10.0, topk_confidence=None):
    normal = plane_equation[:, :3]
    d = plane_equation[:, 3:]
    normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-8

    dot = torch.einsum('bnd,bd->bn', contact_points, normal) + d
    distances = torch.abs(dot) / normal_norm

    penalized = pred_scores.squeeze(-1) * alpha * torch.exp(-1 * beta * distances)

    if topk_confidence is not None:
        topk_penalties, _ = torch.topk(penalized, k=topk_confidence, dim=1)
        mean_loss = topk_penalties.mean()
    else:
        mean_loss = penalized.mean()

    return mean_loss.item()

k_values = [None, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
losses = []
for k in k_values:
    loss_val = support_surface_loss(contact_points, pred_scores, plane_equation, topk_confidence=k)
    losses.append(loss_val)

plt.figure(figsize=(10, 6))
plt.plot([N if k is None else k for k in k_values], losses, marker='o')
plt.xlabel('Top-k Used in Loss Computation')
plt.ylabel('Mean Support Surface Loss')
plt.title('Effect of Top-k Filtering on Support Surface Loss')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()