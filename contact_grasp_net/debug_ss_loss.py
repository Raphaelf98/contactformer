import torch
import numpy as np
from torch_ransac3d.plane import plane_fit
import open3d as o3d
import matplotlib.pyplot as plt
# Mock class to hold your methods for debugging
class PlaneLossDebugger:
    def __init__(self, device='cpu'):
        self.device = device


    def plane_fit(self, pts, thresh=0.01, max_iterations=1000, iterations_per_batch=100, epsilon=1e-8):
        # This is a simple mock for debugging (replace with actual fitting)
        centroid = pts.mean(axis=0)
        normal = np.array([0, 0, 1])  # Known plane normal (e.g. z-plane)
        d = -centroid.dot(normal)
        return torch.tensor(np.append(normal, d), dtype=torch.float32)

    def _batch_plane_ransac(self, pc_batch):
        # Works with both PyTorch tensors and NumPy arrays
        
        # or np.random.rand(1000, 3)
        def _plane_ransac(points):
            # Fit a plane to the points using RANSAC
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
    def visualize_signed_distances(self,signed_distances: torch.Tensor, title="Signed Distances Distribution"):
        """
        Visualizes the distribution of signed distances using a histogram.

        Args:
            signed_distances (torch.Tensor): shape (B, N) tensor of signed distances.
            title (str): Title of the plot.
        """
        # Convert to numpy
        signed_distances_np = signed_distances.cpu().detach().numpy().flatten()

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(signed_distances_np, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

        # Highlight zero line
        plt.axvline(0, color='red', linestyle='--', label='Surface (0 distance)')

        # Titles and labels
        plt.title(title)
        plt.xlabel('Signed Distance to Plane')
        plt.ylabel('Number of Points')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _support_surface_loss(self,
                          contact_points: torch.Tensor,
                          pred_scores: torch.Tensor,
                          plane_equation: torch.Tensor,
                          alpha: float = 1.,
                          beta: float = 10.0,
                          below_surface_penalty: float = 1.0,
                          topk_confidence:float = 512) -> torch.Tensor:
        """
        Compute a normalized exponential distance loss from contact points to a plane.
    
        Args:
            contact_points (torch.Tensor): shape (B, N, 3) — batch of contact points.
            plane_equation (torch.Tensor): shape (B, 4) — [a, b, c, d] for each batch.
            alpha (float): scaling factor for exponential penalty.
            beta (float): growth rate for exponential penalty.
    
        Returns:
            torch.Tensor: scalar normalized mean loss across batch.
        """
        # Extract normal vectors and offset (d)
        normal = plane_equation[:, :3].to(self.device)      # (B, 3)
        d = plane_equation[:, 3:].to(self.device)           # (B, 1)
        normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-8  # (B, 1)
    
        # Compute dot product between normals and contact points: (B, N)
        dot = torch.einsum('bnd,bd->bn', contact_points, normal)# + d  # (B, N)
    
        # Compute distances
        distances = dot / normal_norm  # (B, N)
    
        # Apply exponential penalty: alpha * exp(beta * distance)
        penalized = pred_scores.squeeze(-1) *alpha * torch.exp(-1*beta * distances)  # (B, N)
        if topk_confidence is not None and topk_confidence > 0:
            # Select top-k highest penalties per batch sample
            topk_penalties, _ = torch.topk(penalized, k=topk_confidence, dim=1)  # (B, k)
            mean_loss = topk_penalties.mean()  # scalar
        else:
            mean_loss = penalized.mean()  # fallback to normal mean loss

        # mean_loss = penalized.squeeze(1)  # scalar
        # mean_loss = mean_loss.mean()  # scalar

        return mean_loss #normalized_mean_loss
    
    def visualize_plane_fit(self, points, plane_equation, pred_points):
        points_np = points.cpu().numpy()
        pred_points_np = pred_points.cpu().numpy()

        plane_normal = plane_equation[:3].cpu().numpy()
        plane_offset = plane_equation[3].item()

        # === Point Cloud (Scene Points - green) ===
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_np)
        pc.paint_uniform_color([0.0, 1.0, 0.0])

        # === Predicted Contact Points (blue) ===
        pred_pc = o3d.geometry.PointCloud()
        pred_pc.points = o3d.utility.Vector3dVector(pred_points_np)
        pred_pc.paint_uniform_color([0.0, 0.0, 1.0])

        # === Compute Bounding Box Size for Plane Mesh ===
        bbox = pc.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        plane_size_x = extent[0] * 1.2
        plane_size_y = extent[1] * 1.2

        # === Create Plane Mesh ===
        plane_mesh = o3d.geometry.TriangleMesh.create_box(width=plane_size_x, height=plane_size_y, depth=0.001)
        plane_mesh.translate(-plane_mesh.get_center())  # Center at origin

        # === Align Plane Normal ===
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, plane_normal)
        c = np.dot(z_axis, plane_normal)
        if np.linalg.norm(v) > 1e-6:
            skew_sym_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * (1 / (1 + c + 1e-8))
            plane_mesh.rotate(R, center=(0, 0, 0))

        # === Offset Plane to Correct Position using d ===
        plane_offset_vector = -plane_offset * plane_normal / (np.linalg.norm(plane_normal) ** 2 + 1e-8)
        plane_mesh.translate(plane_offset_vector)

        plane_mesh.paint_uniform_color([0.9, 0.1, 0.1])  # Red plane

        # === Plane Normal Arrow (yellow) ===
        normal_arrow = o3d.geometry.LineSet()
        normal_length = 0.05  # arrow length
        normal_start = plane_offset_vector
        normal_end = normal_start + plane_normal * normal_length
        normal_arrow.points = o3d.utility.Vector3dVector([normal_start, normal_end])
        normal_arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
        normal_arrow.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])

        # === Visualize All ===
        o3d.visualization.draw_geometries([pc, pred_pc, plane_mesh, normal_arrow])



# Assuming PlaneLossDebugger is already implemented as you posted.
# We are focusing on fixing the debug_plane_loss function now.

def debug_plane_loss():
    debugger = PlaneLossDebugger()

    # Load .npy data file
    file = '/home/raphael/thesis/contact_former/analysis/results/ContactFormer/grasp_predictions_ptv2-revised-loss-1-10_20250514_132305.npy'
    data = np.load(file, allow_pickle=True).item()

    # Extract point cloud (N, 3)
    pc_np = data['point_cloud']
    pc_batch_torch = torch.tensor(pc_np, dtype=torch.float32).unsqueeze(0)  # (B=1, N, 3)

    # Extract contact points (dict -> stack to tensor)
    contact_pts_list = []
    for seg_id in data['contact_pts']:
        contact_pts_list.append(torch.tensor(data['contact_pts'][seg_id], dtype=torch.float32))
    contact_pts = torch.cat(contact_pts_list, dim=0).unsqueeze(0)  # (B=1, N, 3)

    # Extract scores (dict -> stack to tensor)
    scores_list = []
    for seg_id in data['scores']:
        scores_list.append(torch.tensor(data['scores'][seg_id], dtype=torch.float32))
    pred_scores = torch.cat(scores_list, dim=0).unsqueeze(0).unsqueeze(-1)  # (B=1, N, 1)

    # Plane fitting on point cloud batch
    plane_equations = debugger._batch_plane_ransac(pc_batch_torch)  # (B, 4)

    # Compute support surface loss
    loss = debugger._support_surface_loss(contact_pts, pred_scores, plane_equations)

    print(f"Support Surface Loss: {loss.item()}")

    # Visualization (optional but useful)
    debugger.visualize_plane_fit(pc_batch_torch[0], plane_equations[0], contact_pts[0])




if __name__ == "__main__":
    # import open3d as o3d
    # import numpy as np

    # # Load geometry
    # pc = o3d.io.read_point_cloud("./debug_vis/support_surface_contacts_2.ply")
    # mesh = o3d.io.read_triangle_mesh("./debug_vis/support_surface_plane_2.ply")
    # normals = o3d.io.read_triangle_mesh("./debug_vis/support_surface_normal_arrow_2.ply")

    # # Load plane equation from your own saved info
    # # Let's say you know this:


    # # Also add world frame at origin
    # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([pc, mesh, normals, world_frame])
    import torch
    import matplotlib.pyplot as plt

    # Parameters
    alpha = 3.0
    beta = 200.0

    # Simulate distances (e.g., fro
    # m -5cm to +5cm)
    distances = torch.linspace(-0.1, 0.05, 500)
    softplus = torch.nn.Softplus()

    # Compute penalty using torch
    distances = torch.clamp(distances, min=-0.02)
    penalized = alpha * softplus(-beta * distances)

    # Plot the result
    plt.figure(figsize=(8, 5))
    plt.plot(distances.numpy(), penalized.detach().numpy(), label='Penalty (Torch Softplus)')
    plt.xlabel("Distance to support surface (meters)")
    plt.ylabel("Penalty")
    plt.title("Torch Softplus Penalty: α = 3, β = 200")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()