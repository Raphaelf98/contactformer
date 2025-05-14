import torch
import numpy as np
import open3d as o3d


def compute_approach_loss(pred_points, plane_eq, approach_vectors, contact_directions=None, alpha=10.0, beta=5.0, gamma=1.0):
    """
    Computes the approach direction loss given the data.
    """
    device = pred_points.device
    normal = plane_eq[:3].unsqueeze(0).to(device)  # (1, 3)
    normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-8)  # normalize (1,3)
    approach_vectors = approach_vectors / (torch.norm(approach_vectors, dim=1, keepdim=True) + 1e-8)  # (N,3)
    # Compute dot product between approach vectors and normal (N,)
    # dot_prods = torch.einsum('nd,bd->n', approach_vectors, normal)  # (N,)
    dot_prods = torch.sum(approach_vectors * normal, dim=1)  # (N,)
    tangential_penalty = torch.exp(-beta * dot_prods**2)  # (N,)
    into_surface_penalty = torch.relu(dot_prods)  # (N,)

    contact_loss = 1.0
    # if contact_directions is not None:
    #     dot_contact = torch.einsum('nd,bd->n', contact_directions, normal)  # (N,)
    #     contact_penalty = dot_contact**2  # (N,)
    #     contact_loss = contact_penalty.mean()

    total_loss = contact_loss * tangential_penalty.mean() + alpha * into_surface_penalty.mean()
    return total_loss.item()

def visualize_approach_contact(debug_file):
    # Load saved debug data
    data = torch.load(debug_file)
    pred_points = data['pred_points'][0]  # (N, 3) first batch
    plane_eq = data['plane_equation'][0]  # (4,)
    approach_vectors = data['approach_vectors'][0]  # (N, 3)
    contact_directions = data['contact_directions'][0] if data['contact_directions'] is not None else None

    # Compute loss
    loss_val = compute_approach_loss(pred_points, plane_eq, approach_vectors, contact_directions)
    print(f"Approach Loss: {loss_val:.6f}")

    # Convert to numpy
    pred_points_np = pred_points.cpu().numpy()
    approach_vectors_np = approach_vectors.cpu().numpy()

    # Plane normal & offset
    plane_normal = plane_eq[:3].cpu().numpy()
    plane_offset = plane_eq[3].item()

    # ==== Create Point Cloud ====
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pred_points_np)
    pcl.paint_uniform_color([0, 1.0, 0.0])  # Green points

    # ==== Visualize plane as mesh ====
    plane_center = pred_points_np.mean(axis=0)
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.001)
    plane_mesh.translate(-plane_mesh.get_center())
    plane_mesh.rotate(plane_mesh.get_rotation_matrix_from_xyz((0, 0, 0)))

    # Align plane normal
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, plane_normal)
    c = np.dot(z_axis, plane_normal)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + skew + skew @ skew * (1 / (1 + c + 1e-8))

    plane_mesh.rotate(R, center=(0, 0, 0))
    plane_mesh.translate(plane_center)
    plane_mesh.paint_uniform_color([0.9, 0.1, 0.1])  # Red plane

    # ==== Visualize Approach Vectors ====
    approach_arrows = []
    arrow_scale = 0.02  # Length of the arrow lines
    for point, vec in zip(pred_points_np, approach_vectors_np):
        line = o3d.geometry.LineSet()
        points = [point, point + vec * arrow_scale]
        lines = [[0, 1]]
        colors = [[0.0, 0.0, 1.0]]  # Blue for approach vectors

        line.points = o3d.utility.Vector3dVector(points)
        line.lines = o3d.utility.Vector2iVector(lines)
        line.colors = o3d.utility.Vector3dVector(colors)
        approach_arrows.append(line)

    # ==== Visualize Contact Directions ====
    contact_arrows = []
    if contact_directions is not None:
        contact_directions_np = contact_directions.cpu().numpy()
        for point, vec in zip(pred_points_np, contact_directions_np):
            line = o3d.geometry.LineSet()
            points = [point, point + vec * arrow_scale]
            lines = [[0, 1]]
            colors = [[1.0, 0.5, 0.0]]  # Orange for contact directions

            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector(lines)
            line.colors = o3d.utility.Vector3dVector(colors)
            contact_arrows.append(line)

  
    # ==== Visualize Plane Normal Vector ====
    normal_arrow = o3d.geometry.LineSet()
    normal_length = 0.05  # Length of the normal arrow

    normal_start = plane_center
    normal_end = plane_center + plane_normal * normal_length

    normal_arrow.points = o3d.utility.Vector3dVector([normal_start, normal_end])
    normal_arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_arrow.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # Yellow normal vector

    # ==== Visualize ====
    geometry_list = [pcl, plane_mesh, normal_arrow] + approach_arrows + contact_arrows
    o3d.visualization.draw_geometries(geometry_list)



if __name__ == "__main__":
    debug_file = "/home/raphael/thesis/contact_former/contact_grasp_net/debug_ss_loss.pt"  # adjust path if needed
    visualize_approach_contact(debug_file)

