# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import open3d as o3d
# from torch_ransac3d.plane import plane_fit

# class PlaneFitter:
#     def __init__(self, device='cpu'):
#         self.device = device

#     def batch_plane_ransac(self, pc_batch):
#         def _plane_ransac(points):
#             result = plane_fit(
#                 pts=points,
#                 thresh=0.005,
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

#         return torch.stack(plane_eqs, dim=0).to(self.device)

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import open3d as o3d
# from torch_ransac3d.plane import plane_fit
# from tqdm import tqdm


# def process_file(npy_file, distance_thresh=0.01, device='cpu'):
#     data = np.load(npy_file, allow_pickle=True).item()
#     point_cloud = data['point_cloud']
#     contact_pts = data['contact_pts']
#     scores_dict = data['scores']

#     all_contact_points = []
#     all_scores = []
#     for seg_id, points in contact_pts.items():
#         if len(points) > 0:
#             all_contact_points.append(points)
#             all_scores.append(scores_dict[seg_id])

#     if len(all_contact_points) == 0:
#         return None, None

#     all_contact_points = np.vstack(all_contact_points)
#     all_scores = np.concatenate(all_scores)

#     pc_batch = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)
#     fitter = PlaneFitter(device=device)
#     plane_equation = fitter.batch_plane_ransac(pc_batch)[0]

#     normal = plane_equation[:3]
#     d = plane_equation[3]
#     normal_norm = torch.norm(normal) + 1e-8

#     contact_points_torch = torch.tensor(all_contact_points, dtype=torch.float32).to(device)
#     distances = torch.abs(torch.matmul(contact_points_torch, normal) + d) / normal_norm
#     distances = distances.cpu().numpy()
#     inlier_mask = distances < distance_thresh

#     return all_scores[inlier_mask], all_scores[~inlier_mask]

# def evaluate_directory_with_std(dir_path, distance_thresh=0.01, device='cpu'):
#     inlier_scores_all = []
#     outlier_scores_all = []

#     files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
#     if not files:
#         print("No .npy files found in:", dir_path)
#         return

#     for file in tqdm(files, desc="Collecting scores"):
#         path = os.path.join(dir_path, file)
#         try:
#             inliers, outliers = process_file(path, distance_thresh, device)
#             if inliers is not None and len(inliers) > 0:
#                 inlier_scores_all.extend(inliers)
#             if outliers is not None and len(outliers) > 0:
#                 outlier_scores_all.extend(outliers)
#         except Exception as e:
#             print(f"Error processing {file}: {e}")

#     if len(inlier_scores_all) == 0 and len(outlier_scores_all) == 0:
#         print("No usable contact point data.")
#         return

#     # Combine scores and define histogram bins
#     all_scores_combined = np.array(inlier_scores_all + outlier_scores_all)
#     score_min = np.min(all_scores_combined)
#     score_max = np.max(all_scores_combined)
#     bins = np.linspace(score_min, score_max, 31)
#     bin_centers = (bins[:-1] + bins[1:]) / 2

#     # Rebuild per-scene histograms
#     inlier_hists = []
#     outlier_hists = []

#     for file in tqdm(files, desc="Building histograms"):
#         path = os.path.join(dir_path, file)
#         try:
#             inliers, outliers = process_file(path, distance_thresh, device)
#             if inliers is not None and len(inliers) > 0:
#                 inlier_hists.append(np.histogram(inliers, bins=bins)[0])
#             if outliers is not None and len(outliers) > 0:
#                 outlier_hists.append(np.histogram(outliers, bins=bins)[0])
#         except:
#             continue

#     # Convert to arrays and compute mean/std
#     if inlier_hists:
#         inlier_hists = np.stack(inlier_hists)
#         inlier_hist_mean = np.mean(inlier_hists, axis=0)
#         inlier_hist_std = np.std(inlier_hists, axis=0)
#     else:
#         inlier_hist_mean = inlier_hist_std = np.zeros_like(bin_centers)

#     if outlier_hists:
#         outlier_hists = np.stack(outlier_hists)
#         outlier_hist_mean = np.mean(outlier_hists, axis=0)
#         outlier_hist_std = np.std(outlier_hists, axis=0)
#     else:
#         outlier_hist_mean = outlier_hist_std = np.zeros_like(bin_centers)

#     plt.figure(figsize=(10, 6))

#     width = (bins[1] - bins[0]) * 0.8
#     bin_centers = (bins[:-1] + bins[1:]) / 2

#     # Plot bars per bin, choosing order based on height
#     for i, center in enumerate(bin_centers):
#         in_val = inlier_hist_mean[i]
#         out_val = outlier_hist_mean[i]

#         if in_val > out_val:
#             # Draw in-plane (red) in back, object (green) in front
#             plt.bar(center, in_val, width=width, color='red', edgecolor='black', label='In-plane Predictions' if i == 0 else "")
#             plt.bar(center, out_val, width=width, color='green', edgecolor='black', label='Object Predictions' if i == 0 else "")
#         else:
#             # Draw object (green) in back, in-plane (red) in front
#             plt.bar(center, out_val, width=width, color='green', edgecolor='black', label='Object Predictions' if i == 0 else "")
#             plt.bar(center, in_val, width=width , color='red', edgecolor='black', label='In-plane Predictions' if i == 0 else "")

#     # Axis labels and title
#     plt.xlabel('Confidence Score')
#     plt.ylabel('Avg Number of Predictions per Scene (per Bin)')
#     plt.title(f'Prediction Distribution' )
#     plt.grid(True)
#     plt.legend()

#     # Stats text
#     avg_in_plane = np.mean(np.sum(inlier_hists, axis=1)) if inlier_hists.size > 0 else 0
#     avg_object = np.mean(np.sum(outlier_hists, axis=1)) if outlier_hists.size > 0 else 0
#     # Build stats text
#     stats_text = (f"Avg In-plane Predictions per Scene: {avg_in_plane:.1f}\n"
#               f"Avg Object Predictions per Scene: {avg_object:.1f}")

#     # Use axes coordinates (0,1) = top-left
#     plt.gca().text(0.01, 0.98, stats_text,
#                 transform=plt.gca().transAxes,
#                 fontsize=10, verticalalignment='top',
#                 bbox=dict(facecolor='white', alpha=0.6))

#     plt.tight_layout()
#     plt.show()
#     # Print statistics
#     print(f"\nScore Ranges:")
#     print(f"  Inliers  → min: {np.min(inlier_scores_all):.4f}, max: {np.max(inlier_scores_all):.4f}, avg: {np.mean(inlier_scores_all):.4f}")
#     print(f"  Outliers → min: {np.min(outlier_scores_all):.4f}, max: {np.max(outlier_scores_all):.4f}, avg: {np.mean(outlier_scores_all):.4f}")
#     print(f"  All      → avg: {np.mean(all_scores_combined):.4f}")

# # Example usage
# evaluate_directory_with_std('results/ptv2-final', distance_thresh=0.01, device='cpu')
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from torch_ransac3d.plane import plane_fit
import os
from tqdm import tqdm

# Use LaTeX rendering for fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

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

def process_file(npy_file, distance_thresh=0.01, device='cpu'):
    data = np.load(npy_file, allow_pickle=True).item()
    point_cloud = data['point_cloud']
    contact_pts = data['contact_pts']
    scores_dict = data['scores']

    all_contact_points = []
    all_scores = []
    for seg_id, points in contact_pts.items():
        if len(points) > 0:
            all_contact_points.append(points)
            all_scores.append(scores_dict[seg_id])

    if len(all_contact_points) == 0:
        return None, None

    all_contact_points = np.vstack(all_contact_points)
    all_scores = np.concatenate(all_scores)

    pc_batch = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)
    fitter = PlaneFitter(device=device)
    plane_equation = fitter.batch_plane_ransac(pc_batch)[0]

    normal = plane_equation[:3]
    d = plane_equation[3]
    normal_norm = torch.norm(normal) + 1e-8

    contact_points_torch = torch.tensor(all_contact_points, dtype=torch.float32).to(device)
    distances = torch.abs(torch.matmul(contact_points_torch, normal) + d) / normal_norm
    distances = distances.cpu().numpy()
    inlier_mask = distances < distance_thresh

    return all_scores[inlier_mask], all_scores[~inlier_mask]

def evaluate_directory_with_std(dir_path, distance_thresh=0.01, device='cpu'):
    inlier_scores_all = []
    outlier_scores_all = []

    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    if not files:
        print("No .npy files found in:", dir_path)
        return

    for file in tqdm(files, desc="Collecting scores"):
        path = os.path.join(dir_path, file)
        try:
            inliers, outliers = process_file(path, distance_thresh, device)
            if inliers is not None and len(inliers) > 0:
                inlier_scores_all.extend(inliers)
            if outliers is not None and len(outliers) > 0:
                outlier_scores_all.extend(outliers)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if len(inlier_scores_all) == 0 and len(outlier_scores_all) == 0:
        print("No usable contact point data.")
        return

    all_scores_combined = np.array(inlier_scores_all + outlier_scores_all)
    score_min = np.min(all_scores_combined)
    score_max = np.max(all_scores_combined)
    bins = np.linspace(score_min, score_max, 31)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    inlier_hists = []
    outlier_hists = []

    for file in tqdm(files, desc="Building histograms"):
        path = os.path.join(dir_path, file)
        try:
            inliers, outliers = process_file(path, distance_thresh, device)
            if inliers is not None and len(inliers) > 0:
                inlier_hists.append(np.histogram(inliers, bins=bins)[0])
            if outliers is not None and len(outliers) > 0:
                outlier_hists.append(np.histogram(outliers, bins=bins)[0])
        except:
            continue

    if inlier_hists:
        inlier_hists = np.stack(inlier_hists)
        inlier_hist_mean = np.mean(inlier_hists, axis=0)
        inlier_hist_std = np.std(inlier_hists, axis=0)
    else:
        inlier_hist_mean = inlier_hist_std = np.zeros_like(bin_centers)

    if outlier_hists:
        outlier_hists = np.stack(outlier_hists)
        outlier_hist_mean = np.mean(outlier_hists, axis=0)
        outlier_hist_std = np.std(outlier_hists, axis=0)
    else:
        outlier_hist_mean = outlier_hist_std = np.zeros_like(bin_centers)

    plt.figure(figsize=(10, 6))
    width = (bins[1] - bins[0]) * 0.8

    for i, center in enumerate(bin_centers):
        in_val = inlier_hist_mean[i]
        out_val = outlier_hist_mean[i]

        if in_val > out_val:
            plt.bar(center, in_val, width=width, color='red', edgecolor='black',
                    label=r'\textbf{In-plane Predictions}' if i == 0 else "")
            plt.bar(center, out_val, width=width, color='green', edgecolor='black',
                    label=r'\textbf{Object Predictions}' if i == 0 else "")
        else:
            plt.bar(center, out_val, width=width, color='green', edgecolor='black',
                    label=r'\textbf{Object Predictions}' if i == 0 else "")
            plt.bar(center, in_val, width=width, color='red', edgecolor='black',
                    label=r'\textbf{In-plane Predictions}' if i == 0 else "")

    plt.xlabel(r"\textbf{Confidence Score}")
    plt.ylabel(r"\textbf{Avg Number of Predictions per Scene (per Bin)}")
    plt.title(r"\textbf{Prediction Distribution}")
    plt.grid(True)
    plt.legend()

    avg_in_plane = np.mean(np.sum(inlier_hists, axis=1)) if inlier_hists.size > 0 else 0
    avg_object = np.mean(np.sum(outlier_hists, axis=1)) if outlier_hists.size > 0 else 0

    stats_text = (
        rf"\textbf{{Avg In-plane Predictions per Scene}}: {avg_in_plane:.1f}" "\n"
        rf"\textbf{{Avg Object Predictions per Scene}}: {avg_object:.1f}"
    )

    plt.gca().text(
        0.01, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6)
    )

    plt.tight_layout()
    # plt.savefig("/home/raphael/mount/plots/prediction_distribution_cfloss.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("/home/raphael/mount/plots/prediction_distribution_cf.pdf", format="pdf", bbox_inches="tight")
    
    plt.show()

    print(f"\nScore Ranges:")
    print(f"  Inliers  → min: {np.min(inlier_scores_all):.4f}, max: {np.max(inlier_scores_all):.4f}, avg: {np.mean(inlier_scores_all):.4f}")
    print(f"  Outliers → min: {np.min(outlier_scores_all):.4f}, max: {np.max(outlier_scores_all):.4f}, avg: {np.mean(outlier_scores_all):.4f}")
    print(f"  All      → avg: {np.mean(all_scores_combined):.4f}")

# Example usage
evaluate_directory_with_std('results/ptv2-revised-robo-eval', distance_thresh=0.01, device='cpu')
# evaluate_directory_with_std('results/ptv2-final', distance_thresh=0.01, device='cpu')
