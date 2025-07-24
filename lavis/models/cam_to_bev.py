import torch
import matplotlib.pyplot as plt

def create_bev_grid(H=90, W=90, x_range=(-51.2, 51.2), y_range=(-51.2, 51.2)):
    xs = torch.linspace(x_range[0], x_range[1], W)
    ys = torch.linspace(y_range[0], y_range[1], H)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    zeros = torch.zeros_like(xx)
    bev_xyz = torch.stack([xx, yy, zeros, torch.ones_like(xx)], dim=-1)  # [H, W, 4]
    return bev_xyz.view(-1, 4)  # [H*W, 4]

def project_points_with_lidar2img(bev_points, lidar2img):
    proj = (lidar2img @ bev_points.T).T  # [N, 4]
    eps = 1e-5
    proj[:, :2] = proj[:, :2] / (proj[:, 2:3] + eps)  # Normalize by Z
    return proj[:, :2], proj[:, 2]

def generate_bev_feature_map(cam_feats, lidar2img_matrices, image_size):
    N_cam, _, C, Hf, Wf = cam_feats.shape
    bev_grid = create_bev_grid(H=90, W=90)
    bev_features = torch.zeros(bev_grid.shape[0], C)
    counts = torch.zeros(bev_grid.shape[0], 1)

    for i in range(N_cam):
        img_feat = cam_feats[i, 0]  # [C, Hf, Wf]
        proj_pts, depth = project_points_with_lidar2img(bev_grid, lidar2img_matrices[i])

        valid = (depth > 0) & \
                (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < image_size[1]) & \
                (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < image_size[0])

        px = proj_pts[valid].long()
        feat_vals = img_feat[:, px[:, 1], px[:, 0]].T  # [N, C]
        bev_features[valid] += feat_vals
        counts[valid] += 1

    counts[counts == 0] = 1
    bev_features /= counts
    return bev_features.view(90, 90, C)

import torch

def generate_bev_feature_fpn_map(cam_feats, lidar2img_matrices, image_size, use_confidence=False):
    N_cam, _, C, Hf, Wf = cam_feats.shape
    bev_grid = create_bev_grid(H=90, W=90)
    bev_features = torch.zeros(bev_grid.shape[0], C)
    weights = torch.zeros(bev_grid.shape[0], 1)

    for i in range(N_cam):
        img_feat = cam_feats[i, 0]  # [C, Hf, Wf]
        proj_pts, depth = project_points_with_lidar2img(bev_grid, lidar2img_matrices[i])

        if torch.isnan(proj_pts).any() or torch.isinf(proj_pts).any():
            print(f"[WARN] NaN or Inf in proj_pts for camera {i}")
            continue

        if proj_pts.ndim != 2 or proj_pts.shape[1] != 2:
            print(f"[ERROR] Invalid proj_pts shape for camera {i}: {proj_pts.shape}")
            continue

        # Scale projections to feature map resolution
        proj_pts[:, 0] *= (Wf / 1600.0)
        proj_pts[:, 1] *= (Hf / 928.0)

        valid = (depth > 0) & \
                (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < image_size[1]) & \
                (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < image_size[0])

        px = proj_pts[valid].long()
        feat_vals = img_feat[:, px[:, 1], px[:, 0]].T  # [N, C]

        if use_confidence:
            confidence = compute_confidence_scores(feat_vals)  # [N, 1]
            fuse_with_confidence(bev_features, weights, valid, feat_vals, confidence)
        else:
            fuse_with_average(bev_features, weights, valid, feat_vals)

    weights[weights == 0] = 1
    bev_features /= weights

    bev_grid = bev_features.view(90, 90, C).permute(2, 0, 1)
    bev_features_raw = bev_features.clone()

    return bev_grid, bev_features_raw, weights

def fuse_with_average(bev_features, weights, valid_mask, features):
    bev_features[valid_mask] += features
    weights[valid_mask] += 1

def compute_confidence_scores(features):
    # Simple confidence: L2 norm per feature vector
    return features.norm(dim=1, keepdim=True)

def fuse_with_confidence(bev_features, weights, valid_mask, features, confidence):
    bev_features[valid_mask] += features * confidence
    weights[valid_mask] += confidence


def main():
    # Load from your dataset
    sample = torch.load("path/to/sample.pt")  # Replace with actual sample path

    cam_feats = sample["images"]           # [6, 1, 2048, 29, 50]
    lidar2img = sample["lidar2img"]        # [6, 4, 4]
    img_shape = sample["img_shapes"][0]    # [(928, 1600, 3)], we only need [0:2]

    bev_map = generate_bev_feature_map(cam_feats, lidar2img, image_size=(29, 50))
    bev_map_np = bev_map.detach().cpu().numpy()

    # Visualize one channel
    plt.imshow(bev_map_np[:, :, 0], cmap="viridis")
    plt.title("BEV Map (Channel 0)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
