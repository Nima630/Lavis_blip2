import torch
import numpy as np
import os


def crop_lidar_bev_to_camera_fov(lidar_feat, lidar2img, img_shape, bev_resolution, pc_range):
    """
    Crops LiDAR BEV features to the visible region of the front camera.

    Args:
        lidar_feat (torch.Tensor): LiDAR BEV feature map [C, H, W]
        lidar2img (np.ndarray): 4x4 lidar-to-image projection matrix
        img_shape (tuple): (H, W) of image
        bev_resolution (tuple): resolution per pixel (dx, dy)
        pc_range (list): point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        torch.Tensor: Cropped (masked) LiDAR BEV features of shape [C, H, W]
    """
    C, H, W = lidar_feat.shape
    dx = (pc_range[3] - pc_range[0]) / W
    dy = (pc_range[4] - pc_range[1]) / H

    xs = np.linspace(pc_range[0] + dx / 2, pc_range[3] - dx / 2, W)
    ys = np.linspace(pc_range[1] + dy / 2, pc_range[4] - dy / 2, H)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros_like(xs)

    points_lidar = np.stack([xs, ys, zs, np.ones_like(xs)], axis=-1).reshape(-1, 4)
    points_img = (lidar2img @ points_lidar.T).T
    points_img[:, 0] /= points_img[:, 2] + 1e-5
    points_img[:, 1] /= points_img[:, 2] + 1e-5

    img_w, img_h = img_shape[1], img_shape[0]
    valid = (
        (points_img[:, 0] >= 0) & (points_img[:, 0] < img_w) &
        (points_img[:, 1] >= 0) & (points_img[:, 1] < img_h) &
        (points_img[:, 2] > 0)
    )
    valid_mask = valid.reshape(H, W)
    if not isinstance(valid_mask, torch.Tensor):
        valid_mask = torch.from_numpy(valid_mask)
    valid_mask = valid_mask.float().to(lidar_feat.device)

    cropped_feat = lidar_feat * valid_mask
    # soft masking
    # invalid_weight = 0.3
    # soft_mask = valid_mask * 1.0 + (1.0 - valid_mask) * invalid_weight
    # cropped_feat = lidar_feat * soft_mask

    return cropped_feat


def run_crop_from_files(camera_path, lidar_path, bev_resolution, pc_range):
    camera_data = torch.load(camera_path, weights_only=False)
    lidar_data = torch.load(lidar_path, weights_only=False)

    lidar_feat = lidar_data['feat'].squeeze(0)  # [C, H, W]
    lidar2img = camera_data['lidar2img']
    img_shape = tuple(camera_data['img_shape'][0])[:2]

    cropped = crop_lidar_bev_to_camera_fov(
        lidar_feat, lidar2img, img_shape, bev_resolution, pc_range
    )
    return cropped


if __name__ == "__main__":
    # Example usage
    camera_example = "/home/draiman/Desktop/Datasets/validation/resnet/CAM_FRONT/ffec010a27b4462f86b78e2e41d14f42.pt"
    lidar_example = "/home/draiman/Desktop/Datasets/validation/lidar_val/ffec010a27b4462f86b78e2e41d14f42.pt"

    bev_resolution = (1.14, 1.14)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    _ = run_crop_from_files(camera_example, lidar_example, bev_resolution, pc_range)
