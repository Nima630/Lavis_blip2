
import os
import glob
import torch
import numpy as np
import math
from camera_bev_homography import project_to_bev_homography

# ------------ Settings -------------
fpn_level = 0  # 0 = P2, 1 = P3, 2 = P4, 3 = P5 (BEVFormer default: 2 or 3)
camera_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT"
lidar_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/lidar"
orig_hw = (928, 1600)  # [H_img, W_img] (from nuScenes, adjust if different)
K_img = np.array([
    [1266.4172, 0, 816.2670],
    [0, 1266.4172, 491.5070],
    [0, 0, 1]
])

def scale_K(K, orig_hw, feat_hw):
    K = K.copy()
    s_x = feat_hw[1] / orig_hw[1]
    s_y = feat_hw[0] / orig_hw[0]
    K[0, 0] *= s_x
    K[0, 2] *= s_x
    K[1, 1] *= s_y
    K[1, 2] *= s_y
    return K

def get_matched_files(camera_root, lidar_root, fpn_level):
    # Match only the chosen FPN level
    camera_files = {
        os.path.splitext(os.path.basename(f))[0].replace(f"_lvl{fpn_level}", ""): f
        for f in glob.glob(os.path.join(camera_root, f"*lvl{fpn_level}.pt"))
    }
    lidar_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(lidar_dir, "*.pt"))
    }
    matched_keys = sorted(set(camera_files) & set(lidar_files))
    cam_paths = [camera_files[k] for k in matched_keys]
    lid_paths = [lidar_files[k] for k in matched_keys]
    return cam_paths, lid_paths

def main():
    cam_files, lid_files = get_matched_files(camera_dir, lidar_dir, fpn_level)
    print(f"Found {len(cam_files)} matched pairs for FPN level {fpn_level}")

    for i in range(1):
        cam = torch.load(cam_files[i], map_location='cpu')
        lid = torch.load(lid_files[i], map_location='cpu')
        cam_feat = cam['feat']        # [1,256,H_feat,W_feat] or [256,H_feat,W_feat]
        lid_feat = lid['feat']        # [1,256,90,90] or [256,90,90]

        # Always ensure cam_feat is 4D [B,C,H,W]
        if cam_feat.ndim == 3:
            cam_feat = cam_feat.unsqueeze(0)
        print(f"Sample {i}")
        print(f"  Camera feature shape: {cam_feat.shape}")
        print(f"  LiDAR feature shape:  {lid_feat.shape}")

        # Dynamically scale K for this feature map
        feat_hw = cam_feat.shape[-2:]        # (H_feat, W_feat)
        print(" Feature map hw:", feat_hw)
        K_scaled = scale_K(K_img, orig_hw, feat_hw)

        # Camera pose (example: 1.5m height, -10° pitch)
        T = np.eye(4)
        T[2, 3] = 1.5   # height in meters
        # pitch = math.radians(-10)
        val = -35
        pitch = math.radians(val)
        print("Camera pitch radians(", val ,"):", pitch)
        R = np.array([
            [1, 0, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch), 0],
            [0, math.sin(pitch),  math.cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        T = R @ T
        bev_bounds = (0, 4, -1, 1)
        # bev_bounds=(0, 8, -2, 2)
        # bev_bounds = (0, 15, -5, 5)
        print("BEV bounds:", bev_bounds)
        # Project to BEV (use a moderate BEV crop)
        cam_bev = project_to_bev_homography(
            cam_feat,
            K_scaled,
            T,
            bev_bounds,  # 40m forward, 20m wide
            bev_shape=(120, 120),
            T_is_ego2cam=True
        )
        # Mask reporting if implemented in your projector
        print("Projected camera BEV:", cam_bev.shape)

if __name__ == "__main__":
    main()













































































