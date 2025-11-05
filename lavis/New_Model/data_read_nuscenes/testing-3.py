#!/usr/bin/env python3
"""
nuScenes CAM->BEV projection (organized and refactored rewrite)
- Reads pre-processed metadata from a CSV file for fast data lookups.
- Computes ground-plane intersections and splats features into a BEV grid.
Author: you :)
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion

# (The data models NuScenesPaths, CameraCalib, and BEVSpec remain the same) @1
@dataclass
class NuScenesPaths:
    metadata_csv_path: str
    camera_feat_dir: str

@dataclass
class CameraCalib:
    K: np.ndarray      # (3, 3)
    R: np.ndarray      # (3, 3) sensor->ego rotation
    T: np.ndarray      # (3,)   sensor->ego translation

@dataclass
class BEVSpec:
    x_min: float = 0.0
    x_max: float = 50.0
    cell_x: float = 0.4
    y_min: float = -25.0
    y_max: float = 25.0
    cell_y: float = 0.25

    @property
    def H(self) -> int:
        return int(np.ceil((self.x_max - self.x_min) / self.cell_x))

    @property
    def W(self) -> int:
        return int(np.ceil((self.y_max - self.y_min) / self.cell_y))

# ------------------------------- I/O and Data Parsing ------------------------- #
def torch_load(path: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")

# def extract_calibration_from_row(row: pd.Series) -> CameraCalib: # @1 @2
#     """Build intrinsics/extrinsics from a pandas Series (a row from the CSV)."""
#     K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3)
    
#     # Rotation is stored as a quaternion [w, x, y, z]
#     quat_elements = [float(x) for x in row["sensor2ego_rotation"].split(';')]
#     R = Quaternion(quat_elements).rotation_matrix
    
#     T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)
#     return CameraCalib(K=K, R=R, T=T)

def extract_calibration_from_row(row: pd.Series) -> CameraCalib:
    # K will now be a string of 9 float values separated by semicolons (e.g., 'f1;f2;f3;f4;...')
    
    # 1. Split the string into a list of 9 string elements
    # 2. Convert these 9 strings to float64
    # 3. Reshape them into a (3, 3) matrix
    K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3) 
    
    # R (Quaternions are 4 elements, 1D)
    quat_elements = [float(x) for x in row["sensor2ego_rotation"].split(';')]
    R = Quaternion(quat_elements).rotation_matrix
    
    # T (Translation is 3 elements, 1D)
    T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)
    
    return CameraCalib(K=K, R=R, T=T)




# (All geometry functions like pixel_centers, backproject_to_ground, # @2
# xy_to_bev_indices, and splat_features_to_bev remain exactly the same) # @3
# [Functions from original script would be pasted here]
def pixel_centers(Hf: int, Wf: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]: # @2
    j = torch.arange(Wf, dtype=torch.float32) + 0.5
    i = torch.arange(Hf, dtype=torch.float32) + 0.5
    u = j * stride
    v = i * stride
    U, V = torch.meshgrid(u, v, indexing="xy")
    U = U.t().contiguous()
    V = V.t().contiguous()
    return U, V

def backproject_to_ground(U: torch.Tensor, V: torch.Tensor, calib: CameraCalib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # @2
    Hf, Wf = U.shape
    uv1 = np.stack([U.reshape(-1).numpy(), V.reshape(-1).numpy(), np.ones(Hf * Wf, dtype=np.float32)], axis=0)
    K_inv = np.linalg.inv(calib.K)
    rays_cam = K_inv @ uv1
    rays_cam[1:3, :] *= -1
    rays_ego = calib.R @ rays_cam
    dir_z = rays_ego[2, :]
    eps = 1e-6
    valid_dir = np.abs(dir_z) > eps
    cam_z = calib.T[2]
    s = np.zeros_like(dir_z, dtype=np.float64)
    s[valid_dir] = -cam_z / dir_z[valid_dir]
    mask_valid = valid_dir & (s > 0.0)
    Xg = calib.T[0] + s * rays_ego[0, :]
    Yg = calib.T[1] + s * rays_ego[1, :]
    return Xg, Yg, mask_valid

def xy_to_bev_indices(Xg: np.ndarray, Yg: np.ndarray, valid: np.ndarray, spec: BEVSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # @2 @3
    ix = ((Xg - spec.x_min) / spec.cell_x).astype(np.int64)
    iy = ((Yg - spec.y_min) / spec.cell_y).astype(np.int64)
    in_bounds = ( (ix >= 0) & (ix < spec.H) & (iy >= 0) & (iy < spec.W) )
    keep = valid & in_bounds
    return ix, iy, keep

def splat_features_to_bev(cam_feat: torch.Tensor, ix: np.ndarray, iy: np.ndarray, keep: np.ndarray, spec: BEVSpec) -> torch.Tensor: # @3
    B, C, Hf, Wf = cam_feat.shape
    N = Hf * Wf
    bev_grid = torch.zeros((B, C, spec.H, spec.W), dtype=cam_feat.dtype)
    if keep.sum() == 0:
        return bev_grid
    feat_flat = cam_feat.reshape(B, C, N)
    ix_v = torch.from_numpy(ix[keep])
    iy_v = torch.from_numpy(iy[keep])
    lin_v = (ix_v * spec.W + iy_v).long()
    bev_flat = bev_grid.view(B, C, spec.H * spec.W)
    mask_idx = torch.from_numpy(np.nonzero(keep)[0]).long()
    src = feat_flat.index_select(dim=2, index=mask_idx)
    idx = lin_v.unsqueeze(0).expand(C, -1)
    for b in range(B):
        bev_flat[b].scatter_add_(dim=1, index=idx, src=src[b])
    return bev_grid

# ---------------------------------- Main ------------------------------------ #
def run_pipeline(
    paths: NuScenesPaths,
    metadata_df: pd.DataFrame,
    target_sample_token: str,
    camera_name: str = "CAM_FRONT",
    fpn_level: int = 0,
    stride_p2: int = 8,
    bev_spec: Optional[BEVSpec] = None,
) -> Dict[str, Any]:
    """Full end-to-end pipeline using pre-processed metadata."""
    bev_spec = bev_spec or BEVSpec()

    # --- Fast lookup from the DataFrame ---
    query = f"sample_token == '{target_sample_token}' and camera_name == '{camera_name}'"
    results = metadata_df.query(query)
    if results.empty:
        raise KeyError(f"No entry found for sample_token='{target_sample_token}' and camera='{camera_name}'")
    
    # Use the first row found
    sample_info = results.iloc[0]

    # --- Calibration ---
    calib = extract_calibration_from_row(sample_info)

    # --- Feature map ---
    feature_filename = f"{target_sample_token}_lvl{fpn_level}.pt"
    feature_path = os.path.join(paths.camera_feat_dir, feature_filename)
    feat_blob = torch_load(feature_path)
    feat = feat_blob["feat"].unsqueeze(0) if feat_blob["feat"].ndim == 3 else feat_blob["feat"]
    B, C, Hf, Wf = feat.shape

    # --- Pixel centers & backproject ---
    U, V = pixel_centers(Hf, Wf, stride=stride_p2) # @2
    Xg, Yg, valid = backproject_to_ground(U, V, calib) # @2

    # --- Discretize to BEV & splat ---
    ix, iy, keep = xy_to_bev_indices(Xg, Yg, valid, bev_spec) # @2 @3
    bev = splat_features_to_bev(feat, ix, iy, keep, bev_spec) # @3

    # --- Debug prints (optional) ---
    print(f"Found sample info for {camera_name} via direct lookup.") # @3 @4
    print("Camera Intrinsic (K):\n", calib.K) # @3 @4
    print("Camera->ego rotation (R):\n", calib.R) # @3 @4
    print("Camera->ego translation (T):\n", calib.T) # @3 @4
    print(f"Input feat shape: {tuple(feat.shape)}") # @4
    print(f"BEV grid: {bev_spec.H} x {bev_spec.W}") # @4
    print("Valid projected pixels:", int(keep.sum())) # @4
    print("BEV tensor shape:", tuple(bev.shape)) # @4

    return {"bev": bev, "calib": calib}

def main():
    # (The argparse and main execution block would be similar, but point to the new CSV) @4 @5
    p = argparse.ArgumentParser(description="nuScenes CAM->BEV projection (Refactored)")
    # validation args
    # p.add_argument("--metadata_csv", type=str, default="nusc_val_metadata.csv")
    # p.add_argument("--camera_feat_dir", type=str, default="/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT")
    # training args    
    p.add_argument("--metadata_csv", type=str, default="nusc_val_metadata.csv")
    p.add_argument("--camera_feat_dir", type=str, default="/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT")

    p.add_argument("--sample_token", type=str, default="fd8420396768425eabec9bdddf7e64b6")
    p.add_argument("--camera_name", type=str, default="CAM_FRONT")
    p.add_argument("--fpn_level", type=int, default=0)
    p.add_argument("--stride", type=int, default=4)
    # BEV spec args...
    args = p.parse_args()
    # print(vars(args))
    paths = NuScenesPaths(
        metadata_csv_path=args.metadata_csv,
        camera_feat_dir=args.camera_feat_dir,
    )
    bev_spec = BEVSpec() # Simplified for brevity

    # Load metadata ONCE
    metadata_df = pd.read_csv(paths.metadata_csv_path)
    print(f"Loaded metadata with {len(metadata_df)} records.")

    run_pipeline(
        paths=paths,
        metadata_df=metadata_df,
        target_sample_token=args.sample_token,
        camera_name=args.camera_name,
        fpn_level=args.fpn_level,
        stride_p2=args.stride,
        bev_spec=bev_spec,
    )

if __name__ == "__main__":
    main()





















