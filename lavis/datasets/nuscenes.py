# import os
# import glob
# import torch
# import torch.nn.functional as F
# from torch.utils.data import ConcatDataset
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pyquaternion import Quaternion

# ########################################
# # Geometry + Projection Utilities
# ########################################

# class BEVSpec:
#     """
#     Defines the canonical BEV grid we want to fuse both camera and LiDAR into.
#     This matches your run_pipeline BEVSpec.
#     """
#     def __init__(self,
#                  x_min=0.0, x_max=50.0, cell_x=0.4,
#                  y_min=-25.0, y_max=25.0, cell_y=0.25):
#         self.x_min = x_min
#         self.x_max = x_max
#         self.cell_x = cell_x
#         self.y_min = y_min
#         self.y_max = y_max
#         self.cell_y = cell_y

#     @property
#     def H(self):
#         # forward/back (x direction)
#         return int(np.ceil((self.x_max - self.x_min) / self.cell_x))

#     @property
#     def W(self):
#         # lateral (y direction)
#         return int(np.ceil((self.y_max - self.y_min) / self.cell_y))


# class CameraCalib:
#     """
#     Camera intrinsics + extrinsics (sensor->ego).
#     """
#     def __init__(self, K, R, T):
#         self.K = K  # (3,3)
#         self.R = R  # (3,3)
#         self.T = T  # (3,)


# def extract_calibration_from_row(row: pd.Series) -> CameraCalib:
#     """
#     Given one row from metadata_df (with columns:
#       cam_intrinsic, sensor2ego_rotation, sensor2ego_translation),
#     build intrinsics/extrinsics.
#     """
#     # Intrinsics K: 9 semicolon-separated floats
#     K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3)

#     # Rotation as quaternion [w,x,y,z]
#     quat_elements = [float(x) for x in row["sensor2ego_rotation"].split(';')]
#     R = Quaternion(quat_elements).rotation_matrix

#     # Translation: 3 floats
#     T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)

#     return CameraCalib(K=K, R=R, T=T)


# def pixel_centers(Hf: int, Wf: int, stride: int):
#     """
#     Compute pixel centers in the original image coordinates
#     for each feature location on an FPN level with given stride.
#     Returns U,V each [Hf, Wf] in "full-res pixel space".
#     """
#     j = torch.arange(Wf, dtype=torch.float32) + 0.5  # width index
#     i = torch.arange(Hf, dtype=torch.float32) + 0.5  # height index
#     u = j * stride
#     v = i * stride
#     U, V = torch.meshgrid(u, v, indexing="xy")
#     U = U.t().contiguous()
#     V = V.t().contiguous()
#     return U, V


# def backproject_to_ground(U, V, calib: CameraCalib):
#     """
#     Lift each (u,v,1) pixel ray from camera frame into ego frame,
#     then intersect with z=0 ground plane. Returns Xg,Yg in ego coords
#     plus a validity mask.
#     """
#     Hf, Wf = U.shape
#     uv1 = np.stack(
#         [
#             U.reshape(-1).numpy(),
#             V.reshape(-1).numpy(),
#             np.ones(Hf * Wf, dtype=np.float32),
#         ],
#         axis=0,
#     )  # [3, N]

#     K_inv = np.linalg.inv(calib.K)
#     rays_cam = K_inv @ uv1  # camera frame rays

#     # Convert coord convention if needed
#     rays_cam[1:3, :] *= -1

#     rays_ego = calib.R @ rays_cam  # ego frame rays

#     dir_z = rays_ego[2, :]
#     eps = 1e-6
#     valid_dir = np.abs(dir_z) > eps

#     cam_z = calib.T[2]
#     s = np.zeros_like(dir_z, dtype=np.float64)
#     s[valid_dir] = -cam_z / dir_z[valid_dir]  # scale until z=0 ground
#     mask_valid = valid_dir & (s > 0.0)

#     Xg = calib.T[0] + s * rays_ego[0, :]
#     Yg = calib.T[1] + s * rays_ego[1, :]
#     return Xg, Yg, mask_valid


# def xy_to_bev_indices(Xg, Yg, valid, spec: BEVSpec):
#     """
#     Take ego-frame (Xg,Yg) (meters) and convert to integer BEV grid indices (ix,iy).
#     We keep only those that land in bounds.
#     """
#     ix = ((Xg - spec.x_min) / spec.cell_x).astype(np.int64)
#     iy = ((Yg - spec.y_min) / spec.cell_y).astype(np.int64)

#     in_bounds = (
#         (ix >= 0) & (ix < spec.H) &
#         (iy >= 0) & (iy < spec.W)
#     )
#     keep = valid & in_bounds
#     return ix, iy, keep


# def splat_features_to_bev(cam_feat: torch.Tensor,
#                           ix: np.ndarray,
#                           iy: np.ndarray,
#                           keep: np.ndarray,
#                           spec: BEVSpec) -> torch.Tensor:
#     """
#     Scatter-add (splat) per-pixel camera features into a BEV grid.

#     cam_feat: [B, C, Hf, Wf]
#     Returns: [B, C, spec.H, spec.W]
#     """
#     B, C, Hf, Wf = cam_feat.shape
#     N = Hf * Wf

#     bev_grid = torch.zeros((B, C, spec.H, spec.W), dtype=cam_feat.dtype)

#     if keep.sum() == 0:
#         return bev_grid

#     feat_flat = cam_feat.reshape(B, C, N)
#     ix_v = torch.from_numpy(ix[keep])
#     iy_v = torch.from_numpy(iy[keep])
#     lin_v = (ix_v * spec.W + iy_v).long()  # linear index in BEV plane

#     bev_flat = bev_grid.view(B, C, spec.H * spec.W)

#     mask_idx = torch.from_numpy(np.nonzero(keep)[0]).long()

#     # gather the valid pixel features
#     src = feat_flat.index_select(dim=2, index=mask_idx)  # [B,C,#keep]

#     # scatter-add by linear index
#     idx = lin_v.unsqueeze(0).expand(C, -1)  # [C,#keep]
#     for b in range(B):
#         bev_flat[b].scatter_add_(dim=1, index=idx, src=src[b])

#     return bev_grid  # [B,C,H_bev,W_bev]


# def resample_lidar_to_cam_bev(lidar_bev_feat: torch.Tensor,
#                               pc_range,
#                               target_spec: BEVSpec):
#     """
#     Take LiDAR BEV features that live on a different grid/range and resample
#     them to the camera BEVSpec grid using F.grid_sample, like in your snippet.

#     lidar_bev_feat: [B, C, H_lidar, W_lidar]
#       where that grid covers pc_range = [x_min_l, y_min_l, z_min_l, x_max_l, y_max_l, z_max_l]

#     target_spec: BEVSpec that defines (x_min,x_max,y_min,y_max) and output H,W
#     """
#     B, C, Hs, Ws = lidar_bev_feat.shape

#     x_min_l, y_min_l, _, x_max_l, y_max_l, _ = pc_range

#     # Build target BEV world sampling locations
#     H_cam = target_spec.H
#     W_cam = target_spec.W

#     cell_x = (target_spec.x_max - target_spec.x_min) / H_cam
#     cell_y = (target_spec.y_max - target_spec.y_min) / W_cam

#     xs = target_spec.x_min + (torch.arange(H_cam, dtype=torch.float32) + 0.5) * cell_x
#     ys = target_spec.y_min + (torch.arange(W_cam, dtype=torch.float32) + 0.5) * cell_y
#     Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')  # [H_cam, W_cam]

#     # Map world -> normalized lidar BEV coords in [-1, 1]
#     grid_x = 2.0 * (Yg - y_min_l) / (y_max_l - y_min_l) - 1.0  # corresponds to W axis
#     grid_y = 2.0 * (Xg - x_min_l) / (x_max_l - x_min_l) - 1.0  # corresponds to H axis

#     grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)   # [1, H_cam, W_cam, 2]

#     # Bilinear resample
#     lidar_on_cam = F.grid_sample(
#         lidar_bev_feat,
#         grid,
#         mode='bilinear',
#         padding_mode='zeros',
#         align_corners=True
#     )  # [B, C, H_cam, W_cam]

#     return lidar_on_cam


# ########################################
# # Dataset Classes
# ########################################

# class TorchCameraBEVDataset:
#     """
#     For each sample token, load camera FPN features and metadata,
#     lift-splat them into the canonical BEV grid (BEVSpec).
#     """
#     def __init__(
#         self,
#         sample_tokens,
#         metadata_df: pd.DataFrame,
#         camera_feat_root: str,
#         camera_name: str,
#         fpn_level: int,
#         stride: int,
#         bev_spec: BEVSpec,
#     ):
#         self.sample_tokens = sample_tokens
#         self.metadata_df = metadata_df
#         self.camera_feat_root = camera_feat_root
#         self.camera_name = camera_name
#         self.fpn_level = fpn_level
#         self.stride = stride
#         self.bev_spec = bev_spec

#     def __len__(self):
#         return len(self.sample_tokens)

#     def _load_camera_feat(self, token: str):
#         feat_filename = f"{token}_lvl{self.fpn_level}.pt"
#         feat_path = os.path.join(self.camera_feat_root, feat_filename)
#         blob = torch.load(feat_path, map_location="cpu")
#         cam_feat = blob["feat"]
#         if cam_feat.ndim == 3:
#             cam_feat = cam_feat.unsqueeze(0)  # [1,C,Hf,Wf]
#         return cam_feat  # [B,C,Hf,Wf]

#     def _lookup_calib(self, token: str):
#         rows = self.metadata_df.query(
#             f"sample_token == '{token}' and camera_name == '{self.camera_name}'"
#         )
#         if rows.empty:
#             raise KeyError(
#                 f"No metadata for token={token}, camera={self.camera_name}"
#             )
#         row = rows.iloc[0]
#         return extract_calibration_from_row(row)

#     def __getitem__(self, idx):
#         token = self.sample_tokens[idx]

#         # 1. Load feature map
#         cam_feat = self._load_camera_feat(token)  # [1,C,Hf,Wf]
#         B, C, Hf, Wf = cam_feat.shape

#         # 2. Load calibration for that camera frame
#         calib = self._lookup_calib(token)

#         # 3. Get pixel centers in original image space for each cam_feat cell
#         U, V = pixel_centers(Hf, Wf, stride=self.stride)

#         # 4. Raycast each FPN cell to ground plane
#         Xg, Yg, valid = backproject_to_ground(U, V, calib)

#         # 5. Discretize to BEV indices
#         ix, iy, keep = xy_to_bev_indices(Xg, Yg, valid, self.bev_spec)

#         # 6. Splat into BEVSpec grid
#         cam_bev = splat_features_to_bev(cam_feat, ix, iy, keep, self.bev_spec)
#         # cam_bev: [1, C, bev_spec.H, bev_spec.W]

#         return {
#             "cam_bev": cam_bev.squeeze(0),  # [C,H_bev,W_bev]
#             "image_id": token,
#         }

#     def collater(self, samples):
#         samples = [s for s in samples if s is not None]
#         if len(samples) == 0:
#             return {}
#         cam_bev_batch = torch.stack([s["cam_bev"] for s in samples])  # [B,C,H,W]
#         image_ids = [s["image_id"] for s in samples]
#         return {
#             "cam_bev": cam_bev_batch,
#             "image_id": image_ids,
#         }


# class TorchLidarBEVDataset:
#     """
#     For each sample token, load LiDAR BEV features and resample them
#     into the SAME BEVSpec grid as camera.
#     """
#     def __init__(
#         self,
#         sample_tokens,
#         lidar_root: str,
#         pc_range,            # e.g. [-54,-54,-5,54,54,3]
#         bev_spec: BEVSpec,
#     ):
#         self.sample_tokens = sample_tokens
#         self.lidar_root = lidar_root
#         self.pc_range = pc_range
#         self.bev_spec = bev_spec

#     def __len__(self):
#         return len(self.sample_tokens)

#     def _load_lidar_feat(self, token: str):
#         feat_path = os.path.join(self.lidar_root, f"{token}.pt")
#         blob = torch.load(feat_path, map_location="cpu")
#         lidar_feat = blob["feat"]
#         if lidar_feat.ndim == 3:
#             lidar_feat = lidar_feat.unsqueeze(0)  # [1,C,Hs,Ws]
#         return lidar_feat  # [1,C,Hs,Ws]

#     def __getitem__(self, idx):
#         token = self.sample_tokens[idx]

#         # 1. Load LiDAR BEV grid from disk
#         lidar_feat = self._load_lidar_feat(token)  # [1,C,Hs,Ws]

#         # 2. Resample LiDAR BEV to camera BEVSpec
#         lidar_on_cam = resample_lidar_to_cam_bev(
#             lidar_feat,
#             pc_range=self.pc_range,
#             target_spec=self.bev_spec
#         )  # [1,C,H_cam,W_cam]

#         return {
#             "lidar_bev": lidar_on_cam.squeeze(0),  # [C,H_cam,W_cam]
#             "image_id": token,
#         }


# ########################################
# # Dataset Wrapper that returns paired BEV tensors
# ########################################

# class WrappedConcatBEVDataset(ConcatDataset):
#     """
#     Combine camera-projected BEV and LiDAR-projected BEV for the SAME token.
#     Both should now already share [H_bev, W_bev].
#     """
#     def __init__(self, cam_dataset: TorchCameraBEVDataset, lidar_dataset: TorchLidarBEVDataset):
#         assert len(cam_dataset) == len(lidar_dataset), \
#             "Camera and LiDAR token sets must align 1:1 after matching."
#         super().__init__([cam_dataset, lidar_dataset])
#         self.cam_dataset = cam_dataset
#         self.lidar_dataset = lidar_dataset

#     def __len__(self):
#         # enforce paired length
#         return min(len(self.cam_dataset), len(self.lidar_dataset))

#     def __getitem__(self, idx):
#         cam_sample = self.cam_dataset[idx]
#         lidar_sample = self.lidar_dataset[idx]

#         if cam_sample is None or lidar_sample is None:
#             return None

#         # sanity: assert same token
#         # (they both expose "image_id")
#         if cam_sample["image_id"] != lidar_sample["image_id"]:
#             # If this ever triggers, your matching logic is off.
#             # We return None so collater can skip it.
#             return None

#         return {
#             "cam_bev": cam_sample["cam_bev"],       # [C_cam,H,W]
#             "lidar_bev": lidar_sample["lidar_bev"], # [C_lidar,H,W]
#             "image_id": cam_sample["image_id"],
#             "label": 1,
#         }

#     def collater(self, samples):
#         samples = [s for s in samples if s is not None]
#         if len(samples) == 0:
#             return {}

#         cam_bev_batch   = torch.stack([s["cam_bev"]    for s in samples])   # [B,Cc,H,W]
#         lidar_bev_batch = torch.stack([s["lidar_bev"] for s in samples])    # [B,Cl,H,W]
#         image_ids       = [s["image_id"] for s in samples]
#         labels          = torch.tensor([s["label"] for s in samples])

#         return {
#             "cam_bev": cam_bev_batch,
#             "lidar_bev": lidar_bev_batch,
#             "image_id": image_ids,
#             "label": labels,
#         }


# ########################################
# # Helpers to match tokens and build datasets
# ########################################

# def get_matched_tokens(camera_feat_root, lidar_root, fpn_level):
#     """
#     Match samples by token. For camera we expect {token}_lvl{fpn_level}.pt.
#     For lidar we expect {token}.pt.
#     Returns a sorted list of tokens that exist in BOTH.
#     """
#     cam_files = glob.glob(os.path.join(camera_feat_root, f"*lvl{fpn_level}.pt"))
#     lidar_files = glob.glob(os.path.join(lidar_root, "*.pt"))

#     cam_tokens = {
#         os.path.basename(f).replace(f"_lvl{fpn_level}.pt", ""): f
#         for f in cam_files
#     }
#     lidar_tokens = {
#         os.path.splitext(os.path.basename(f))[0]: f
#         for f in lidar_files
#     }

#     print(f" Found {len(cam_tokens)} camera feature files in {camera_feat_root}")
#     print(f" Found {len(lidar_tokens)} lidar  feature files in {lidar_root}")

#     matched = sorted(set(cam_tokens.keys()) & set(lidar_tokens.keys()))
#     print(f" Matched {len(matched)} tokens")

#     if len(matched) == 0:
#         print(" Example camera tokens:", list(cam_tokens.keys())[:5])
#         print(" Example lidar  tokens:", list(lidar_tokens.keys())[:5])
#         print(" Check filename consistency.")

#     return matched


# class NuScenesDatasetBuilder:
#     """
#     Builder that creates train/val/test datasets returning aligned BEV tensors.
#     Assumes cfg contains:
#       cfg.datasets_cfg["nuscenes"].build_info["camera"][split].storage
#       cfg.datasets_cfg["nuscenes"].build_info["lidar"][split].storage
#       cfg.datasets_cfg["nuscenes"].metadata_csv   (path to CSV)
#       cfg.run_cfg.{train_splits,valid_splits,test_splits}
#       cfg.bev / cfg.cam_projection params like camera_name, fpn_level, stride, pc_range, etc.
#     """
#     def __init__(self, cfg):
#         self.cfg = cfg

#     # def build_datasets(self):
#     #     build_info = self.cfg.datasets_cfg["nuscenes"].build_info
#     #     datasets = {}
        
#     #     # Load shared metadata once
#     #     metadata_csv_path = build_info["metadata_csv"]
        
#     #     metadata_csv_path = build_info.get(f"metadata_csv_{split}")
#     #     metadata_df = pd.read_csv(metadata_csv_path)

#     #     bev_cfg = build_info["bev"]
#     #     cam_proj = build_info["cam_proj"]
#     #     lidar_proj = build_info["lidar_proj"]
        

#     #     # bev_cfg = self.cfg.build_info.bev
#     #     bev_spec = BEVSpec(
#     #         x_min=bev_cfg.x_min,
#     #         x_max=bev_cfg.x_max,
#     #         cell_x=bev_cfg.cell_x,
#     #         y_min=bev_cfg.y_min,
#     #         y_max=bev_cfg.y_max,
#     #         cell_y=bev_cfg.cell_y,
#     #     )


#     #     for split in ["train", "val", "test"]:
#     #         if split in (
#     #             self.cfg.run_cfg.train_splits
#     #             + self.cfg.run_cfg.valid_splits
#     #             + self.cfg.run_cfg.test_splits
#     #         ):
#     #             camera_root = build_info["camera"][split].storage
#     #             lidar_root  = build_info["lidar"][split].storage

#     #             tokens = get_matched_tokens(
#     #                 camera_feat_root=camera_root,
#     #                 lidar_root=lidar_root,
#     #                 fpn_level= cam_proj.fpn_level,
#     #             )

#     #             cam_ds = TorchCameraBEVDataset(
#     #                 sample_tokens=tokens,
#     #                 metadata_df=metadata_df,
#     #                 camera_feat_root=camera_root,
#     #                 camera_name=cam_proj.camera_name,
#     #                 fpn_level=cam_proj.fpn_level,
#     #                 stride=cam_proj.stride,
#     #                 bev_spec=bev_spec,
#     #             )

#     #             lidar_ds = TorchLidarBEVDataset(
#     #                 sample_tokens=tokens,
#     #                 lidar_root=lidar_root,
#     #                 pc_range=lidar_proj.pc_range,  # [-54,-54,-5,54,54,3]
#     #                 bev_spec=bev_spec,
#     #             )

#     #             datasets[split] = WrappedConcatBEVDataset(cam_ds, lidar_ds)

#     #     return datasets


#     def build_datasets(self):
#         build_info = self.cfg.datasets_cfg["nuscenes"].build_info
#         datasets = {}

#         bev_cfg   = build_info["bev"]
#         cam_proj  = build_info["cam_proj"]
#         lidar_proj= build_info["lidar_proj"]

#         # Canonical BEV grid spec shared across splits
#         bev_spec = BEVSpec(
#             x_min = bev_cfg.x_min,
#             x_max = bev_cfg.x_max,
#             cell_x= bev_cfg.cell_x,
#             y_min = bev_cfg.y_min,
#             y_max = bev_cfg.y_max,
#             cell_y= bev_cfg.cell_y,
#         )

#         # Loop over each split we care about
#         for split in ["train", "val", "test"]:
#             # Only build splits that are actually requested by run_cfg
#             if split not in (
#                 self.cfg.run_cfg.train_splits
#                 + self.cfg.run_cfg.valid_splits
#                 + self.cfg.run_cfg.test_splits
#             ):
#                 continue

#             # -------------------------------------------------
#             # 1. Get per-split paths
#             # -------------------------------------------------
#             camera_root = build_info["camera"][split].storage
#             lidar_root  = build_info["lidar"][split].storage

#             # metadata_csv_train / metadata_csv_val / metadata_csv_test
#             metadata_csv_key = f"metadata_csv_{split}"
#             if metadata_csv_key in build_info:
#                 metadata_csv_path = build_info[metadata_csv_key]
#             else:
#                 # fallback if you didn't define metadata_csv_test etc.
#                 metadata_csv_path = build_info["metadata_csv"]

#             # Load the right calibration table for THIS split
#             metadata_df = pd.read_csv(metadata_csv_path)

#             # -------------------------------------------------
#             # 2. Match tokens that exist in BOTH camera+lidar
#             # -------------------------------------------------
#             tokens = get_matched_tokens(
#                 camera_feat_root=camera_root,
#                 lidar_root=lidar_root,
#                 fpn_level=cam_proj.fpn_level,
#             )

#             # -------------------------------------------------
#             # 3. Build per-modality datasets
#             # -------------------------------------------------
#             cam_ds = TorchCameraBEVDataset(
#                 sample_tokens     = tokens,
#                 metadata_df       = metadata_df,
#                 camera_feat_root  = camera_root,
#                 camera_name       = cam_proj.camera_name,
#                 fpn_level         = cam_proj.fpn_level,
#                 stride            = cam_proj.stride,
#                 bev_spec          = bev_spec,
#             )

#             lidar_ds = TorchLidarBEVDataset(
#                 sample_tokens = tokens,
#                 lidar_root    = lidar_root,
#                 pc_range      = lidar_proj.pc_range,  # e.g. [-54,-54,-5,54,54,3]
#                 bev_spec      = bev_spec,
#             )

#             # -------------------------------------------------
#             # 4. Wrap them together (cam & lidar aligned BEVs)
#             # -------------------------------------------------
#             datasets[split] = WrappedConcatBEVDataset(cam_ds, lidar_ds)

#         return datasets


# ########################################
# # Debug / Visualization
# ########################################

# def visualize_sample(sample, title_prefix=""):
#     """
#     Visualize BEV-aligned camera vs BEV-aligned LiDAR.
#     Both are now [C,H,W] in the same grid.
#     We'll just average channels for display.
#     """
#     cam_bev   = sample["cam_bev"].mean(dim=0).cpu().numpy()     # [H,W]
#     lidar_bev = sample["lidar_bev"].mean(dim=0).cpu().numpy()   # [H,W]

#     fig, axes = plt.subplots(1, 2, figsize=(12,5))

#     axes[0].imshow(cam_bev, cmap='viridis', origin='lower')
#     axes[0].set_title(f"{title_prefix}Camera→BEV")
#     axes[0].axis('off')

#     axes[1].imshow(lidar_bev, cmap='viridis', origin='lower')
#     axes[1].set_title(f"{title_prefix}LiDAR→BEV (resampled)")
#     axes[1].axis('off')

#     plt.tight_layout()
#     plt.show()


# ########################################
# # Standalone smoke test (like your old main)
# ########################################

# def main():
#     # ---- Paths (example; update to your actual dirs) ----
#     camera_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT"
#     lidar_dir  = "/home/draiman/Desktop/Datasets_nuscenes/validation/lidar"
#     metadata_csv = "/home/draiman/Desktop/Datasets_nuscenes/nuscenes_metadata.csv"

#     # ---- Projection / grid params ----
#     camera_name = "CAM_FRONT"
#     fpn_level   = 0
#     stride      = 8     # stride_p2 you used when computing pixel_centers
#     pc_range    = [-54, -54, -5, 54, 54, 3]  # LiDAR original range

#     bev_spec = BEVSpec(
#         x_min=0.0,   x_max=50.0,  cell_x=0.4,
#         y_min=-25.0, y_max=25.0,  cell_y=0.25,
#     )

#     # ---- Prep token list ----
#     tokens = get_matched_tokens(camera_dir, lidar_dir, fpn_level=fpn_level)

#     # ---- Load metadata once ----
#     metadata_df = pd.read_csv(metadata_csv)

#     cam_ds = TorchCameraBEVDataset(
#         sample_tokens=tokens,
#         metadata_df=metadata_df,
#         camera_feat_root=camera_dir,
#         camera_name=camera_name,
#         fpn_level=fpn_level,
#         stride=stride,
#         bev_spec=bev_spec,
#     )

#     lidar_ds = TorchLidarBEVDataset(
#         sample_tokens=tokens,
#         lidar_root=lidar_dir,
#         pc_range=pc_range,
#         bev_spec=bev_spec,
#     )

#     dataset = WrappedConcatBEVDataset(cam_ds, lidar_ds)

#     # Inspect a few samples
#     for i in range(min(3, len(dataset))):
#         sample = dataset[i]
#         print(f"\n📦 Sample {i}:")
#         print(f" - cam_bev shape:   {tuple(sample['cam_bev'].shape)}")
#         print(f" - lidar_bev shape: {tuple(sample['lidar_bev'].shape)}")
#         print(f" - Token/Image ID:  {sample['image_id']}")
#         print(f" - Label:           {sample['label']}")

#         visualize_sample(sample, title_prefix=f"Sample {i}: ")


# if __name__ == "__main__":
#     main()
















# import os
# import numpy as np
# import pandas as pd
# import torch
# from pyquaternion import Quaternion

# ########################################
# # Geometry + Projection Utilities
# ########################################

# class BEVSpec:
#     """
#     Defines the canonical BEV grid for fusing camera features.
#     """
#     def __init__(self, x_min=0.0, x_max=50.0, cell_x=0.4,
#                  y_min=-25.0, y_max=25.0, cell_y=0.25):
#         self.x_min = x_min
#         self.x_max = x_max
#         self.cell_x = cell_x
#         self.y_min = y_min
#         self.y_max = y_max
#         self.cell_y = cell_y

#     @property
#     def H(self):
#         return int(np.ceil((self.x_max - self.x_min) / self.cell_x))

#     @property
#     def W(self):
#         return int(np.ceil((self.y_max - self.y_min) / self.cell_y))

# class CameraCalib:
#     def __init__(self, K, R, T):
#         self.K = K
#         self.R = R
#         self.T = T

# def extract_calibration_from_row(row: pd.Series) -> CameraCalib:
#     K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3)
#     R = Quaternion([float(x) for x in row["sensor2ego_rotation"].split(';')]).rotation_matrix
#     T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)
#     return CameraCalib(K=K, R=R, T=T)

# def main():
#     # ----- Paths -----
#     metadata_csv = "/home/draiman/Desktop/Datasets_nuscenes/nuscenes_metadata.csv"
#     camera_feat_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT"
#     camera_name = "CAM_FRONT"
#     fpn_level = 0
    
#     # ----- Load metadata -----
#     metadata_df = pd.read_csv(metadata_csv)
#     print(f"Loaded {len(metadata_df)} metadata records.")

#     # ----- Prep token list -----
#     tokens = [os.path.basename(f).replace(f"_lvl{fpn_level}.pt", "") 
#               for f in os.listdir(camera_feat_dir) if f.endswith(f"_lvl{fpn_level}.pt")]

#     # ----- BEV grid spec -----
#     cell_x, cell_y = 0.4, 0.25
#     x_min, x_max, y_min, y_max = 0.0, 50.0, -25.0, 25.0
#     H_bev = int(np.ceil((x_max - x_min) / cell_x))
#     W_bev = int(np.ceil((y_max - y_min) / cell_y))
#     print(f"BEV grid: {H_bev} x {W_bev}")

#     bev_spec = BEVSpec(x_min=x_min, x_max=x_max, cell_x=cell_x,
#                        y_min=y_min, y_max=y_max, cell_y=cell_y)

#     for idx in range(min(3, len(tokens))):
#         token = tokens[idx]
#         print(f"\nSample {idx}:")

#         # -- Camera calibration row --
#         row = metadata_df.query(f"sample_token == '{token}' and camera_name == '{camera_name}'")
#         if row.empty:
#             raise RuntimeError("No matching camera record.")
#         row = row.iloc[0]

#         K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3)
#         R = Quaternion([float(x) for x in row["sensor2ego_rotation"].split(';')]).rotation_matrix
#         T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)
#         calib = CameraCalib(K=K, R=R, T=T)

#         # -- Load features --
#         feat_path = os.path.join(camera_feat_dir, f"{token}_lvl{fpn_level}.pt")
#         cam_blob = torch.load(feat_path, map_location='cpu')
#         cam_feat = cam_blob['feat']
#         if cam_feat.ndim == 3:
#             cam_feat = cam_feat.unsqueeze(0)
#         B, C, Hf, Wf = cam_feat.shape
#         print("Camera feature shape:", cam_feat.shape)

#         # -- Get true input image shape and compute stride --
#         if 'img_shape' in cam_blob:
#             H_img, W_img, _ = cam_blob['img_shape'][0]
#         else:
#             H_img = int(row['img_height']) if 'img_height' in row else 928
#             W_img = int(row['img_width']) if 'img_width' in row else 1600

#         print("Loaded img_shape:", H_img, W_img)
#         stride_u = W_img / Wf
#         stride_v = H_img / Hf
#         assert abs(stride_u - stride_v) < 1e-2, f"Stride is asymmetric ({stride_u} vs {stride_v}); check preprocessing pipeline."
#         stride = stride_u
#         print("Stride used:", stride)

#         # -- Compute pixel centers --
#         i = torch.arange(Hf, dtype=torch.float32) + 0.5
#         j = torch.arange(Wf, dtype=torch.float32) + 0.5
#         V, U = torch.meshgrid(i * stride, j * stride, indexing='ij')

#         # -- Backproject to ground (z=0) --
#         uv1 = np.stack([U.reshape(-1).numpy(), V.reshape(-1).numpy(), np.ones(Hf * Wf)], axis=0)
#         K_inv = np.linalg.inv(K)
#         rays_cam = K_inv @ uv1
#         rays_cam[1, :] *= -1.0
#         rays_ego = R @ rays_cam
#         dir_z = rays_ego[2, :]
#         cam_z = T[2]
#         s_ground = -cam_z / (dir_z + 1e-12)
#         Xg = T[0] + s_ground * rays_ego[0, :]
#         Yg = T[1] + s_ground * rays_ego[1, :]

#         # -- Discretize to BEV indices --
#         ix = ((Xg - x_min) / cell_x).astype(np.int64)
#         iy = ((Yg - y_min) / cell_y).astype(np.int64)
#         valid = (ix >= 0) & (ix < H_bev) & (iy >= 0) & (iy < W_bev) & (s_ground > 0)

#         # -- Scatter features to BEV --
#         bev_grid = torch.zeros((B, C, H_bev, W_bev), dtype=cam_feat.dtype)
#         cam_feat_flat = cam_feat.reshape(B, C, -1)
#         valid_idx = np.flatnonzero(valid)
#         ix_valid = torch.from_numpy(ix[valid]).long()
#         iy_valid = torch.from_numpy(iy[valid]).long()
#         cam_feat_valid = cam_feat_flat[:, :, valid_idx]
#         for b in range(B):
#             bev_flat = bev_grid[b].reshape(C, -1)
#             linear_idx = ix_valid * W_bev + iy_valid
#             bev_flat.scatter_add_(
#                 1,
#                 linear_idx.unsqueeze(0).expand(C, -1),
#                 cam_feat_valid[b]
#             )
#             bev_grid[b] = bev_flat.reshape(C, H_bev, W_bev)

#         print(f"[2D] Valid splats: {valid.sum()} / {Hf*Wf}")
#         print("BEV grid shape:", bev_grid.shape)
#         print(" - cam_bev shape:", tuple(bev_grid.squeeze(0).shape))
#         print(" - Token/Image ID:", token)

# if __name__ == "__main__":
#     main()









































































import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
from pyquaternion import Quaternion


class BEVSpec:
    """
    Defines the canonical BEV grid for fusion.
    """
    def __init__(self, x_min=0.0, x_max=50.0, cell_x=0.4, y_min=-25.0, y_max=25.0, cell_y=0.25):
        self.x_min = x_min
        self.x_max = x_max
        self.cell_x = cell_x
        self.y_min = y_min
        self.y_max = y_max
        self.cell_y = cell_y

    @property
    def H(self):
        return int(np.ceil((self.x_max - self.x_min) / self.cell_x))

    @property
    def W(self):
        return int(np.ceil((self.y_max - self.y_min) / self.cell_y))


class CameraCalib:
    def __init__(self, K, R, T):
        self.K = K
        self.R = R
        self.T = T

def extract_calibration_from_row(row: pd.Series) -> CameraCalib:
    K = np.array(row["cam_intrinsic"].split(';'), dtype=np.float64).reshape(3, 3)
    R = Quaternion([float(x) for x in row["sensor2ego_rotation"].split(';')]).rotation_matrix
    T = np.array(row["sensor2ego_translation"].split(';'), dtype=np.float64)
    return CameraCalib(K=K, R=R, T=T)


def get_matched_tokens(camera_feat_root, lidar_root, fpn_level):
    cam_files = glob.glob(os.path.join(camera_feat_root, f"*lvl{fpn_level}.pt"))
    lidar_files = glob.glob(os.path.join(lidar_root, "*.pt"))
    cam_tokens = {os.path.basename(f).replace(f"_lvl{fpn_level}.pt", ""): f for f in cam_files}
    lidar_tokens = {os.path.splitext(os.path.basename(f))[0]: f for f in lidar_files}
    matched = sorted(set(cam_tokens.keys()) & set(lidar_tokens.keys()))
    return matched


class TorchCameraBEVDataset:
    """
    Loads and projects FPN camera features into BEV grid using dynamic stride.
    """
    def __init__(self, sample_tokens, metadata_df: pd.DataFrame, camera_feat_root: str,
                 camera_name: str, fpn_level: int, bev_spec: BEVSpec):
        self.sample_tokens = sample_tokens
        self.metadata_df = metadata_df
        self.camera_feat_root = camera_feat_root
        self.camera_name = camera_name
        self.fpn_level = fpn_level
        self.bev_spec = bev_spec

    def __len__(self):
        return len(self.sample_tokens)

    def _load_camera_feat(self, token: str):
        feat_filename = f"{token}_lvl{self.fpn_level}.pt"
        feat_path = os.path.join(self.camera_feat_root, feat_filename)
        blob = torch.load(feat_path, map_location="cpu")
        cam_feat = blob["feat"]
        if cam_feat.ndim == 3:
            cam_feat = cam_feat.unsqueeze(0)
        # Get true image shape (for stride calculation)
        if 'img_shape' in blob:
            H_img, W_img, _ = blob['img_shape'][0]
        else:
            # fallback: take from metadata (or constants)
            H_img = int(self.metadata_df.query(
                f"sample_token == '{token}' and camera_name == '{self.camera_name}'"
            ).iloc[0].get('img_height', 928))
            W_img = int(self.metadata_df.query(
                f"sample_token == '{token}' and camera_name == '{self.camera_name}'"
            ).iloc[0].get('img_width', 1600))
        Hf, Wf = cam_feat.shape[-2], cam_feat.shape[-1]
        stride_u = W_img / Wf
        stride_v = H_img / Hf
        assert abs(stride_u - stride_v) < 1e-2, f"Stride asymmetric: {stride_u} vs {stride_v}"
        self._stride = stride_u
        self._img_shape = (H_img, W_img)
        return cam_feat

    def _lookup_calib(self, token: str):
        row = self.metadata_df.query(f"sample_token == '{token}' and camera_name == '{self.camera_name}'").iloc[0]
        return extract_calibration_from_row(row)

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        cam_feat = self._load_camera_feat(token)  # [1,C,Hf,Wf]
        B, C, Hf, Wf = cam_feat.shape
        stride = self._stride

        calib = self._lookup_calib(token)
        # Compute pixel centers (dynamic stride)
        i = torch.arange(Hf, dtype=torch.float32) + 0.5
        j = torch.arange(Wf, dtype=torch.float32) + 0.5
        V, U = torch.meshgrid(i * stride, j * stride, indexing='ij')
        uv1 = np.stack([U.reshape(-1).numpy(), V.reshape(-1).numpy(), np.ones(Hf * Wf)], axis=0)
        K_inv = np.linalg.inv(calib.K)
        rays_cam = K_inv @ uv1
        rays_cam[1, :] *= -1.0
        rays_ego = calib.R @ rays_cam
        dir_z = rays_ego[2, :]
        cam_z = calib.T[2]
        s_ground = -cam_z / (dir_z + 1e-12)
        Xg = calib.T[0] + s_ground * rays_ego[0, :]
        Yg = calib.T[1] + s_ground * rays_ego[1, :]
        ix = ((Xg - self.bev_spec.x_min) / self.bev_spec.cell_x).astype(np.int64)
        iy = ((Yg - self.bev_spec.y_min) / self.bev_spec.cell_y).astype(np.int64)
        valid = (ix >= 0) & (ix < self.bev_spec.H) & (iy >= 0) & (iy < self.bev_spec.W) & (s_ground > 0)

        bev_grid = torch.zeros((B, C, self.bev_spec.H, self.bev_spec.W), dtype=cam_feat.dtype)
        cam_feat_flat = cam_feat.reshape(B, C, -1)
        valid_idx = np.flatnonzero(valid)
        ix_valid = torch.from_numpy(ix[valid]).long()
        iy_valid = torch.from_numpy(iy[valid]).long()
        cam_feat_valid = cam_feat_flat[:, :, valid_idx]
        for b in range(B):
            bev_flat = bev_grid[b].reshape(C, -1)
            linear_idx = ix_valid * self.bev_spec.W + iy_valid
            bev_flat.scatter_add_(1, linear_idx.unsqueeze(0).expand(C, -1), cam_feat_valid[b])
            bev_grid[b] = bev_flat.reshape(C, self.bev_spec.H, self.bev_spec.W)
        
        return {
            "cam_bev": bev_grid.squeeze(0),
            "image_id": token,
        }


    # def __getitem__(self, idx):
    #     token = self.sample_tokens[idx]
    #     cam_feat = self._load_camera_feat(token)  # [1,C,Hf,Wf]
    #     B, C, Hf, Wf = cam_feat.shape
    #     stride = self._stride

    #     calib = self._lookup_calib(token)
    #     # Compute pixel centers (dynamic stride)
    #     i = torch.arange(Hf, dtype=torch.float32) + 0.5
    #     j = torch.arange(Wf, dtype=torch.float32) + 0.5
    #     V, U = torch.meshgrid(i * stride, j * stride, indexing='ij')
    #     uv1 = np.stack([U.reshape(-1).numpy(), V.reshape(-1).numpy(), np.ones(Hf * Wf)], axis=0)
    #     K_inv = np.linalg.inv(calib.K)
    #     rays_cam = K_inv @ uv1
    #     rays_cam[1, :] *= -1.0
    #     rays_ego = calib.R @ rays_cam
    #     dir_z = rays_ego[2, :]
    #     cam_z = calib.T[2]
    #     s_ground = -cam_z / (dir_z + 1e-12)
    #     Xg = calib.T[0] + s_ground * rays_ego[0, :]
    #     Yg = calib.T[1] + s_ground * rays_ego[1, :]
    #     ix = ((Xg - self.bev_spec.x_min) / self.bev_spec.cell_x).astype(np.int64)
    #     iy = ((Yg - self.bev_spec.y_min) / self.bev_spec.cell_y).astype(np.int64)
    #     valid = (ix >= 0) & (ix < self.bev_spec.H) & (iy >= 0) & (iy < self.bev_spec.W) & (s_ground > 0)

    #     # Print valid splats info
    #     print(f"[2D] Valid splats: {valid.sum()} / {Hf * Wf} for token {token}")

    #     bev_grid = torch.zeros((B, C, self.bev_spec.H, self.bev_spec.W), dtype=cam_feat.dtype)
    #     cam_feat_flat = cam_feat.reshape(B, C, -1)
    #     valid_idx = np.flatnonzero(valid)
    #     ix_valid = torch.from_numpy(ix[valid]).long()
    #     iy_valid = torch.from_numpy(iy[valid]).long()
    #     cam_feat_valid = cam_feat_flat[:, :, valid_idx]
    #     for b in range(B):
    #         bev_flat = bev_grid[b].reshape(C, -1)
    #         linear_idx = ix_valid * self.bev_spec.W + iy_valid
    #         bev_flat.scatter_add_(1, linear_idx.unsqueeze(0).expand(C, -1), cam_feat_valid[b])
    #         bev_grid[b] = bev_flat.reshape(C, self.bev_spec.H, self.bev_spec.W)

    #     return {
    #         "cam_bev": bev_grid.squeeze(0),
    #         "image_id": token,
    #     }



    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        cam_bev_batch = torch.stack([s["cam_bev"] for s in samples])
        image_ids = [s["image_id"] for s in samples]
        return {
            "cam_bev": cam_bev_batch,
            "image_id": image_ids,
        }


class TorchLidarBEVDataset:
    """
    Loads Lidar BEV features and resamples to the canonical BEV grid.
    """
    def __init__(self, sample_tokens, lidar_root: str, pc_range, bev_spec: BEVSpec):
        self.sample_tokens = sample_tokens
        self.lidar_root = lidar_root
        self.pc_range = pc_range
        self.bev_spec = bev_spec

    def __len__(self):
        return len(self.sample_tokens)

    def _load_lidar_feat(self, token: str):
        feat_path = os.path.join(self.lidar_root, f"{token}.pt")
        blob = torch.load(feat_path, map_location="cpu")
        lidar_feat = blob["feat"]
        if lidar_feat.ndim == 3:
            lidar_feat = lidar_feat.unsqueeze(0)
        return lidar_feat

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        lidar_feat = self._load_lidar_feat(token)
        # Resample to canonical BEV
        x_min_l, y_min_l, _, x_max_l, y_max_l, _ = self.pc_range
        H_cam = self.bev_spec.H
        W_cam = self.bev_spec.W
        cell_x = (self.bev_spec.x_max - self.bev_spec.x_min) / H_cam
        cell_y = (self.bev_spec.y_max - self.bev_spec.y_min) / W_cam
        xs = self.bev_spec.x_min + (torch.arange(H_cam, dtype=torch.float32) + 0.5) * cell_x
        ys = self.bev_spec.y_min + (torch.arange(W_cam, dtype=torch.float32) + 0.5) * cell_y
        Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')
        grid_x = 2.0 * (Yg - y_min_l) / (y_max_l - y_min_l) - 1.0
        grid_y = 2.0 * (Xg - x_min_l) / (x_max_l - x_min_l) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        lidar_on_cam = F.grid_sample(
            lidar_feat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        return {
            "lidar_bev": lidar_on_cam.squeeze(0),
            "image_id": token,
        }



class WrappedConcatBEVDataset(ConcatDataset):
    """
    Returns paired (camera, lidar) BEV tensors for each token.
    """
    def __init__(self, cam_dataset: TorchCameraBEVDataset, lidar_dataset: TorchLidarBEVDataset):
        assert len(cam_dataset) == len(lidar_dataset), "Camera and Lidar token sets must align 1:1."
        super().__init__([cam_dataset, lidar_dataset])
        self.cam_dataset = cam_dataset
        self.lidar_dataset = lidar_dataset

    def __len__(self):
        return min(len(self.cam_dataset), len(self.lidar_dataset))

    def __getitem__(self, idx):
        cam_sample = self.cam_dataset[idx]
        lidar_sample = self.lidar_dataset[idx]
        if cam_sample is None or lidar_sample is None:
            return None
        if cam_sample["image_id"] != lidar_sample["image_id"]:
            return None
        return {
            "cam_bev": cam_sample["cam_bev"],
            "lidar_bev": lidar_sample["lidar_bev"],
            "image_id": cam_sample["image_id"],
            "label": 1,
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        cam_bev_batch = torch.stack([s["cam_bev"] for s in samples])
        lidar_bev_batch = torch.stack([s["lidar_bev"] for s in samples])
        image_ids = [s["image_id"] for s in samples]
        labels = torch.tensor([s["label"] for s in samples])
        return {
            "cam_bev": cam_bev_batch,
            "lidar_bev": lidar_bev_batch,
            "image_id": image_ids,
            "label": labels,
        }


class NuScenesDatasetBuilder:
    """
    Builds train/val/test datasets aligned to BEV tensors.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def build_datasets(self):
        build_info = self.cfg.datasets_cfg["nuscenes"].build_info
        datasets = {}
        bev_cfg = build_info["bev"]
        cam_proj = build_info["cam_proj"]
        lidar_proj = build_info["lidar_proj"]
        bev_spec = BEVSpec(
            x_min=bev_cfg.x_min,
            x_max=bev_cfg.x_max,
            cell_x=bev_cfg.cell_x,
            y_min=bev_cfg.y_min,
            y_max=bev_cfg.y_max,
            cell_y=bev_cfg.cell_y,
        )
        for split in ["train", "val", "test"]:
            if split not in (self.cfg.run_cfg.train_splits
                            + self.cfg.run_cfg.valid_splits
                            + self.cfg.run_cfg.test_splits):
                continue
            camera_root = build_info["camera"][split].storage
            lidar_root = build_info["lidar"][split].storage
            metadata_csv_key = f"metadata_csv_{split}"
            if metadata_csv_key in build_info:
                metadata_csv_path = build_info[metadata_csv_key]
            else:
                metadata_csv_path = build_info["metadata_csv"]
            metadata_df = pd.read_csv(metadata_csv_path)
            tokens = get_matched_tokens(
                camera_feat_root=camera_root,
                lidar_root=lidar_root,
                fpn_level=cam_proj.fpn_level,
            )
            cam_ds = TorchCameraBEVDataset(
                sample_tokens=tokens,
                metadata_df=metadata_df,
                camera_feat_root=camera_root,
                camera_name=cam_proj.camera_name,
                fpn_level=cam_proj.fpn_level,
                bev_spec=bev_spec,
            )
            lidar_ds = TorchLidarBEVDataset(
                sample_tokens=tokens,
                lidar_root=lidar_root,
                pc_range=lidar_proj.pc_range,
                bev_spec=bev_spec,
            )
            datasets[split] = WrappedConcatBEVDataset(cam_ds, lidar_ds)
        return datasets



def main():
    # ---- Paths (Replace with your actual directories) ----
    camera_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT"
    lidar_dir  = "/home/draiman/Desktop/Datasets_nuscenes/validation/lidar"
    metadata_csv = "/home/draiman/Desktop/Datasets_nuscenes/nuscenes_metadata.csv"

    # ---- Projection / grid params ----
    camera_name = "CAM_FRONT"
    fpn_level = 0  # FPN level used for features
    pc_range = [-54, -54, -5, 54, 54, 3]  # LiDAR original range (example)
    bev_spec = BEVSpec(
        x_min=0.0, x_max=50.0, cell_x=0.4,
        y_min=-25.0, y_max=25.0, cell_y=0.25,
    )

    # ---- Prep token list ----
    tokens = get_matched_tokens(camera_dir, lidar_dir, fpn_level=fpn_level)

    # ---- Load metadata once ----
    metadata_df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(metadata_df)} metadata records.")
    print(f"BEV grid: {bev_spec.H} x {bev_spec.W}")

    # ---- Instantiate datasets ----
    cam_ds = TorchCameraBEVDataset(
        sample_tokens=tokens,
        metadata_df=metadata_df,
        camera_feat_root=camera_dir,
        camera_name=camera_name,
        fpn_level=fpn_level,
        bev_spec=bev_spec,
    )
    lidar_ds = TorchLidarBEVDataset(
        sample_tokens=tokens,
        lidar_root=lidar_dir,
        pc_range=pc_range,
        bev_spec=bev_spec,
    )
    dataset = WrappedConcatBEVDataset(cam_ds, lidar_ds)

    # ---- Inspect a few samples ----
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Camera feature shape: {tuple(sample['cam_bev'].shape)}")
        print(f"LiDAR BEV shape:     {tuple(sample['lidar_bev'].shape)}")
        print(f"Token/Image ID:      {sample['image_id']}")
        print(f"Label:               {sample['label']}")

        # Optionally, visualize the sample
        # visualize_sample(sample, title_prefix=f"Sample {i}: ")

if __name__ == "__main__":
    main()
