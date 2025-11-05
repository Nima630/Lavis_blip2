
import torch
import os
import glob
import numpy as np


def read_cam_path(camera_root, fpn_level):
        
    camera_files = {
        os.path.splitext(os.path.basename(f))[0].replace(f"_lvl{fpn_level}", ""): f
        for f in glob.glob(os.path.join(camera_root, f"*lvl{fpn_level}.pt"))
    }
    return camera_files

# --- Demo (remove this block if you have real P2) ---
# Dummy P2 to test the printouts:
if __name__ == "__main__":
    
    fpn_level = 0  # 0 = P2, 1 = P3, 2 = P4, 3 = P5 (BEVFormer default: 2 or 3)
    camera_dir = "/home/draiman/Desktop/Datasets_nuscenes/validation/fpn/CAM_FRONT"

    cam_files = read_cam_path(camera_dir, fpn_level)
    cam_paths = list(cam_files.values())
    for i in range(1):
        cam = torch.load(cam_paths[i], map_location='cpu')
        cam_feat = cam['feat']   
        # print("keys", cam.keys())     # [1,256,H_feat,W_feat] or [256,H_feat,W_feat]
        print("img_shape ", cam['img_shape'])     # [1,256,H_feat,W_feat] or [256,H_feat,W_feat]
        if cam_feat.ndim == 3:
            cam_feat = cam_feat.unsqueeze(0)
        print(f"Sample {i}")
        print(f"  Camera feature shape: {cam_feat.shape}")




    # ---- Step 2: Define BEV grid ----
    # Set BEV spatial coverage and cell size
    cell_x = 0.4    # meters per cell (longitudinal, X/forward)
    cell_y = 0.25   # meters per cell (lateral, Y/sideways)
    x_min, x_max = 0.0, 50.0      # forward: 0 to 50 meters
    y_min, y_max = -25.0, 25.0    # lateral: -25 to +25 meters

    H_bev = int(np.ceil((x_max - x_min) / cell_x))
    W_bev = int(np.ceil((y_max - y_min) / cell_y))
    print(f"BEV grid: {H_bev} x {W_bev} cells  (cell size: {cell_x}m x {cell_y}m)")




    # ---- Step 3: Compute image pixel centers for each P2 cell ----
    B, C, Hf, Wf = cam_feat.shape  # from your loaded P2
    stride_p2 = 4                  # Adjust if your FPN uses a different stride

    # Each P2 cell corresponds to image pixel:
    #   u = (j + 0.5) * stride_p2
    #   v = (i + 0.5) * stride_p2
    j = torch.arange(Wf, dtype=torch.float32) + 0.5
    i = torch.arange(Hf, dtype=torch.float32) + 0.5
    u = j * stride_p2
    v = i * stride_p2

    # Create meshgrid of pixel centers (U: horizontal, V: vertical)
    U, V = torch.meshgrid(u, v, indexing='xy')
    U = U.t().contiguous()  # shape [Hf, Wf]
    V = V.t().contiguous()  # shape [Hf, Wf]

    print(f"U (image x pixel centers): shape {U.shape}, min={U.min().item():.1f}, max={U.max().item():.1f}")
    print(f"V (image y pixel centers): shape {V.shape}, min={V.min().item():.1f}, max={V.max().item():.1f}")


















import pickle
from pyquaternion import Quaternion
import numpy as np

# Load the info file (as in your script)
nuscenes_infos = "/home/draiman/Desktop/datasets/nuscenes"
with open(f"{nuscenes_infos}/nuscenes_infos_temporal_val.pkl", "rb") as f:
    data = pickle.load(f)
infos = data['infos']  # Top key as printed

# Pick a sample (e.g., index 0)
sample = infos[0]

# Pick the camera (e.g., 'CAM_FRONT')
cam_name = 'CAM_FRONT'
cam_info = sample['cams'][cam_name]
# print(f"Inspecting camera: {cam_info.keys()}")
# Camera intrinsics (3x3)
K = np.array(cam_info['cam_intrinsic'])  # 3x3

# Camera-to-ego rotation (quaternion to rotation matrix)
q = cam_info['sensor2ego_rotation']  # [w, x, y, z]
R = Quaternion(q).rotation_matrix    # 3x3

# Camera-to-ego translation
T = np.array(cam_info['sensor2ego_translation'])  # (3,)

print("Intrinsic matrix K:\n", K)
print("Rotation matrix R:\n", R)
print("Translation vector T:\n", T)


















import numpy as np
import torch
from pyquaternion import Quaternion

# 1. Assume you have cam_feat [B, C, Hf, Wf] loaded, and K, R, T as before
#    Also, set stride_p2 and BEV grid params

stride_p2 = 4  # or whatever matches your P2
B, C, Hf, Wf = cam_feat.shape

# Unified BEV grid definition (same as before)
x_min, x_max, cell_x = 0.0, 50.0, 0.4
y_min, y_max, cell_y = -25.0, 25.0, 0.25
H_bev = int(np.ceil((x_max - x_min) / cell_x))
W_bev = int(np.ceil((y_max - y_min) / cell_y))
bev_grid = torch.zeros((B, C, H_bev, W_bev), dtype=cam_feat.dtype)

# 2. For each P2 feature cell, compute its (u, v) image pixel center
i = torch.arange(Hf, dtype=torch.float32) + 0.5
j = torch.arange(Wf, dtype=torch.float32) + 0.5
v = i * stride_p2
u = j * stride_p2
U, V = torch.meshgrid(u, v, indexing='xy')
U = U.t().contiguous()  # [Hf, Wf]
V = V.t().contiguous()  # [Hf, Wf]

# 3. Convert image pixel centers (u, v) to 3D rays, then to ground points
#    Assume ground is at z=0, and all pixels "look" forward

# Rebuild camera calibration matrices as numpy arrays
K_np = np.asarray(K)
R_np = np.asarray(R)
T_np = np.asarray(T)

# Prepare inverse intrinsic for backprojection
K_inv = np.linalg.inv(K_np)
all_uv1 = np.stack([U.reshape(-1), V.reshape(-1), np.ones(Hf * Wf)], axis=0)  # [3, N]

# Backproject to camera rays (normalized direction)
rays_cam = K_inv @ all_uv1  # [3, N]
# Assume camera at (0,0,0), rays_cam[:,i] points into scene

# 4. Transform rays to ego (vehicle) coordinates
rays_ego = R_np @ rays_cam  # [3, N]

# 5. Intersect rays with ground plane (z=0), solve for scale s:
#    0 = cam_z + s * dir_z  -->  s = -cam_z / dir_z
cam_height = T_np[2]
s = -cam_height / rays_ego[2, :]
X_ground = T_np[0] + s * rays_ego[0, :]
Y_ground = T_np[1] + s * rays_ego[1, :]
# (Ignore points where rays point up: s <= 0)

print("X_ground min/max:", X_ground.min(), X_ground.max())
print("Y_ground min/max:", Y_ground.min(), Y_ground.max())


# 6. Convert (X_ground, Y_ground) to BEV grid indices
ix = ((X_ground - x_min) / cell_x).astype(np.int64)
iy = ((Y_ground - y_min) / cell_y).astype(np.int64)
valid = (ix >= 0) & (ix < H_bev) & (iy >= 0) & (iy < W_bev) & (s > 0)
print("Number of valid points:", valid.sum())
# 7. Accumulate (splat) features into BEV
cam_feat_flat = cam_feat.reshape(B, C, -1)  # [B, C, N]
for b in range(B):
    for n in range(Hf * Wf):
        if valid[n]:
            bev_grid[b, :, ix[n], iy[n]] += cam_feat_flat[b, :, n]

# 8. (Optional) Normalize by count if multiple features hit the same cell

print(f"BEV grid shape: {bev_grid.shape}")












































