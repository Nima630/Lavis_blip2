"""
camera_bev_homography.py — CAM_FRONT flat-ground projection → BEV

Fixes:
- Normalize grid by *image size only*.
- Visibility mask (z<=0 or out-of-bounds) via 4D broadcasted masked_fill_.
- No self-import in __main__.
"""

from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import math
import numpy as np


def _as_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)

def project_to_bev_homography(
    img_feat: torch.Tensor,
    K,                              # [3,3] (numpy or torch)
    T_ego_cam_or_cam_ego,           # [4,4] (numpy or torch)
    bev_bounds: tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max) in meters
    bev_shape: tuple[int, int],                   # (H_bev, W_bev)
    z_ground: float = 0.0,
    T_is_ego2cam: bool = True,     # True if T is ego->cam; False if T is cam->ego
) -> torch.Tensor:
    """
    Returns [B, C, H_bev, W_bev]
    """
    device, dtype = img_feat.device, img_feat.dtype
    B, C, H_img, W_img = img_feat.shape
    H_bev, W_bev = bev_shape
    x_min, x_max, y_min, y_max = bev_bounds

    # Ego-plane grid (z = z_ground)
    xs = torch.linspace(x_min, x_max, W_bev, device=device, dtype=dtype)
    ys = torch.linspace(y_min, y_max, H_bev, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H_bev, W_bev]
    zz = torch.full_like(xx, float(z_ground))
    ones = torch.ones_like(xx)
    pts_ego = torch.stack([xx, yy, zz, ones], dim=-1).view(-1, 4).T  # [4, N]

    # Extrinsics: need ego->cam for projection
    T = _as_torch(T_ego_cam_or_cam_ego, device, torch.float32)
    if not T_is_ego2cam:
        T = torch.linalg.inv(T)  # convert cam->ego to ego->cam
    pts_cam = (T @ pts_ego)[:3]  # [3, N]

    # Intrinsics
    Kt = _as_torch(K, device, torch.float32)
    pix = Kt @ pts_cam  # [3, N]
    u = (pix[0] / (pix[2] + 1e-8)).view(H_bev, W_bev)
    v = (pix[1] / (pix[2] + 1e-8)).view(H_bev, W_bev)
    z = pts_cam[2].view(H_bev, W_bev)


    
    # Debugging: print projected pixel ranges
    print("u range:", u.min().item(), u.max().item())
    print("v range:", v.min().item(), v.max().item())
    print("H_img/W_img:", H_img, W_img)


    # Grid for grid_sample (normalize by *image* size)
    grid = torch.empty(1, H_bev, W_bev, 2, device=device, dtype=dtype)
    grid[0, ..., 0] = 2.0 * (u / (W_img - 1.0)) - 1.0
    grid[0, ..., 1] = 2.0 * (v / (H_img - 1.0)) - 1.0

    # Visibility + in-bounds mask → send OOB so zeros are sampled
    mask = (z <= 0) | (u < 0) | (u > (W_img - 1)) | (v < 0) | (v > (H_img - 1))  # [H_bev, W_bev]
    mask4d = mask.unsqueeze(0).unsqueeze(-1).expand(1, H_bev, W_bev, 2)          # [1,H_bev,W_bev,2]
    grid.masked_fill_(mask4d, -2.0)
    print(f"Valid BEV grid cells: {(~mask).float().mean().item()*100:.2f}%")

    bev_feat = F.grid_sample(
        img_feat,
        grid.expand(B, -1, -1, -1),
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    )
    return bev_feat

# -----------------------------
# __main__: smoke test
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Self-test for camera BEV homography (front cam)")
    ap.add_argument("--H_img", type=int, default=160)
    ap.add_argument("--W_img", type=int, default=320)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--bev_x0", type=float, default=0.0)
    ap.add_argument("--bev_x1", type=float, default=60.0)
    ap.add_argument("--bev_y0", type=float, default=-30.0)
    ap.add_argument("--bev_y1", type=float, default=30.0)
    ap.add_argument("--H_bev", type=int, default=120)
    ap.add_argument("--W_bev", type=int, default=120)
    ap.add_argument("--ego2cam", action="store_true", help="Interpret T as ego→cam (default: cam→ego)")
    args = ap.parse_args()

    # img_feat = torch.randn(1, args.C, args.H_img, args.W_img)
    img_feat = torch.arange(args.C*args.H_img*args.W_img, dtype=torch.float32).reshape(1, args.C, args.H_img, args.W_img)

    K = [
        [900.0, 0.0, args.W_img / 2.0],
        [0.0, 900.0, args.H_img / 2.0],
        [0.0, 0.0, 1.0],
    ]



    T_cam_ego = torch.eye(4, dtype=torch.float32)
    T_cam_ego[0, 3] = -5.0
    T_cam_ego[2, 3] = 2.0

    theta = math.radians(-10)
    ct = math.cos(theta)
    st = math.sin(theta)

    R = torch.tensor([
        [ct, 0, st, 0],
        [0,  1,  0, 0],
        [-st,0, ct, 0],
        [0,  0,  0, 1],
    ], dtype=torch.float32)

    R_pitch = torch.tensor([
        [1, 0,    0,   0],
        [0, ct, -st,   0],
        [0, st,  ct,   0],
        [0, 0,    0,   1],
        ], dtype=torch.float32)

    # T_cam_ego = R @ T_cam_ego
    T_cam_ego = R_pitch @ T_cam_ego

    bev_bounds = (args.bev_x0, args.bev_x1, args.bev_y0, args.bev_y1)
    bev_shape = (args.H_bev, args.W_bev)

    bev = project_to_bev_homography(
        img_feat, K, T_cam_ego, bev_bounds, bev_shape, T_is_ego2cam=args.ego2cam
    )
    print("Camera BEV shape:", tuple(bev.shape))
    print("Finite:", bool(torch.isfinite(bev).all().item()), "| mean |bev|:", float(bev.abs().mean().item()))

















