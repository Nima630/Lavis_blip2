"""
centerpoint_bev_front_crop.py

Extracts a front-facing BEV crop (e.g., 0-60m forward, -30m to +30m lateral) from a CenterPoint (or similar MMDetection3D) BEV feature map.

- Assumes BEV features of shape [B, C, H, W] (e.g., [1,256,90,90]) from backbone.
- Uses pc_range and voxel_size from your config.
- Supports bilinear resize to any output (default 120x120).

Example usage:
    feat = torch.load('bev_feat.pt')['feat']
    bev_front = crop_front_bev(
        feat, x_min=-54, y_min=-54, x_max=54, y_max=54, vx=0.075, vy=0.075, s=16,
        x0=0, x1=60, y0=-30, y1=30, out_hw=(120,120)
    )
    # bev_front: [B,256,120,120]
"""

import torch
import torch.nn.functional as F
import math

def crop_front_bev(feat, x_min, y_min, x_max, y_max, vx, vy, s, x0, x1, y0, y1, out_hw=(120,120)):
    """
    Crop and resize front BEV region from CenterPoint features.

    Args:
        feat:   [B,C,H,W] BEV feature map
        x_min, y_min, x_max, y_max: full pc_range from config
        vx, vy: voxel sizes in meters
        s:      BEV downsample factor (e.g. 16)
        x0, x1: desired front range (meters, forward)
        y0, y1: desired side range (meters, left/right)
        out_hw: output crop size (e.g. 120x120)
    Returns:
        crop_resized: [B,C,out_hw[0],out_hw[1]]
    """
    def x_to_j(x): return (x - x_min) / vx / s
    def y_to_i_top(y): return (y_max - y) / vy / s
    j0 = int(math.ceil(x_to_j(x0)))
    j1 = int(math.floor(x_to_j(x1)))
    i_top = int(math.floor(y_to_i_top(y1)))
    i_bottom = int(math.ceil(y_to_i_top(y0)))
    # Clamp to feature map bounds
    _,_,H,W = feat.shape
    j0 = max(0, min(W-1, j0)); j1 = max(j0+1, min(W, j1))
    i_top = max(0, min(H-1, i_top)); i_bottom = max(i_top+1, min(H, i_bottom))
    crop = feat[:, :, i_top:i_bottom, j0:j1]
    crop_resized = F.interpolate(crop, size=out_hw, mode="bilinear", align_corners=False)
    return crop_resized

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pt CenterPoint BEV feature (dict with 'feat' key)")
    parser.add_argument("--output", required=True, help="Path to save cropped front BEV (.pt)")
    parser.add_argument("--out_hw", type=int, nargs=2, default=[120,120], help="Output crop size")
    parser.add_argument("--s", type=int, default=16, help="Downsample factor (CenterPoint 90x90 usually s=16)")
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--x1", type=float, default=60.0)
    parser.add_argument("--y0", type=float, default=-30.0)
    parser.add_argument("--y1", type=float, default=30.0)
    args = parser.parse_args()

    # Update these to match your config
    x_min, y_min, x_max, y_max = -54, -54, 54, 54
    vx, vy = 0.075, 0.075
    s = args.s

    # Load BEV feature
    d = torch.load(args.input, map_location='cpu')
    feat = d['feat']           # [B,256,90,90] from CenterPoint
    crop = crop_front_bev(feat, x_min, y_min, x_max, y_max, vx, vy, s,
                         args.x0, args.x1, args.y0, args.y1, tuple(args.out_hw))
    torch.save({'feat': crop}, args.output)
    print(f"Saved {args.output}  shape={crop.shape}")







# python centerpoint_bev_front_crop.py \
#   --input path/to/your/bev_feat.pt \
#   --output path/to/front_bev.pt \
#   --out_hw 120 120



