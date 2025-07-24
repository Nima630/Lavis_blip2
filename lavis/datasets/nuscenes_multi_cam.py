
import os
import glob
import torch
from torch.utils.data import ConcatDataset
from PIL import Image
from collections import defaultdict
from  lavis.models.cam_to_bev import generate_bev_feature_fpn_map  # adjust to your actual import path
import argparse
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from collections import defaultdict
import torch
import os
import glob

import torch
import torch.nn.functional as F
import os
import traceback

def get_grouped_multi_cam_fpn_files(camera_root, lidar_root):
    grouped = defaultdict(dict)  # sample_idx → {cam_name: {level: path}}

    # print(f"[INFO] Scanning camera_root: {camera_root}")
    for cam_name in os.listdir(camera_root):
        cam_dir = os.path.join(camera_root, cam_name)
        if not os.path.isdir(cam_dir) or not cam_name.startswith("CAM_"):
            # print(f"[SKIP] {cam_name} is not a valid camera folder")
            continue

        # print(f"[SCAN] Camera: {cam_name}")
        pt_files = glob.glob(os.path.join(cam_dir, "*.pt"))
        # print(f"[INFO] Found {len(pt_files)} .pt files in {cam_name}")

        for pt_file in pt_files:
            try:
                base = os.path.basename(pt_file)
                if "_lvl" not in base:
                    print(f"[WARN] Skipping non-FPN file: {base}")
                    continue

                sample_id, lvl_part = base.split("_lvl")
                level = int(lvl_part.replace(".pt", ""))
                # print(f"[PARSE] File: {base} → sample_id={sample_id}, level={level}")

                data = torch.load(pt_file, map_location="cpu")
                sample_idx = data.get("sample_idx", None)

                if sample_idx is None:
                    print(f"[ERROR] sample_idx not found in {base}")
                    continue

                if cam_name not in grouped[sample_idx]:
                    grouped[sample_idx][cam_name] = {}

                grouped[sample_idx][cam_name][level] = pt_file
            except Exception as e:
                print(f"[ERROR] Failed loading {pt_file}: {e}")

    # print(f"[INFO] Loaded grouped samples: {len(grouped)} unique sample_idx entries.")

    # Find matched samples
    lidar_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(lidar_root, "*.pt"))
    }
    # print(f"[INFO] Found {len(lidar_files)} lidar files.")

    matched = []
    for sid, cams in grouped.items():
        if len(cams) != 6:
            print(f"[SKIP] sample_idx={sid} → only {len(cams)} cameras present")
            continue

        if sid not in lidar_files:
            print(f"[SKIP] sample_idx={sid} → no corresponding lidar file")
            continue

        all_have_4_levels = all(len(levels) == 4 for levels in cams.values())
        if not all_have_4_levels:
            print(f"[SKIP] sample_idx={sid} → not all cameras have 4 levels")
            continue

        # print(f"[MATCH] sample_idx={sid}")
        matched.append((cams, lidar_files[sid]))

    print(f"[INFO] Found {len(matched)} matched samples with all 6 cameras × 4 levels + lidar.")
    return matched



class TorchMultiCameraFPNDataset:
    def __init__(self, matched_samples):
        self.matched_samples = matched_samples
        print(f"[INIT] Loaded {len(matched_samples)} matched samples.")

    def __len__(self):
        return len(self.matched_samples)

    def __getitem__(self, idx):
        print(f"\n[GETITEM] idx={idx}")
        cam_dict, lidar_path = self.matched_samples[idx]

        try:
            cam_feats = []
            lidar2img_list = []
            cam_names = []

            cam_order = sorted(cam_dict.keys())
            print(f"[DEBUG] Expected cameras: {cam_order}")
            
            for cam_name in cam_order:
                levels = []
                print(f"[DEBUG] Reading FPN levels for camera: {cam_name}")
                pt_files = cam_dict[cam_name]
                print(f"[DEBUG] Level files: {pt_files}")



                for lvl in sorted(pt_files):  # Ensures consistent level order
                    
                    pt_file = pt_files[lvl]
                    if not os.path.exists(pt_file):
                        print(f"[ERROR] Missing file: {pt_file}")
                        raise FileNotFoundError(pt_file)
                    data = torch.load(pt_file, map_location="cpu")

                    if "feat" not in data:
                        print(f"[ERROR] 'feat' key missing in {pt_file}")
                        raise KeyError("'feat' not in FPN .pt file")

                    feat = data["feat"]  # shape: [1, C, H, W]
                    if feat.shape[-2:] != (116, 200):
                        print(f"[WARN] Resizing feature from {feat.shape[-2:]} to (116, 200)")
                        feat = F.interpolate(feat, size=(116, 200), mode="bilinear", align_corners=False)

                    feat = feat.squeeze(0)  # shape: [C, H, W]
                    levels.append(feat)

                    if lvl == 0:
                        lidar2img_list.append(data.get("lidar2img", torch.eye(4)))  # fallback
                        cam_names.append(data.get("cam_name", cam_name))

                print(f"[DEBUG] Shapes of levels for {cam_name}: {[x.shape for x in levels]}")
                cam_feats.append(torch.stack(levels))  # [4, C, H, W]

            cam_feats = torch.stack(cam_feats)  # [6, 4, C, 116, 200]
            print(f"[DEBUG] cam_feats.shape = {cam_feats.shape}")

            lidar2img = torch.stack(lidar2img_list)  # [6, 4, 4]
            print(f"[DEBUG] lidar2img.shape = {lidar2img.shape}")

            img_size = (116, 200)

            # Choose level 3: [6, C, H, W] → [6, 1, C, H, W]
            images_lvl3 = cam_feats[:, 3].unsqueeze(1)
            print(f"[DEBUG] images_lvl3.shape = {images_lvl3.shape}")

            # BEV Projection
            print("[DEBUG] Calling generate_bev_feature_fpn_map()...")
            bev_map, bev_raw, count_map = generate_bev_feature_fpn_map(images_lvl3, lidar2img, img_size)
            print(f"[DEBUG] bev_map.shape = {bev_map.shape}, bev_raw.shape = {bev_raw.shape}, count_map.shape = {count_map.shape}")
            bev_map = bev_map.unsqueeze(0)

            if not os.path.exists(lidar_path):
                print(f"[ERROR] Lidar path not found: {lidar_path}")
                raise FileNotFoundError(lidar_path)

            lidar = torch.load(lidar_path)["feat"]
            print(f"[DEBUG] lidar.shape = {lidar.shape}")

            sample_idx = os.path.splitext(os.path.basename(lidar_path))[0]
            print(f"[DEBUG] sample_idx = {sample_idx}")

        except Exception as e:
            print(f"[ERROR] Exception at idx={idx}: {e}")
            traceback.print_exc()
            return None

        return {
            "images": images_lvl3,         # [6, 1, C, 116, 200]
            "bev_image": bev_map,            # [1, C, H, W]
            "bev_raw": bev_raw,            # [H*W, C]
            "count_map": count_map,        # [H*W, 1]
            "lidar": lidar,                # [1, C, H, W]
            "sample_idx": sample_idx,
            "lidar2img": lidar2img,        # [6, 4, 4]
        }

    def collater(self, samples):
        print(f"[COLLATER] Collating batch of {len(samples)} samples...")
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            print("[COLLATER WARNING] All samples are None. Returning empty dict.")
            return {}

        out = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
                print(f"[COLLATER] Stacked tensor for '{key}': shape = {out[key].shape}")
            else:
                out[key] = vals
                print(f"[COLLATER] List field '{key}': len = {len(vals)}")
        return out



class TorchLidarDataset:
    def __init__(self, lidar_files):
        self.lidar_files = lidar_files

    def __getitem__(self, idx):
        path = self.lidar_files[idx]
        try:
            data = torch.load(path)
            lidar = data['feat']  # Updated to match your file format
        except Exception as e:
            print(f"[LiDAR] Error loading {path}: {e}")
            return None

        return {
            "lidar": lidar,
            "image_id": os.path.splitext(os.path.basename(path))[0]
        }

    def __len__(self):
        return len(self.lidar_files)

# creates train/val/test splits from config
class NuScenesDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg
    def build_datasets(self):
        build_info = self.config.datasets_cfg["nuscenes"].build_info
        datasets = {}

        for split in ["train", "val", "test"]:
            if split in self.config.run_cfg.train_splits + \
                    self.config.run_cfg.valid_splits + \
                    self.config.run_cfg.test_splits:

                camera_root = build_info["camera"][split].storage
                lidar_root = build_info["lidar"][split].storage

                matched_samples = get_grouped_multi_cam_fpn_files(camera_root, lidar_root)

                # Each sample now includes all 6 cameras + lidar + lidar2img
                # dataset = TorchMultiCameraDataset(matched_samples)
                dataset = TorchMultiCameraFPNDataset(matched_samples)
                datasets[split] = dataset

        return datasets


def get_matched_files(camera_root, lidar_root):
    camera_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(camera_root, "*.pt"))
    }
    lidar_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(lidar_root, "*.pt"))
    }

    matched_keys = sorted(set(camera_files) & set(lidar_files))
    matched_camera_files = [camera_files[k] for k in matched_keys]
    matched_lidar_files = [lidar_files[k] for k in matched_keys]

    return matched_camera_files, matched_lidar_files

