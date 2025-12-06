
import os
import glob
import csv
import ast
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import numpy as np
from lavis.models.crop_lidar_bev import crop_lidar_bev_to_camera_fov


# ============================================================
# BEV spec
# ============================================================

class BEVSpec:
    """
    Defines the canonical BEV grid for fusion.
    """
    def __init__(self, x_min=0.0, x_max=50.0, cell_x=0.4,
                 y_min=-25.0, y_max=25.0, cell_y=0.25):
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


# ============================================================
# Calibration handling (from calib_trainval_front.csv)
# ============================================================

class CameraCalib:
    """
    Holds per-sample camera & LiDAR calibration.
    """
    def __init__(self, K, R, T, lidar2img, img_shape):
        self.K = K          # 3x3
        self.R = R          # 3x3, cam->ego
        self.T = T          # (3,)
        self.lidar2img = lidar2img  # 3x4
        self.img_shape = img_shape  # (H, W)


def load_calib_db(csv_file, cam_channel="CAM_FRONT"):
    """
    Build a dict:
        (sample_token, cam_channel) -> CameraCalib

    Same parsing semantics as your Jupyter load_calib_from_csv.
    """
    calib_db = {}
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row["sample_token"]
            ch = row["cam_channel"]
            # Only keep the camera channel we care about (e.g., CAM_FRONT)
            if ch != cam_channel:
                continue

            K = np.array(ast.literal_eval(row["K_flat"]), dtype=np.float64).reshape(3, 3)
            R = np.array(ast.literal_eval(row["R_flat"]), dtype=np.float64).reshape(3, 3)
            T = np.array(ast.literal_eval(row["T_flat"]), dtype=np.float64).reshape(3,)
            lidar2img = np.array(ast.literal_eval(row["lidar2img_flat"]),
                                 dtype=np.float64).reshape(3, 4)
            img_shape = (int(row["img_height"]), int(row["img_width"]))
            calib_db[(token, ch)] = CameraCalib(K=K, R=R, T=T,
                                                lidar2img=lidar2img,
                                                img_shape=img_shape)
    return calib_db


# ============================================================
# Token matching from camera & lidar dirs
# ============================================================

def get_matched_tokens(camera_feat_root, lidar_root, fpn_level):
    """
    Match tokens based on filenames:
      camera: {token}_lvl{fpn}.pt
      lidar:  {token}.pt
    """
    cam_files = glob.glob(os.path.join(camera_feat_root, f"*lvl{fpn_level}.pt"))
    lidar_files = glob.glob(os.path.join(lidar_root, "*.pt"))

    cam_tokens = {
        os.path.basename(f).replace(f"_lvl{fpn_level}.pt", ""): f
        for f in cam_files
    }
    lidar_tokens = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in lidar_files
    }

    matched = sorted(set(cam_tokens.keys()) & set(lidar_tokens.keys()))
    return matched


# ============================================================
# Camera BEV dataset (uses K,R,T,img_shape from calib CSV)
# ============================================================

class TorchCameraBEVDataset:
    """
    Loads and projects FPN camera features into BEV grid using
    explicit calibration (K, R, T) from calib_trainval_front.csv.
    """
    def __init__(self,
                 sample_tokens,
                 camera_feat_root: str,
                 camera_name: str,
                 fpn_level: int,
                 bev_spec: BEVSpec,
                 calib_db: dict):
        self.sample_tokens = sample_tokens
        self.camera_feat_root = camera_feat_root
        self.camera_name = camera_name
        self.fpn_level = fpn_level
        self.bev_spec = bev_spec
        self.calib_db = calib_db

    def __len__(self):
        return len(self.sample_tokens)

    def _lookup_calib(self, token: str) -> CameraCalib:
        key = (token, self.camera_name)
        if key not in self.calib_db:
            raise KeyError(f"Calibration not found for token={token}, cam={self.camera_name}")
        return self.calib_db[key]

    def _load_camera_feat(self, token: str, calib: CameraCalib):
        """
        Load camera FPN feature and compute stride using image shape.
        """
        feat_filename = f"{token}_lvl{self.fpn_level}.pt"
        feat_path = os.path.join(self.camera_feat_root, feat_filename)
        blob = torch.load(feat_path, map_location="cpu")
        cam_feat = blob["feat"]   # [1,C,Hf,Wf] or [C,Hf,Wf]

        if cam_feat.ndim == 3:
            cam_feat = cam_feat.unsqueeze(0)  # [1,C,Hf,Wf]

        Hf, Wf = cam_feat.shape[-2], cam_feat.shape[-1]
        H_img, W_img = calib.img_shape

        # Your Jupyter code did: stride = img_shape[0] // Hf
        # Use float stride for more precision; it's equivalent if divisible.
        stride = H_img / Hf  # assuming same aspect ratio

        self._stride = stride
        self._img_shape = (H_img, W_img)
        return cam_feat

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        calib = self._lookup_calib(token)
        cam_feat = self._load_camera_feat(token, calib)  # [1,C,Hf,Wf]
        B, C, Hf, Wf = cam_feat.shape
        stride = self._stride

        # ---- Compute pixel centers (like in your Jupyter code) ----
        i = torch.arange(Hf, dtype=torch.float32) + 0.5
        j = torch.arange(Wf, dtype=torch.float32) + 0.5
        V, U = torch.meshgrid(i * stride, j * stride, indexing='ij')  # [Hf,Wf]

        U = U.numpy()
        V = V.numpy()

        # ---- Backproject rays → camera → ego ----
        K = calib.K
        R_cam2ego = calib.R
        T_cam2ego = calib.T

        K_inv = np.linalg.inv(K)
        uv1 = np.stack([U.reshape(-1), V.reshape(-1), np.ones(Hf * Wf)], axis=0)
        rays_cam = K_inv @ uv1
        rays_cam[1, :] *= -1.0  # flip y (same as your Jupyter code)

        rays_ego = R_cam2ego @ rays_cam
        C_ego = T_cam2ego.reshape(3)

        cam_z = C_ego[2]
        dir_z = rays_ego[2, :]
        s = -cam_z / (dir_z + 1e-12)  # intersection with ground plane z=0
        valid = s > 0

        X = C_ego[0] + s * rays_ego[0, :]
        Y = C_ego[1] + s * rays_ego[1, :]

        # ---- Define canonical BEV grid (same as your Jupyter) ----
        x_min, x_max = self.bev_spec.x_min, self.bev_spec.x_max
        y_min, y_max = self.bev_spec.y_min, self.bev_spec.y_max
        cell_x, cell_y = self.bev_spec.cell_x, self.bev_spec.cell_y
        H_bev, W_bev = self.bev_spec.H, self.bev_spec.W

        bev = torch.zeros((C, H_bev, W_bev), dtype=cam_feat.dtype)
        

        ix = ((X - x_min) / cell_x).astype(np.int64)
        iy = ((Y - y_min) / cell_y).astype(np.int64)
        inside = valid & (ix >= 0) & (ix < H_bev) & (iy >= 0) & (iy < W_bev)

        feat_flat = cam_feat.reshape(C, -1)  # [C, Hf*Wf]
        idxs = np.where(inside)[0]
        for n in idxs:
            bev[:, ix[n], iy[n]] += feat_flat[:, n]
        # print("Camera sum:", bev.sum().item())
        # print("CAM BEV SHAPE:", bev.shape) 
        # print("CAM FEATURE SHAPE:", cam_feat.shape, 
        # "from", os.path.join(self.camera_feat_root, f"{token}_lvl{self.fpn_level}.pt"))
        # if idx == 0:
        #     torch.save(bev.cpu(), "/home/draiman/Desktop/debug_/raw_cam_bev.pt")

        # [C,H_bev,W_bev]
        return {
            "cam_bev": bev,
            "image_id": token,
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        cam_bev_batch = torch.stack([s["cam_bev"] for s in samples])
        image_ids = [s["image_id"] for s in samples]
        
        
        # # if random.random() < 0.01:   # occasionally debug
        # raw = cam_bev_batch[0].cpu()
        # torch.save(raw, "/home/draiman/Desktop/debug_recons_2/raw_cam_bev.pt")

        
        return {
            "cam_bev": cam_bev_batch,
            "image_id": image_ids,
        }


# ============================================================
# LiDAR BEV dataset (uses lidar2img + FOV masking + grid_sample)
# ============================================================

class TorchLidarBEVDataset:
    """
    Loads LiDAR BEV features, masks them to the camera FOV using lidar2img,
    then resamples to the canonical BEV grid.
    """
    def __init__(self,
                 sample_tokens,
                 lidar_root: str,
                 pc_range,
                 bev_spec: BEVSpec,
                 calib_db: dict,
                 camera_name: str):
        self.sample_tokens = sample_tokens
        self.lidar_root = lidar_root
        self.pc_range = pc_range  # e.g., [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.bev_spec = bev_spec
        self.calib_db = calib_db
        self.camera_name = camera_name

    def __len__(self):
        return len(self.sample_tokens)

    def _lookup_calib(self, token: str) -> CameraCalib:
        key = (token, self.camera_name)
        if key not in self.calib_db:
            raise KeyError(f"Calibration not found for token={token}, cam={self.camera_name}")
        return self.calib_db[key]

    def _load_lidar_feat(self, token: str):
        """
        Load LiDAR BEV tensor for a token.
        You used blob['spatial'].squeeze(0) in your notebook.
        """
        feat_path = os.path.join(self.lidar_root, f"{token}.pt")
        blob = torch.load(feat_path, map_location="cpu")

        if "spatial" in blob:
            lidar_feat = blob["spatial"]
        # if "bev_c1" in blob:
        #     lidar_feat = blob["bev_c1"]

        else:
            # fallback if stored as "feat"
            lidar_feat = blob["feat"]
            # lidar_feat = blob["spatial"]

        if lidar_feat.ndim == 3:
            lidar_feat = lidar_feat.unsqueeze(0)  # [1, C, H_raw, W_raw]

        return lidar_feat

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        calib = self._lookup_calib(token)
        lidar_feat = self._load_lidar_feat(token)  # [1, C, H_raw, W_raw]
        B, C, H_raw, W_raw = lidar_feat.shape
        assert B == 1, "Expected LiDAR feat with batch dim 1."

        lidar_bev = lidar_feat.squeeze(0)  # [C, H_raw, W_raw]

        x_min_l, y_min_l, _, x_max_l, y_max_l, _ = self.pc_range

        # ---- Compute BEV resolution from pc_range & raw shape ----
        # (You used bev_resolution=(0.57,0.57) in the notebook; here we compute it generically.)
        bev_res_x = (x_max_l - x_min_l) / H_raw
        bev_res_y = (y_max_l - y_min_l) / W_raw
        bev_resolution = (bev_res_x, bev_res_y)

        # ---- FOV masking using lidar2img + img_shape ----
        lidar2img = calib.lidar2img
        img_shape = calib.img_shape  # (H_img, W_img)

        masked_lidar = crop_lidar_bev_to_camera_fov(
            lidar_bev,            # [C, H_raw, W_raw]
            lidar2img,            # 3x4 numpy array
            img_shape,            # (H, W)
            bev_resolution=bev_resolution,
            pc_range=self.pc_range,
        )  # [C, H_raw, W_raw] masked to camera FOV
        
        # masked_lidar = lidar_bev # no masking for testing

        # ---- Resample masked LiDAR into canonical BEV grid via grid_sample ----
        masked_lidar_batch = masked_lidar.unsqueeze(0)  # [1, C, H_raw, W_raw]

        H_cam = self.bev_spec.H
        W_cam = self.bev_spec.W
        cell_x = (self.bev_spec.x_max - self.bev_spec.x_min) / H_cam
        cell_y = (self.bev_spec.y_max - self.bev_spec.y_min) / W_cam

        xs = self.bev_spec.x_min + (torch.arange(H_cam, dtype=torch.float32) + 0.5) * cell_x
        ys = self.bev_spec.y_min + (torch.arange(W_cam, dtype=torch.float32) + 0.5) * cell_y
        Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')

        grid_x = 2.0 * (Yg - y_min_l) / (y_max_l - y_min_l) - 1.0
        grid_y = 2.0 * (Xg - x_min_l) / (x_max_l - x_min_l) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H_cam, W_cam, 2]

        lidar_bev_canonical = F.grid_sample(
            masked_lidar_batch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)  # [C, H_cam, W_cam]
        # print("raw lidar:", lidar_bev.shape)               # after squeeze(0)
        # print("masked lidar:", masked_lidar.shape)         # after crop_lidar_bev_to_camera_fov
        # print("canonical:", lidar_bev_canonical.shape)     # after grid_sample

        return {
            "lidar_bev": lidar_bev_canonical,
            "image_id": token,
        }


# ============================================================
# Wrapper dataset to return (cam_bev, lidar_bev) pairs
# ============================================================

class WrappedConcatBEVDataset(ConcatDataset):
    """
    Returns paired (camera, lidar) BEV tensors for each token.
    """
    def __init__(self,
                 cam_dataset: TorchCameraBEVDataset,
                 lidar_dataset: TorchLidarBEVDataset):
        assert len(cam_dataset) == len(lidar_dataset), \
            "Camera and LiDAR token sets must align 1:1."
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
    def __init__(self, cfg):
        self.cfg = cfg

    def build_datasets(self):
        build_info = self.cfg.datasets_cfg["nuscenes"].build_info
        datasets = {}

        for split in ["train", "val", "test"]:
            # Only build if split in run config
            if split not in (self.cfg.run_cfg.train_splits + 
                             self.cfg.run_cfg.valid_splits +
                             self.cfg.run_cfg.test_splits):
                continue

            camera_root = build_info["camera"][split].storage
            lidar_root = build_info["lidar"][split].storage
            calib_csv = build_info["metadata_csv"]
            # calib_csv = build_info.get("calib_csv", 
                        #  "/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/final_resources/calib_trainval_front.csv")
            camera_name = build_info["cam_proj"].camera_name
            fpn_level = build_info["cam_proj"].fpn_level
            pc_range = build_info["lidar_proj"].pc_range

            bev_cfg = build_info["bev"]
            bev_spec = BEVSpec(
                x_min=bev_cfg.x_min, x_max=bev_cfg.x_max, cell_x=bev_cfg.cell_x,
                y_min=bev_cfg.y_min, y_max=bev_cfg.y_max, cell_y=bev_cfg.cell_y,
            )

            calib_db = load_calib_db(calib_csv, cam_channel=camera_name)
            tokens = get_matched_tokens(camera_root, lidar_root, fpn_level)

            cam_ds = TorchCameraBEVDataset(
                sample_tokens=tokens,
                camera_feat_root=camera_root,
                camera_name=camera_name,
                fpn_level=fpn_level,
                bev_spec=bev_spec,
                calib_db=calib_db,
            )
            lidar_ds = TorchLidarBEVDataset(
                sample_tokens=tokens,
                lidar_root=lidar_root,
                pc_range=pc_range,
                bev_spec=bev_spec,
                calib_db=calib_db,
                camera_name=camera_name,
            )
            datasets[split] = WrappedConcatBEVDataset(cam_ds, lidar_ds)

        return datasets

# ============================================================
# main(): same style as your simple example
# ============================================================
def main():
    # ---- Paths (you can change these later) ----
    camera_dir = "/home/draiman/Desktop/Datasets_nuscenes/train/fpn/CAM_FRONT"  # where {token}_lvl0.pt live
    lidar_dir = "/data/nusc_lidar/train"    # where {token}.pt live
    calib_csv = "/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/final_resources/calib_trainval_front.csv"

    camera_name = "CAM_FRONT"
    fpn_level = 0
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    bev_spec = BEVSpec(
        x_min=0.0, x_max=50.0, cell_x=0.4,
        y_min=-25.0, y_max=25.0, cell_y=0.25,
    )

    # ---- Load calibration DB once ----
    calib_db = load_calib_db(calib_csv, cam_channel=camera_name)
    print(f"Loaded {len(calib_db)} calibration entries for {camera_name}.")

    # ---- Match tokens from camera + lidar dirs ----
    tokens = get_matched_tokens(camera_dir, lidar_dir, fpn_level=fpn_level)
    print(f"Found {len(tokens)} matched tokens.")

    # ---- Instantiate datasets ----
    cam_ds = TorchCameraBEVDataset(
        sample_tokens=tokens,
        camera_feat_root=camera_dir,
        camera_name=camera_name,
        fpn_level=fpn_level,
        bev_spec=bev_spec,
        calib_db=calib_db,
    )

    lidar_ds = TorchLidarBEVDataset(
        sample_tokens=tokens,
        lidar_root=lidar_dir,
        pc_range=pc_range,
        bev_spec=bev_spec,
        calib_db=calib_db,
        camera_name=camera_name,
    )

    dataset = WrappedConcatBEVDataset(cam_ds, lidar_ds)

    # ---- Inspect a few samples ----
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        if sample is None:
            continue
        print(f"\nSample {i}:")
        print(f"Camera BEV shape: {tuple(sample['cam_bev'].shape)}")
        print(f"LiDAR  BEV shape: {tuple(sample['lidar_bev'].shape)}")
        print(f"Token/Image ID:  {sample['image_id']}")
        print(f"Label:           {sample['label']}")


        
        print("CAM FEATURE SHAPE:", sample['cam_bev'].shape, "from", os.path.join(camera_dir, f"{tokens}_lvl{0}.pt"))

    return dataset  
if __name__ == "__main__":
    dataset = main()
