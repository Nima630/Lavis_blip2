import os
import glob
import torch
from torch.utils.data import ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
import torch
from  lavis.models.crop_lidar_bev import crop_lidar_bev_to_camera_fov


class TorchCameraDataset:
    def __init__(self, camera_files):
        self.camera_files = camera_files

    def __getitem__(self, idx):
        path = self.camera_files[idx]
        try:
            data = torch.load(path, weights_only=False)  # ← Fix for PyTorch 2.6+
            # print(f"[Camera] Loaded {path}")
            if 'feat' not in data:
                print(f"⚠️  'feat' key missing in file: {path}")
                return None
            image = data['feat'].squeeze(0)
            # print(f"🔍 Camera feature shape: {image.shape}")
            # print(f"📄 Keys in loaded dict: {list(data.keys())}")
        except Exception as e:
            print(f"[Camera] Error loading {path}: {e}")
            return None

        return {
            "image": image,
            "image_id": os.path.basename(path)
        }

    def __len__(self):
        return len(self.camera_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        return {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
        }


# class TorchLidarDataset:
#     def __init__(self, lidar_files):
#         self.lidar_files = lidar_files

#     def __getitem__(self, idx):
#         path = self.lidar_files[idx]
#         try:
#             data = torch.load(path)
#             lidar = data['feat']  # Updated to match your file format
#         except Exception as e:
#             print(f"[LiDAR] Error loading {path}: {e}")
#             return None

#         return {
#             "lidar": lidar,
#             "image_id": os.path.basename(path)
#         }

#     def __len__(self):
#         return len(self.lidar_files)

#     def collater(self, samples):
#         samples = [s for s in samples if s is not None]
#         if len(samples) == 0:
#             return {}
#         return {
#             k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
#             for k in samples[0]
#         }


class TorchLidarDataset:
    def __init__(self, lidar_files, camera_root, bev_resolution, pc_range):
        self.lidar_files = lidar_files
        self.camera_root = camera_root
        self.bev_resolution = bev_resolution
        self.pc_range = pc_range

    def __len__(self):
        return len(self.lidar_files)
 
    def __getitem__(self, idx):
        path = self.lidar_files[idx]
        sample_id = os.path.splitext(os.path.basename(path))[0]
        camera_path = os.path.join(self.camera_root, f"{sample_id}.pt")

        try:
            lidar_data = torch.load(path)
            lidar_feat = lidar_data['feat'].squeeze(0)  # [C, H, W]

            # Load camera data
            camera_data = torch.load(camera_path)
            lidar2img = camera_data['lidar2img']
            img_shape = tuple(camera_data['img_shape'][0])[:2]  # (H, W)

            # Crop using your existing function
            cropped_feat = crop_lidar_bev_to_camera_fov(
                lidar_feat, lidar2img, img_shape, self.bev_resolution, self.pc_range
            )

        except Exception as e:
            print(f"[LiDAR] Error loading {path} or {camera_path}: {e}")
            return None

        return {
            "lidar": cropped_feat,
            "image_id": sample_id
        }



class WrappedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        assert len(datasets) == 2
        super().__init__(datasets)
        self.camera_dataset = datasets[0]
        self.lidar_dataset = datasets[1]

    def __getitem__(self, idx):
        if idx >= len(self.camera_dataset) or idx >= len(self.lidar_dataset):
            return None
        cam_sample = self.camera_dataset[idx]
        lidar_sample = self.lidar_dataset[idx]
        if cam_sample is None or lidar_sample is None:
            return None
        return {**cam_sample, **lidar_sample, "label": 1}

    def __len__(self):
        return min(len(self.camera_dataset), len(self.lidar_dataset))

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        batch = self.camera_dataset.collater(samples)
        batch.update({"lidar": torch.stack([s["lidar"] for s in samples])})
        batch["label"] = torch.tensor([s["label"] for s in samples])
        return batch


class NuScenesDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg

    def build_datasets(self):
        build_info = self.config.datasets_cfg["nuscenes"].build_info
        datasets = {}

        for split in ["train", "val", "test"]:
            if split in self.config.run_cfg.train_splits + self.config.run_cfg.valid_splits + self.config.run_cfg.test_splits:
                camera_root = build_info["camera"][split].storage
                lidar_root = build_info["lidar"][split].storage

                matched_camera_files, matched_lidar_files = get_matched_files(camera_root, lidar_root)

                camera_dataset = TorchCameraDataset(matched_camera_files)
                # lidar_dataset = TorchLidarDataset(matched_lidar_files)
                lidar_dataset = TorchLidarDataset(
                    matched_lidar_files,
                    camera_root=camera_root,
                    bev_resolution=(1.14, 1.14),  # Use your actual resolution
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # Your BEV pc_range
                )
                datasets[split] = WrappedConcatDataset([camera_dataset, lidar_dataset])

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

    print(f" Found {len(camera_files)} camera files in {camera_root}")
    print(f"  Found {len(lidar_files)} lidar files in {lidar_root}")

    matched_keys = sorted(set(camera_files) & set(lidar_files))
    print(f" Matched {len(matched_keys)} files by token")

    if len(matched_keys) == 0:
        cam_example = list(camera_files.keys())[:5]
        lidar_example = list(lidar_files.keys())[:5]
        print(f" Sample camera tokens: {cam_example}")
        print(f" Sample lidar tokens: {lidar_example}")
        print("  Check if filenames match exactly (excluding '.pt')")

    matched_camera_files = [camera_files[k] for k in matched_keys]
    matched_lidar_files = [lidar_files[k] for k in matched_keys]

    return matched_camera_files, matched_lidar_files



def visualize_sample(sample, title_prefix=""):
    """
    Visualize camera feature heatmap and cropped LiDAR BEV feature heatmap.
    Args:
        sample (dict): Must contain 'image', 'lidar', and 'image_id'
    """
    cam_feat = sample['image'].squeeze(0).mean(dim=0).cpu().numpy()  # [H, W]
    lidar_feat = sample['lidar'].mean(dim=0).cpu().numpy()           # [H, W]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cam_feat, cmap='viridis')
    axes[0].set_title(f"{title_prefix}Camera Feature")
    axes[0].axis('off')

    axes[1].imshow(lidar_feat, cmap='viridis')
    axes[1].set_title(f"{title_prefix}LiDAR (Cropped to FOV)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    
def main():
    camera_dir = "/home/draiman/Desktop/Datasets/validation/resnet/CAM_FRONT"
    lidar_dir = "/home/draiman/Desktop/Datasets/validation/lidar_val"

    matched_camera_files, matched_lidar_files = get_matched_files(camera_dir, lidar_dir)

    print(f"🔍 Found {len(matched_camera_files)} matched samples.")

    camera_dataset = TorchCameraDataset(matched_camera_files)
    print("camera_dataset", type(camera_dataset))
    # lidar_dataset = TorchLidarDataset(matched_lidar_files)
    lidar_dataset = TorchLidarDataset(
    matched_lidar_files,
    camera_root=camera_dir,
    bev_resolution=(1.14, 1.14),  # ← Adjust if needed
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
)


    dataset = WrappedConcatDataset([camera_dataset, lidar_dataset])
    sample = None
#    Show a few matched samples
    for i in range(3):
        sample = dataset[i]
        print(f"\n📦 Sample {i}:")
        print(f" - Camera feat shape: {sample['image'].shape}")
        print(f" - Lidar feat shape: {sample['lidar'].shape}")
        print(f" - Token/Image ID: {sample['image_id']}")
        print(f" - Label: {sample['label']}")
    
      # 🔍 Visualize it
        visualize_sample(sample, title_prefix=f"Sample {i}: ")
    
if __name__ == "__main__":
    main()

    
# run the main with 
# python lavis/datasets/nuscenes.py --base-path lavis/my_datasets/nuscenes_dataset/train