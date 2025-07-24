import os
import glob
import torch
from torch.utils.data import ConcatDataset
from PIL import Image



class TorchCameraDataset:
    def __init__(self, camera_files):
        self.camera_files = camera_files

    def __getitem__(self, idx):
        path = self.camera_files[idx]
        try:
            data = torch.load(path)
            image = data['feat']  # Updated to match your file format
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
            "image_id": os.path.basename(path)
        }

    def __len__(self):
        return len(self.lidar_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        return {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
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
                lidar_dataset = TorchLidarDataset(matched_lidar_files)

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

    matched_keys = sorted(set(camera_files) & set(lidar_files))
    matched_camera_files = [camera_files[k] for k in matched_keys]
    matched_lidar_files = [lidar_files[k] for k in matched_keys]

    return matched_camera_files, matched_lidar_files

if __name__ == "__main__":
    import os

    # === Simulate config structure with classes ===
    class DummyStorage:
        def __init__(self, path):
            self.storage = path

    class DummySplit:
        def __init__(self, camera_path, lidar_path):
            self.camera = {"train": DummyStorage(camera_path)}
            self.lidar = {"train": DummyStorage(lidar_path)}

    class DummyNuscenesConfig:
        def __init__(self, camera_path, lidar_path):
            self.build_info = {
                "camera": {"train": DummyStorage(camera_path)},
                "lidar": {"train": DummyStorage(lidar_path)}
            }

            # Mimic the nested OmegaConf-style object
            self.vis_processor = None  # Optional: for future use

    class DummyCfg:
        class RunCfg:
            train_splits = ["train"]
            valid_splits = []
            test_splits = []

        def __init__(self, camera_path, lidar_path):
            self.run_cfg = DummyCfg.RunCfg()
            self.datasets_cfg = {
                "nuscenes": DummyNuscenesConfig(camera_path, lidar_path)
            }

    # === Define correct data paths relative to nuscenes.py ===
    data_root = os.path.join(os.path.dirname(__file__), "../my_datasets/nuscenes_dataset/train")
    camera_data_path = os.path.join(data_root, "camera_front")
    lidar_data_path = os.path.join(data_root, "lidar")

    # === Create dummy config
    cfg = DummyCfg(camera_data_path, lidar_data_path)

    # === Use your dataset builder
    builder = NuScenesDatasetBuilder(cfg)
    datasets = builder.build_datasets()

    # === Access and test train split
    dataset = datasets.get("train")
    print(f"[INFO] Dataset has {len(dataset)} matched samples.")

    sample = dataset[0]
    if sample:
        print(f"[SAMPLE] Image ID: {sample['image_id']}")
        print(f"[SAMPLE] Camera tensor shape: {sample['image'].shape}")
        print(f"[SAMPLE] LiDAR tensor shape: {sample['lidar'].shape}")
    else:
        print("[WARNING] Sample 0 is None.")


# import torch
# import pprint

# # Change these paths to actual files
# cam_path = "../my_datasets/nuscenes_dataset/train/camera_front/000cf4dfaab54d21a7314036fde74966.pt"
# lidar_path = "../my_datasets/nuscenes_dataset/train/lidar/000cf4dfaab54d21a7314036fde74966.pt"

# pp = pprint.PrettyPrinter(indent=2)

# print("[Camera PT]")
# cam_data = torch.load(cam_path)
# print(type(cam_data))
# if isinstance(cam_data, dict):
#     print("Keys:", cam_data.keys())
#     if "meta" in cam_data:
#         print("Meta info:")
#         pp.pprint(cam_data["meta"])
#     else:
#         print("No 'meta' key found in camera file.")
# else:
#     print("Camera .pt file is not a dict, got:", cam_data.shape if hasattr(cam_data, 'shape') else cam_data)

# print("\n[LiDAR PT]")
# lidar_data = torch.load(lidar_path)
# print(type(lidar_data))
# if isinstance(lidar_data, dict):
#     print("Keys:", lidar_data.keys())
#     if "meta" in lidar_data:
#         print("Meta info:")
#         pp.pprint(lidar_data["meta"])
#     else:
#         print("No 'meta' key found in lidar file.")
# else:
#     print("LiDAR .pt file is not a dict, got:", lidar_data.shape if hasattr(lidar_data, 'shape') else lidar_data)
