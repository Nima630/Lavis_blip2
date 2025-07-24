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
            "image_id": os.path.splitext(os.path.basename(path))[0]

        }

    def __len__(self):
        return len(self.camera_files)



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





class WrappedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        assert len(datasets) == 2
        super().__init__(datasets)
        self.camera_dataset = datasets[0]
        self.lidar_dataset = datasets[1]
        self.length = min(len(self.camera_dataset), len(self.lidar_dataset))

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        if idx >= len(self.camera_dataset) or idx >= len(self.lidar_dataset):
            return None
        cam_sample = self.camera_dataset[idx]
        lidar_sample = self.lidar_dataset[idx]
        if cam_sample is None or lidar_sample is None:
            return None
        return {
            **cam_sample,
            **lidar_sample,
            "label": 1  # or use some logic if label varies
        }

    def collater(self, samples):   
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        output = {}
        keys = samples[0].keys()
        
        for k in keys:
            values = [s[k] for s in samples]
            first = values[0]

            if isinstance(first, torch.Tensor):
                output[k] = torch.stack(values)
            elif isinstance(first, (int, float)):
                output[k] = torch.tensor(values)
            else:
                output[k] = values

        return output

# creates train/val/test splits from config
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


import os
import argparse
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True, help="Path to the nuscenes_dataset/train/tiny folder")
    args = parser.parse_args()

    camera_root = os.path.join(args.base_path, "camera_front")
    lidar_root = os.path.join(args.base_path, "lidar")

    cam_files, lidar_files = get_matched_files(camera_root, lidar_root)
    print(f"[INFO] Matched {len(cam_files)} camera and {len(lidar_files)} lidar files")

    cam_dataset = TorchCameraDataset(cam_files)
    lidar_dataset = TorchLidarDataset(lidar_files)
    print(f"Camera dataset length: {len(cam_dataset)}")
    print(f"Lidar dataset length: {len(lidar_dataset)}")

    dataset = WrappedConcatDataset([cam_dataset, lidar_dataset])
    print(f"[DEBUG] Total paired dataset length: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collater)

    total = 0
    skipped = 0

    for i, batch in enumerate(loader):
        total += 1
        if not batch:
            print(f"[WARNING] Skipped batch {i} — empty dict returned.")
            skipped += 1
            continue

        print(f"[BATCH {i}]")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape {v.shape}")
            else:
                print(f"{k}: {v}")
            # break
        break
    print(f"\n[SUMMARY] Total batches: {total}, Skipped (empty): {skipped}, Valid: {total - skipped}")


if __name__ == "__main__":
    main()
    
# run the main with 
# python lavis/datasets/nuscenes.py --base-path lavis/my_datasets/nuscenes_dataset/train