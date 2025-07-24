"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from PIL import Image

# class BaseDataset(Dataset):
    
#     # def __init__(
#     #     self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
#     # ):
#     #     """
#     #     vis_root (string): Root directory of images (e.g. coco/images/)
#     #     ann_root (string): directory to store the annotation file
#     #     """
#     #     # print(">>> [DEBUG] BaseDataset __init__ called")
#     #     self.vis_root = vis_root
#     #     self.annotation = []
#     #     for ann_path in ann_paths:
#     #         if any(ext in ann_path for ext in ['csv', 'tsv']):
#     #             df = pd.read_csv(ann_path)
#     #             self.annotation.extend(df.to_dict(orient="records"))
                
#     #         elif 'jsonl' in ann_path:
#     #             with open(ann_path, "r") as f:
#     #                 self.annotation.extend([json.loads(line) for line in f])

#     #         else:
#     #             with open(ann_path, "r") as f:
#     #                 loaded = json.load(f)
#     #                 if isinstance(loaded, list):
#     #                     self.annotation.extend(loaded)
#     #                 elif isinstance(loaded, dict):
#     #                    self.annotation.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v} for k, v in loaded.items()])


#     #     self.vis_processor = vis_processor
#     #     self.text_processor = text_processor

#     #     self._add_instance_ids()


#     def __init__(
#         self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None
#     ):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_paths (list or None): List of annotation file paths or None if not used.
#         """
#         self.vis_root = vis_root
#         self.annotation = []

#         if ann_paths is not None:
#             for ann_path in ann_paths:
#                 if any(ext in ann_path for ext in ['csv', 'tsv']):
#                     df = pd.read_csv(ann_path)
#                     self.annotation.extend(df.to_dict(orient="records"))

#                 elif 'jsonl' in ann_path:
#                     with open(ann_path, "r") as f:
#                         self.annotation.extend([json.loads(line) for line in f])

#                 else:  # assume JSON
#                     with open(ann_path, "r") as f:
#                         loaded = json.load(f)
#                         if isinstance(loaded, list):
#                             self.annotation.extend(loaded)
#                         elif isinstance(loaded, dict):
#                             self.annotation.extend([
#                                 {"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v}
#                                 for k, v in loaded.items()
#                             ])

#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         self._add_instance_ids()

#     def __len__(self):
#         return len(self.annotation)

#     def __getitem__(self, index):
#         ann = self.annotation[index]

#         # Resolve full image path
#         image_path = os.path.join(self.vis_root, ann["image"])

#         # Load + transform image
#         # image = self.vis_processor(image_path)

#         # print(f"[DEBUG __getitem__] index={index}, image_path={image_path}")

#         try:
#             image = Image.open(image_path).convert("RGB")
#         except Exception as e:
#             print(f"[ERROR loading image] {e}")
#             return None

#         image = self.vis_processor(image)


#         # Get caption (string or list)
#         caption = ann["caption"]
#         if isinstance(caption, list):
#             caption = caption[0]  # Pick the first caption

#         text_input = self.text_processor(caption)

#         return {
#             "image": image,
#             "text_input": text_input,
#         }


#     def collater(self, samples):
#         print("+++ DEBUG COLLATER IS CALLED +++")  # 🔥 guaranteed print
#         print(f"[DEBUG] Number of samples passed in: {len(samples)}")

#         samples = [s for s in samples if s is not None]

#         if not samples:
#             print("[DEBUG] All samples were None!")
#             return {}

#         collated_dict = {}
#         keys = samples[0].keys()
#         for k in keys:
#             values = [sample[k] for sample in samples if k in sample]
#             collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values

#         return collated_dict



#     def set_processors(self, vis_processor, text_processor):
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#     def _add_instance_ids(self, key="instance_id"):
#         for idx, ann in enumerate(self.annotation):
#             ann[key] = str(idx)

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # 🧹 Filter out None values early
        samples = [s for s in samples if s is not None]

        if not samples:
            return {}

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)

