


import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def find_camera_name(filename: str) -> str:
    """Heuristic to extract camera name like 'CAM_FRONT' from a filename."""
    parts = filename.split('__')
    if len(parts) > 1:
        return parts[1]
    return "UNKNOWN"

# def serialize_array(arr: list) -> str:
#     """Convert a list or nested list (e.g., for matrices) into a semicolon-separated string."""
#     return ';'.join(map(str, sum(arr, []) if isinstance(arr[0], list) else arr))


import numpy as np
from typing import List, Any
# ... (rest of imports)

def serialize_array(arr: List[Any]) -> str:
    """
    Converts a nested list/matrix (like cam_intrinsic) or a 1D list 
    into a flat NumPy array, and then serializes its float elements 
    into a semicolon-separated string.
    
    This fixes the issue where NumPy's string representation was used instead of 
    individual values.
    """
    # 1. Force conversion to NumPy array of float64
    np_arr = np.array(arr, dtype=np.float64)
    # 2. Flatten the array
    flat_arr = np_arr.flatten()
    # 3. Convert elements to strings and join them using semicolon
    return ';'.join(map(str, flat_arr))

# Ensure you use this corrected version in your main processing loop:
# record = {
#     # ... other fields
#     "cam_intrinsic": serialize_array(cam_info["cam_intrinsic"]),
#     "sensor2ego_rotation": serialize_array(cam_info["sensor2ego_rotation"]),
#     "sensor2ego_translation": serialize_array(cam_info["sensor2ego_translation"]),
# }




def main(args):
    print("Loading original nuScenes data files...")
    sample_data = load_json(args.sample_data_json)
    infos_blob = load_pickle(args.infos_pkl)
    infos = infos_blob["infos"]
    print("Files loaded. Building a lookup map for calibration data...")

    # Create a fast lookup map from sample_data_token -> cam_info
    # This avoids repeatedly searching the large 'infos' list.
    token_to_cam_info = {}
    for sample_info in tqdm(infos, desc="Indexing calibration info"):
        for cam_name, cam_details in sample_info["cams"].items():
            sdt = cam_details.get("sample_data_token")
            if sdt:
                token_to_cam_info[sdt] = cam_details

    print(f"Map created with {len(token_to_cam_info)} entries.")
    
    # Process all samples and collect data
    records = []
    for entry in tqdm(sample_data, desc="Processing samples"):
        sample_token = entry.get("sample_token")
        sample_data_token = entry.get("token")
        filename = entry.get("filename", "")

        if not all([sample_token, sample_data_token, filename]):
            continue

        camera_name = find_camera_name(filename)
        cam_info = token_to_cam_info.get(sample_data_token)

        if cam_info:
            record = {
                "sample_token": sample_token,
                "camera_name": camera_name,
                "sample_data_token": sample_data_token,
                "cam_intrinsic": serialize_array(cam_info["cam_intrinsic"]),
                "sensor2ego_rotation": serialize_array(cam_info["sensor2ego_rotation"]),
                "sensor2ego_translation": serialize_array(cam_info["sensor2ego_translation"]),
            }
            records.append(record)

    # Convert to a DataFrame and save as CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Successfully created metadata file with {len(df)} records.")
    print(f"Saved to: {args.output_csv}")
    print("\nFirst 5 rows of the new file:")
    print(df.head())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create a consolidated metadata CSV from nuScenes data.")
    p.add_argument("--sample_data_json", type=str, required=True, help="Path to sample_data.json")
    p.add_argument("--infos_pkl", type=str, required=True, help="Path to nuscenes_infos_temporal_val.pkl")
    p.add_argument("--output_csv", type=str, default="nuscenes_metadata.csv", help="Path for the output CSV file.")
    
    # Example usage:
    # python create_metadata_file.py \
    #   --sample_data_json /path/to/v1.0-trainval/sample_data.json \
    #   --infos_pkl /path/to/nuscenes_infos_temporal_val.pkl
    
    # Using your hardcoded paths for demonstration if you run without args
    try:
        args = p.parse_args()
    except SystemExit:
        print("Running with default hardcoded paths for demonstration.")
        from types import SimpleNamespace
        args = SimpleNamespace(
            sample_data_json="/home/draiman/Desktop/datasets/nuscenes/v1.0-trainval/sample_data.json",
            
            #  infos_pkl="/home/draiman/Desktop/datasets/nuscenes/nuscenes_infos_temporal_val.pkl",
            #  output_csv="nusc_val_metadata.csv"
            infos_pkl="/home/draiman/Desktop/datasets/nuscenes/nuscenes_infos_temporal_train.pkl",
             output_csv="nusc_train_metadata.csv"
            #  sample_data_json="/home/draiman/Desktop/datasets/nuscenes_pseudo_test/v1.0-pseudo-test/sample_data.json",
            #  infos_pkl="/home/draiman/Desktop/datasets/nuscenes_pseudo_test/nuscenes_infos_temporal_test.pkl",
            #  output_csv="nusc_pseudo_test_metadata.csv"


        )
    main(args)











# just to see the structure of the pseudo-test pkl
# import pickle
# from pathlib import Path

# # ===========================================================
# # ABSOLUTE PATH TO YOUR pseudo-test PKL:
# # ===========================================================
# INFOS_PATH = Path("/home/draiman/Desktop/datasets/nuscenes_pseudo_test/nuscenes_infos_temporal_test.pkl")

# print(f"[INFO] Loading infos from:\n{INFOS_PATH}\n")

# if not INFOS_PATH.exists():
#     raise FileNotFoundError(f"[ERROR] File does not exist: {INFOS_PATH}")

# # ===========================================================
# # Load PKL
# # ===========================================================
# data = pickle.load(open(INFOS_PATH, "rb"))

# print("Top-level keys in the PKL:")
# print(list(data.keys()))
# print("-" * 50)

# # ===========================================================
# # Check "infos" key exists
# # ===========================================================
# if "infos" not in data:
#     raise KeyError("[ERROR] 'infos' key not found in PKL. This PKL cannot be used for metadata.")

# infos = data["infos"]
# print(f"[INFO] Number of infos entries: {len(infos)}")

# # ===========================================================
# # Inspect the first few infos
# # ===========================================================
# print("\n[INFO] Checking first 3 entries...\n")

# for i in range(min(3, len(infos))):
#     info = infos[i]
#     print(f"---- INFO {i} ----")

#     # Check for cams key
#     if "cams" not in info:
#         print("[ERROR] 'cams' key missing in this info entry!")
#         continue

#     cams = info["cams"]
#     print("Available cameras:", list(cams.keys()))

#     # Inspect each camera
#     for cam_name, cam_data in cams.items():
#         print(f"\nCamera: {cam_name}")

#         # Check mandatory fields
#         sd_token = cam_data.get("sample_data_token")
#         print("  sample_data_token:", sd_token)

#         ci = cam_data.get("cam_intrinsic")
#         print("  cam_intrinsic:", "FOUND" if ci is not None else "MISSING")

#         r = cam_data.get("sensor2ego_rotation")
#         print("  sensor2ego_rotation:", "FOUND" if r is not None else "MISSING")

#         t = cam_data.get("sensor2ego_translation")
#         print("  sensor2ego_translation:", "FOUND" if t is not None else "MISSING")

#     print("-" * 50)

# print("\n[DONE] Inspection complete.\n")
