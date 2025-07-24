import pickle

with open('/home/draiman/Desktop/Nima/BEVFormer_dev/data/nuscenes/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
# print(data[0])



first_3_keys = list(data.keys())[:10]
print("🔑 First 3 keys:", first_3_keys)



infos = data['infos']
print(type(infos))
print("Number of items in 'infos':", len(infos))
print("First item:", infos[0].keys())
# print("First item:", infos[0]['lidar_path'])
print("First item:", infos[0]['token'])


# metadata = data['metadata']
# print(type(metadata))
# print("Number of items in 'metadata':", len(infos))
# print("First item:", metadata)




# import os
# import pickle
# import shutil

# # --- Paths ---
# pkl_path = '/home/draiman/Desktop/Nima/BEVFormer_dev/data/nuscenes/nuscenes_infos_temporal_val.pkl'
# features_dir = '/home/draiman/Desktop/dataset/features_bevformer_dev/lidar'
# val_output_dir = '/home/draiman/Desktop/dataset/features_bevformer_dev/lidar_val'

# os.makedirs(val_output_dir, exist_ok=True)

# # --- Load the .pkl ---
# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)

# infos = data['infos']
# print(f"🔍 Found {len(infos)} validation tokens in .pkl")

# # --- Move matching .pt files ---
# count = 0
# for info in infos:
#     token = info['token']
#     src_path = os.path.join(features_dir, f"{token}.pt")
#     dst_path = os.path.join(val_output_dir, f"{token}.pt")

#     if os.path.exists(src_path):
#         shutil.move(src_path, dst_path)
#         count += 1
#     else:
#         print(f"⚠️ Not found: {token}.pt")

# print(f"✅ Moved {count}/{len(infos)} files to {val_output_dir}")
