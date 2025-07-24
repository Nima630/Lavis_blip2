



import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lavis.datasets.nuscenes_multi_cam import NuScenesDatasetBuilder
 
from lavis.models.blip2_qformer_n_multi_cam import Blip2Qformer 
from lavis.common.config_w import Config  
from lavis.common.dist_utils import is_main_process
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# Optional: if you're keeping runners for now
from lavis.runners.runner_base_w import RunnerBaseW
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from lavis.common.registry import registry
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override options in YAML with key=value",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False # fastest way to run yeild results that are a little random. So False means slower but more consistent
    cudnn.deterministic = True

def main():
    job_id = now()
    print("[TRACE] Starting train_w.py")
    args = parse_args()
    cfg = Config(args)


    setup_seeds(cfg)
    setup_logger()

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lavis")


    registry.register_path("library_root", root_dir)
    repo_root = os.path.join(root_dir, "..")
    registry.register_path("repo_root", repo_root)


    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

    print(cfg.run_cfg)
    print("cfg.run_cfg")

    builder = NuScenesDatasetBuilder(cfg)
    datasets = builder.build_datasets()

    train_dataset = datasets.get("train")

    if train_dataset is None:
        raise ValueError("Train dataset not found in builder output.")
    print(">>> [DEBUG] Loaded train dataset with", len(train_dataset), "samples")



    model = Blip2Qformer.from_config(cfg.model_cfg)
    print(">>> [DEBUG] Model loaded from config")   
    # print(model)
    with open("blip2qformer_model_structure.txt", "w") as f:
        f.write(str(model))
    # exit()
    if not cfg.run_cfg.amp:
        model = model.float()


    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

    # exit()    
    runner = RunnerBaseW(cfg=cfg, job_id=job_id, model=model, datasets=datasets)
    print("train_dataset", train_dataset)
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check your input paths or data content.")

    torch.cuda.empty_cache()
    runner.train()


if __name__ == "__main__":
    main()
