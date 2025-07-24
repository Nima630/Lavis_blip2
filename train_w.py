import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lavis.datasets.waymo import WaymoDatasetBuilder
# from lavis.datasets.waymo_pointNet import WaymoDatasetBuilder
from lavis.models.blip2_models.blip2_qformer_w import Blip2Qformer  # ✅ your custom model
from lavis.tasks.base_task_w import BaseTask
from lavis.common.config_w import Config  # ✅ your new config_w.py
from lavis.common.dist_utils import get_rank, init_distributed_mode
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
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    job_id = now()
    print("[TRACE] Starting train_w.py")
    args = parse_args()
    cfg = Config(args)


    init_distributed_mode(cfg.run_cfg)
    cfg.run_cfg.gpu = int(os.environ.get("LOCAL_RANK", 0))
    setup_seeds(cfg)
    setup_logger()

    # Manually register the paths previously set in lavis/tasks/__init__.py
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lavis")


    registry.register_path("library_root", root_dir)
    repo_root = os.path.join(root_dir, "..")
    registry.register_path("repo_root", repo_root)


    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

    print(cfg.run_cfg)
    print("cfg.run_cfg")

    # Use unified WaymoDatasetBuilder
    task = BaseTask()
    builder = WaymoDatasetBuilder(cfg)


    datasets = builder.build_datasets()

    print("passed check-------------------------------------")
    train_dataset = datasets.get("train")

    if train_dataset is None:
        raise ValueError("Train dataset not found in builder output.")

    print(">>> [DEBUG] Loaded train dataset with", len(train_dataset), "samples")



    model = Blip2Qformer.from_config(cfg.model_cfg)
    if not cfg.run_cfg.amp:
        model = model.float()
    runner = RunnerBaseW(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check your input paths or data content.")

    runner.train()


if __name__ == "__main__":
    main()






















