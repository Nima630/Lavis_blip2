"""
Simplified RunnerBaseW for non-distributed 2-GPU training using DataParallel
"""
# from lavis.runners.log_utils import close_writer
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
from lavis.runners import log_utils
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import datetime
import json
import logging
import os
import time
from pathlib import Path
from lavis.common.logger import MetricLogger, SmoothedValue
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ChainDataset
from lavis.datasets.data_utils import prepare_sample
import webdataset as wds
from torch import autocast
from lavis.common.dist_utils import (
    download_cached_file,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets
from lavis.datasets.dataloader_utils import IterLoader, MultiIterLoader, PrefetchLoader_
# import torch
# import torch.nn.functional as F
# from collections import defaultdict

class RunnerBaseW:
    def __init__(self, cfg, model, datasets, job_id):
        self.config = cfg
        self.datasets = datasets
        self._model = model
        self.job_id = job_id
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self.global_step = 0
        self.start_epoch = 0
        self.setup_output_dir()
        # log_utils.set_config(cfg)
        self.z_thresh = 0.913159 - 2 * 0.100871


    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            if self._wrapped_model is None:
                self._wrapped_model = nn.DataParallel(self._model).cuda()

        # print(f"[DEBUG] Using {torch.cuda.device_count()} GPUs with DataParallel.")        
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            weight_decay = self.config.run_cfg.get("weight_decay", 0.05)
            optim_params = self._model.get_optimizer_params(weight_decay, lr_scale)

            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        if self.config.run_cfg.get("amp", False):
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                min_lr=self.min_lr,
                init_lr=self.init_lr,
                decay_rate=self.config.run_cfg.get("lr_decay_rate", None),
                warmup_start_lr=self.config.run_cfg.get("warmup_lr", -1),
                warmup_steps=self.config.run_cfg.get("warmup_steps", 0),
            )
        return self._lr_sched


    @property
    def train_loader(self):
        return self.dataloaders["train"]



    @property
    def dataloaders(self):
        if self._dataloaders is None:
            datasets = [self.datasets[split] for split in sorted(self.datasets)]
            is_trains = [split in self.train_splits for split in sorted(self.datasets)]
            batch_sizes = [
                self.config.run_cfg.batch_size_train if is_train else self.config.run_cfg.batch_size_eval
                for is_train in is_trains
            ]
            collate_fns = [
                [getattr(d, "collater", None) for d in dataset] if isinstance(dataset, (list, tuple))
                else getattr(dataset, "collater", None)
                for dataset in datasets
            ]

            dataloaders = self.create_loaders(
                datasets,
                self.config.run_cfg.num_workers,
                batch_sizes,
                is_trains,
                collate_fns,
            )
            self._dataloaders = {k: v for k, v in zip(sorted(self.datasets), dataloaders)}
        return self._dataloaders


    def create_loaders(self, datasets, num_workers, batch_sizes, is_trains, collate_fns, dataset_ratios=None):
        # def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
        # is_train = False
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn, split_name):
            print(f"[DEBUG] split={split_name}, batch_size={bsz}, drop_last={is_train}")

            if isinstance(dataset, ChainDataset) or isinstance(dataset, wds.DataPipeline):
                loader = iter(DataLoader(dataset, batch_size=bsz, num_workers=num_workers, pin_memory=True))
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=is_train,
                    collate_fn=collate_fn,
                    drop_last= is_train,
                )
                loader = PrefetchLoader_(loader)
                # if is_train:
                #     loader = IterLoader(loader, use_distributed=False)

                if is_train and split_name != "test":
                    print(f"Wrapping IterLoader for split {split_name}")
                    loader = IterLoader(loader, use_distributed=False)
                else:
                    print(f"No IterLoader for split {split_name}")


            return loader

        loaders = []
        # for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
        for split_name, dataset, bsz, is_train, collate_fn in zip(sorted(self.datasets), datasets, batch_sizes, is_trains, collate_fns):
    
            # if isinstance(dataset, (list, tuple)):
            #     print(f"[DEBUG] Creating MultiIterLoader for dataset with {len(dataset)} components.")
            #     loader = MultiIterLoader(
            #         loaders=[
            #             _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
            #             for i, d in enumerate(dataset)
            #         ],
            #         ratios=dataset_ratios,
            #     )
            # else:
            print(f"[DEBUG] Creating single DataLoader for dataset {dataset}.")
            
            loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn, split_name)
            loaders.append(loader)
        return loaders



    # Other properties remain the same 
    @property
    def cuda_enabled(self): return self.device.type == "cuda"
    @property
    def max_epoch(self): return int(self.config.run_cfg.max_epoch)
    @property
    def log_freq(self): return int(self.config.run_cfg.get("log_freq", 50))
    @property
    def save_freq(self): return int(self.config.run_cfg.get("save_freq", 5))
    @property
    def val_freq(self): return int(self.config.run_cfg.get("val_freq", 1))
    @property
    def save_last(self): return int(self.config.run_cfg.get("save_last", True))
    @property
    def init_lr(self): return float(self.config.run_cfg.init_lr)
    @property
    def min_lr(self): return float(self.config.run_cfg.min_lr)
    @property
    def accum_grad_iters(self): return int(self.config.run_cfg.get("accum_grad_iters", 1))
    @property
    def valid_splits(self): return self.config.run_cfg.get("valid_splits", [])
    @property
    def test_splits(self): return self.config.run_cfg.get("test_splits", [])
    @property
    def train_splits(self): return self.config.run_cfg.get("train_splits", [])
    @property
    def evaluate_only(self): return self.config.run_cfg.evaluate
    @property
    def resume_ckpt_path(self): return self.config.run_cfg.get("resume_ckpt_path", None)
    @property
    def use_dist_eval_sampler(self): return False

    def unwrap_dist_model(self, model):
        return model.module if isinstance(model, nn.DataParallel) else model

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        model_no_ddp = self.unwrap_dist_model(self.model)
        # state_dict = {
        #     k: v for k, v in model_no_ddp.state_dict().items() if v.requires_grad
        # }
        state_dict = model_no_ddp.state_dict()  #
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        path = os.path.join(self.output_dir, f"checkpoint_{'best' if is_best else cur_epoch}.pth")
        logging.info(f"Saving checkpoint at epoch {cur_epoch} to {path}.")
        torch.save(save_obj, path)

    def _reload_best_model(self, model):
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info(f"Loading checkpoint from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except:
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.unwrap_dist_model(self.model).load_state_dict(checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {f"{split_name}_{k}": v for k, v in stats.items()}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))
        self.result_dir = result_dir
        self.output_dir = output_dir





    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        # print("\n>>> [DEBUG] datasets_cfg-------------------------:", cfg.datasets_cfg)
        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        # print(">>> [DEBUG] Built datasets keys:", datasets.keys())
        return datasets




    def train(self):
        best_val_loss_so_far = float("inf")
        best_epoch = 0
        self.log_config()

        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if not self.evaluate_only:
                train_stats = self.train_epoch(cur_epoch)
                # close_writer()

                self.log_stats(train_stats, "train")

            if len(self.valid_splits) > 0 and (self.evaluate_only or cur_epoch % self.val_freq == 0):
                for split_name in self.valid_splits:
                    val_log = self.eval_epoch(split_name, cur_epoch)
                    if val_log and is_main_process():
                        current_val_loss = val_log["loss"]
                        if current_val_loss < best_val_loss_so_far and split_name == "val":
                            best_epoch, best_val_loss_so_far = cur_epoch, current_val_loss
                            if not self.evaluate_only:
                                self._save_checkpoint(cur_epoch, is_best=True)
                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)

            # if self.evaluate_only:
            #     break

            # Handle testing if in evaluation-only mode
            if self.evaluate_only and len(self.test_splits) > 0:
                for split_name in self.test_splits:
                    print(f"[DEBUG] Evaluating on test split: {split_name}")
                    test_log = self.eval_epoch(split_name, cur_epoch)
                    if test_log and is_main_process():
                        self.log_stats(test_log, split_name)

                #  Don't break until test is done
                break

        if not os.path.exists(os.path.join(self.output_dir, "checkpoint_best.pth")) and not self.evaluate_only:
            self._save_checkpoint(cur_epoch, is_best=True)


    def train_epoch(self, epoch):
        self.model.train()

        stats = self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(self.train_loader),
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            log_freq=self.log_freq,
            cuda_enabled=self.cuda_enabled,
            accum_grad_iters=self.accum_grad_iters,
        )

        stats = {f"{k}": v for k, v in stats.items()}
        stats["epoch"] = epoch
        return stats




    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, f"DataLoader for split {split_name} is None."
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        # self.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.evaluation(self.config, model, data_loader)
        # print("------------------------------------------------------------------------------------------------------------------------------")

        if results is not None:
            return self.after_evaluation(self.config, results, split_name, cur_epoch)
        return None



    def train_step(self, model, samples):
        output = model(samples)
        raw_loss = output["loss"]

        # Ensure it's a scalar (e.g. mean reduction)
        if raw_loss.ndim != 0:
            raw_loss = raw_loss.mean()

        loss_dict = {k: v.mean().item() if v.ndim > 0 else v.item() for k, v in output.items() if "loss" in k}
        return raw_loss, loss_dict



    def valid_step(self,config, model, samples):
        model.eval()
        samples = prepare_sample(samples, cuda_enabled=True)

        retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)
        use_shuffled = getattr(config.run_cfg, "use_shuffled_validation", False)
        use_amp = True

        with torch.no_grad():
            with autocast("cuda", enabled=use_amp):
                if retrieval_eval:
                    if use_shuffled:

                        outputs = model.forward_features_shuffled(samples, shuffle_prob=0.5)
                        return {"loss": None, "output": outputs}
                    else:
                    # Only extract embeddings for retrieval
                        print("------------------------forward_features-------------------------")
                        features = model.forward_features(samples)
                        rgb_feats = features["rgb_feats"]
                        lidar_feats = features["lidar_feats"]
                        # print("[DEBUG] rgb_feats type:", type(rgb_feats), "shape:", rgb_feats.shape if isinstance(rgb_feats, torch.Tensor) else "N/A")
                        # print("[DEBUG] lidar_feats type:", type(lidar_feats), "shape:", lidar_feats.shape if isinstance(lidar_feats, torch.Tensor) else "N/A")

                        return {"loss": None, "output": {"rgb_feats": rgb_feats, "lidar_feats": lidar_feats}}
                else:
                    # Use standard forward that returns loss
                    output = model(samples)
                    loss = output["loss"].item() if "loss" in output else None
                    return {"loss": loss, "output": output}




    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
        header = f"Train: data epoch: [{epoch}]"
        inner_epoch = epoch if start_iters is None else start_iters // iters_per_epoch
        if start_iters is not None:
            header += f"; inner epoch [{inner_epoch}]"
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            # samples = next(data_loader)
            if hasattr(data_loader, "next"):
                samples = data_loader.next()
            else:
                samples = next(data_loader)

    
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            if samples is None or samples.get("is_empty", False):
                continue
            
            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                    "global_step": (i+1)*(inner_epoch +1),    #self.global_step,
                }
            )
            
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            if i % 50 == 0: 
                print(f"[DEBUG] Step {i}, LR: {optimizer.param_groups[0]['lr']:.8f}")

            # samples
            bsz = samples['image'].size(0)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        # self.global_step += 1
        return {
            # k: "{:.3f}".format(meter.global_avg)
            k: ("{:.8f}".format(meter.global_avg) if k == "lr" else "{:.3f}".format(meter.global_avg))
            for k, meter in metric_logger.meters.items()
        }







# ================================================================
# Evaluation 
# ================================================================

    def compute_retrieval_metrics_from_outputs(self, output_list, top_k=(1, 5, 10)):
            rgb_feats_list = []
            lidar_feats_list = []
            cos_sim_list = []
            attack_flag_list = []
            for out in output_list:
                inner_output = out["output"]
                rgb_feats_list.append(inner_output["rgb_feats"])
                lidar_feats_list.append(inner_output["lidar_feats"])
                cos_sim_list.append(inner_output["cos_sim_diag"].cpu())
                attack_flag_list.append(inner_output["attack_flags"].cpu())

            rgb_feats = torch.cat(rgb_feats_list, dim=0)      # [B_total, N, D]
            lidar_feats = torch.cat(lidar_feats_list, dim=0)  # [B_total, N, D]

            rgb_flat = rgb_feats.mean(dim=1)      # [B_total, D]
            lidar_flat = lidar_feats.mean(dim=1)  # [B_total, D]

            sim_matrix = torch.matmul(rgb_flat, lidar_flat.T)  # [B, B]

            metrics = {}
            
            cos_sim_tensor = torch.cat(cos_sim_list, dim=0)        # [B_total]
            attack_flags_tensor = torch.cat(attack_flag_list, dim=0)  # [B_total]

            cos_sim = cos_sim_tensor.numpy()
            labels = attack_flags_tensor.numpy().astype(int)
            scores = 1 - cos_sim

            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_auc = auc(recall, precision)

            print(f"[EVAL] Precision-Recall AUC: {pr_auc:.4f}")
            metrics["pr_auc_cos_sim"] = pr_auc
                        
            plt.figure()
            plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (Cosine Similarity)")
            plt.legend()
            plt.grid(True)

            save_path = "./result/pr_curve_cos_sim.png"
            plt.savefig(save_path)
            print(f"[INFO] Saved PR curve to {save_path}")





            B = sim_matrix.size(0)

            # === Image-to-LiDAR ===
            for i in range(B):
                row = sim_matrix[i]
                topk_indices = torch.topk(row, k=max(top_k)).indices.tolist()
                rank_of_gt = (torch.argsort(row, descending=True) == i).nonzero(as_tuple=True)[0].item() + 1

            for k in top_k:
                hits = sum([i in torch.topk(sim_matrix[i], k=k).indices for i in range(B)])
                metrics[f"recall@{k}_rgb2lidar"] = hits / B

            ranks = [(torch.argsort(sim_matrix[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
            metrics["mrr_rgb2lidar"] = sum(1.0 / rank for rank in ranks) / B

            # === LiDAR-to-Image ===
            sim_matrix_T = sim_matrix.T
            for i in range(B):
                row = sim_matrix_T[i]
                topk_indices = torch.topk(row, k=max(top_k)).indices.tolist()
                rank_of_gt = (torch.argsort(row, descending=True) == i).nonzero(as_tuple=True)[0].item() + 1

            for k in top_k:
                hits = sum([i in torch.topk(sim_matrix_T[i], k=k).indices for i in range(B)])
                metrics[f"recall@{k}_lidar2rgb"] = hits / B

            ranks = [(torch.argsort(sim_matrix_T[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
            metrics["mrr_lidar2rgb"] = sum(1.0 / rank for rank in ranks) / B

            # === Contrastive loss (debug)
            temperature = 0.1
            sim_matrix = sim_matrix / temperature
            targets = torch.arange(B).to(sim_matrix.device)

            loss_i2t = torch.nn.functional.cross_entropy(sim_matrix, targets)
            loss_t2i = torch.nn.functional.cross_entropy(sim_matrix.T, targets)

            contrastive_loss = (loss_i2t + loss_t2i) / 2
            print(f"\n[DEBUG] loss_i2t: {loss_i2t}")
            print(f"\n[DEBUG] loss_t2i: {loss_t2i}")



            print(f"\n[DEBUG] contrastive_loss: {contrastive_loss.item():.4f}")
            metrics["contrastive_loss"] = contrastive_loss.item()
            return metrics



    def compute_batchwise_match_confidence_avg_by_batch(
        self,
        output_list,
        threshold: float = 0.5,
        use_model_temperature: bool = True,
        fixed_temperature: float = 0.1,):
        """
        Computes match accuracy/precision/recall/F1/confidence *per batch*, then averages the batch-level stats.
        Similar to compute_batchwise_retrieval_metrics, but for match classification.
        """

        print("\n[INFO] Starting batchwise-averaged confidence evaluation...")
        base_model = getattr(self, "model", None)
        if base_model is not None and hasattr(base_model, "module"):
            base_model = base_model.module

        batch_metrics = []
        total_samples = 0

        for bidx, out in enumerate(output_list):
            print(f"\n[INFO] Processing batch {bidx}...")

            # ---- Similarity ----
            # if "diagnostics" in out["output"] and "similarity" in out["output"]["diagnostics"]:
            #     sim_raw = out["output"]["diagnostics"]["similarity"]["matrix"]
            #     print("[DEBUG] Using provided similarity matrix.")
            # else:
            print("[DEBUG] Computing similarity from features.")
            rgb_feats = out["output"]["rgb_feats"]
            lidar_feats = out["output"]["lidar_feats"]
            rgb_avg = rgb_feats.mean(dim=1)
            lidar_avg = lidar_feats.mean(dim=1)
            sim_raw = rgb_avg @ lidar_avg.T

            B = sim_raw.size(0)
            targets = torch.arange(B, device=sim_raw.device)
            total_samples += B

            # ---- Logits ----
            if use_model_temperature and (base_model is not None) and hasattr(base_model, "log_temp"):
                logits = sim_raw * torch.exp(base_model.log_temp)
                print(f"[DEBUG] Using model log_temp = {base_model.log_temp.item():.4f}")
            else:
                logits = sim_raw / fixed_temperature
                print(f"[DEBUG] Using fixed temperature = {fixed_temperature}")

            # ---- Softmax Probs ----
            probs_row = F.softmax(logits, dim=1)
            diag_prob_row = probs_row[torch.arange(B), targets]

            probs_col = F.softmax(logits.T, dim=1)
            diag_prob_col = probs_col[torch.arange(B), targets]

            pred_row = (diag_prob_row > threshold).int()
            pred_col = (diag_prob_col > threshold).int()
            gt = torch.ones_like(pred_row)

            # ---- Stats per direction ----
            acc_r2l, prec_r2l, rec_r2l, f1_r2l, *_ = _bin_stats(pred_row, gt)
            acc_l2r, prec_l2r, rec_l2r, f1_l2r, *_ = _bin_stats(pred_col, gt)

            batch_metrics.append({
                "acc_r2l": acc_r2l,
                "prec_r2l": prec_r2l,
                "rec_r2l": rec_r2l,
                "f1_r2l": f1_r2l,
                "conf_r2l": diag_prob_row.mean().item(),
                "conf_r2l_std": diag_prob_row.std().item(),

                "acc_l2r": acc_l2r,
                "prec_l2r": prec_l2r,
                "rec_l2r": rec_l2r,
                "f1_l2r": f1_l2r,
                "conf_l2r": diag_prob_col.mean().item(),
                "conf_l2r_std": diag_prob_col.std().item(),

                "samples": B,
            })

            print(f"[Batch {bidx}] RGB→LiDAR acc={acc_r2l:.4f}, conf_avg={diag_prob_row.mean().item():.4f}")
            print(f"[Batch {bidx}] LiDAR→RGB acc={acc_l2r:.4f}, conf_avg={diag_prob_col.mean().item():.4f}")

        # ---- Average all batch stats ----
        avg = lambda key: sum(d[key] for d in batch_metrics) / len(batch_metrics)
        metrics = {
            "match_thr": threshold,
            "rgb2lidar_match_acc_batch": avg("acc_r2l"),
            "rgb2lidar_match_prec_batch": avg("prec_r2l"),
            "rgb2lidar_match_rec_batch": avg("rec_r2l"),
            "rgb2lidar_match_f1_batch": avg("f1_r2l"),
            "rgb2lidar_conf_avg_batch": avg("conf_r2l"),
            "rgb2lidar_conf_std_batch": avg("conf_r2l_std"),

            "lidar2rgb_match_acc_batch": avg("acc_l2r"),
            "lidar2rgb_match_prec_batch": avg("prec_l2r"),
            "lidar2rgb_match_rec_batch": avg("rec_l2r"),
            "lidar2rgb_match_f1_batch": avg("f1_l2r"),
            "lidar2rgb_conf_avg_batch": avg("conf_l2r"),
            "lidar2rgb_conf_std_batch": avg("conf_l2r_std"),

            "num_batches_eval": len(batch_metrics),
            "num_samples_eval": total_samples,
        }

        print(f"\n[INFO] Batch-averaged confidence evaluation complete over {total_samples} samples.")
        print(f"[INFO] Mean RGB→LiDAR acc: {metrics['rgb2lidar_match_acc_batch']:.4f}")
        print(f"[INFO] Mean LiDAR→RGB acc: {metrics['lidar2rgb_match_acc_batch']:.4f}")

        return metrics



    @torch.no_grad()
    def evaluation(self, config, model, data_loader, cuda_enabled=True):
        print("\n[DEBUG] Running EVALUATION")
        header = "Evaluation"
        print_freq = 10

        total_loss = 0.0
        num_batches = 0
        all_outputs = []

        retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)

        # Print actual dataset length
        print(f"[DEBUG] Dataset size: {len(data_loader.dataset)}")
        
        for i, samples in enumerate(data_loader):
            # if retrieval_eval and i >= 10:
            #     break 
            if i % print_freq == 0:
                print(f"[DEBUG] {header} [{i}/{len(data_loader)}]")

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(config, model=model, samples=samples)
            if retrieval_eval:
                all_outputs.append(eval_output)  # raw features
            else:
                loss = eval_output.get("loss", None)
                output = eval_output.get("output", None)
                if loss is not None:
                    total_loss += loss
                all_outputs.append(output)

            num_batches += 1

        print(f"[DEBUG] Completed batches: {num_batches}")

        if retrieval_eval:
            # print("-----------------------------------------retrieval_eval---------------------------------------------------------------")
            return {
                "output": all_outputs  # for computing Recall@K etc.
            }
        else:
            avg_loss = total_loss / max(num_batches, 1)
            return {
                "loss": avg_loss,
                "output": all_outputs,
            }


    def after_evaluation(self, config, val_result, split_name, epoch):
        # print("------------------------------------------------------------------------------------------------------------------------------")

        """
        Process the results after evaluation.

        Args:
            val_result (dict): a dict containing evaluation outputs (e.g., losses, metrics, retrieval stats).
            split_name (str): name of the split evaluated.
            epoch (int or str): epoch number or "best".

        Returns:
            dict: stats to log (should contain "loss" key at least, or retrieval metrics).
        """
        # print("[AFTER_EVAL] Received val_result:", val_result)

        stats = {
            "epoch": epoch,
            "split": split_name,
        }

        if isinstance(val_result, dict):
            # Log loss if available
            if "loss" in val_result:
                stats["loss"] = val_result["loss"]

            # Log general metrics if present
            if "metrics" in val_result:
                stats.update(val_result["metrics"])

            # If retrieval evaluation results were added
            if "output" in val_result and config.run_cfg.get("retrieval_eval", False):
                retrieval_metrics = self.compute_retrieval_metrics_from_outputs(val_result["output"])  # for all samples 
                # retrieval_metrics = self.compute_batchwise_retrieval_metrics( # for bath by batch, then average batches 
                #     val_result["output"],
                #     top_k=(1, 5, 10),
                #     use_model_temperature=True,   # same as training
                #     fixed_temperature=0.1
                # )

                # retrieval_metrics = self.compute_batchwise_match_confidence_avg_by_batch(
                #     output_list=val_result["output"],
                #     threshold=0.5,
                #     use_model_temperature=True,
                #     fixed_temperature=0.1,
                #     )
                stats.update(retrieval_metrics)

            if "retrieval" in val_result:
                retrieval_metrics = val_result["retrieval"]
                stats.update(retrieval_metrics)

        print("[AFTER_EVAL] Returning stats:", stats)
        return stats


def _bin_stats(preds, gts):
    # print("------------------------------------------------------------------------------------------------------------------------------")
    """preds, gts: 1D tensors of 0/1."""
    tp = ((preds == 1) & (gts == 1)).sum().item()
    fp = ((preds == 1) & (gts == 0)).sum().item()
    fn = ((preds == 0) & (gts == 1)).sum().item()
    tn = ((preds == 0) & (gts == 0)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"[STATS] TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"[STATS] Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    return acc, prec, rec, f1, tp, fp, fn, tn

















