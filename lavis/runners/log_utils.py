

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F

config = None
writer = None

def set_config(cfg):
    global config, writer
    config = cfg
    run_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.run_cfg.output_dir
    writer = SummaryWriter(log_dir=f"{output_dir}/{run_name}")
    print("✅ [log_utils] Config set and writer initialized.")
    print(f"{output_dir}/{run_name}")
    
    writer = SummaryWriter(log_dir=f"lavis/output/BLIP2/CAM_FRONT_4_changes/{run_name}")

# config = None

# run_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# writer = SummaryWriter(log_dir=f"lavis/output/BLIP2/CAM_FRONT_4_changes/{run_name}")




def log_alignment_stats(sim_matrix, contrastive_loss, diversity_loss, total_loss,
                        step=None, rgb_feats=None, lidar_feats=None, model=None,
                        param_stats=None):

    bs = sim_matrix.size(0)
    device = sim_matrix.device

    stats = {
        "loss/total": total_loss.item(),
        "loss/contrastive": contrastive_loss.item(),
        "loss/diversity": diversity_loss.item(),
        "sim/diag_mean": sim_matrix.diag().mean().item(),
        "sim/off_diag_mean": sim_matrix[~torch.eye(bs, dtype=bool, device=device)].mean().item(),
        "retrieval/rgb2lidar_acc": (sim_matrix.argmax(dim=1) == torch.arange(bs, device=device)).float().mean().item(),
        "retrieval/lidar2rgb_acc": (sim_matrix.argmax(dim=0) == torch.arange(bs, device=device)).float().mean().item(),
    }

    if rgb_feats is not None and lidar_feats is not None:
        stats.update({
            "features/rgb_std": rgb_feats.std().item(),
            "features/lidar_std": lidar_feats.std().item(),
            "features/cross_modal_sim": (F.normalize(rgb_feats, dim=1) * F.normalize(lidar_feats, dim=1)).sum(1).mean().item(),
        })

    if param_stats is not None:
        stats.update(param_stats)


    if step % 50 == 0:
        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and ("proj" in name or "Qformer" in name):
                    stats[f"grad/{name.replace('.','/')}"] = param.grad.abs().mean().item()

            if hasattr(model, 'vision_proj') and hasattr(model.vision_proj, 'weight'):
                stats["proj/vision_weight_norm"] = model.vision_proj.weight.norm().item()
            if hasattr(model, 'lidar_proj') and hasattr(model.lidar_proj, 'weight'):
                stats["proj/lidar_weight_norm"] = model.lidar_proj.weight.norm().item()

            for k, v in stats.items():
                writer.add_scalar(k, v, step)

    # Console output
    
        print(f"\n[Step {step}] Loss Breakdown:")
        print(f"  Total: {total_loss.item():.4f} | Contrastive: {contrastive_loss.item():.4f} | Diversity: {diversity_loss.item():.4f}")

        if rgb_feats is not None:
            print(f"\nFeature Statistics:")
            print(f"  RGB STD: {stats['features/rgb_std']:.4f} | LiDAR STD: {stats['features/lidar_std']:.4f}")
            print(f"  Cross-Modal Sim: {stats['features/cross_modal_sim']:.4f}")

        print(f"\nSimilarity Matrix:")
        print(f"  Diag Mean: {stats['sim/diag_mean']:.4f} | Off-Diag Mean: {stats['sim/off_diag_mean']:.4f}")
        print(f"  RGB→LiDAR Acc: {stats['retrieval/rgb2lidar_acc']:.2%} | LiDAR→RGB Acc: {stats['retrieval/lidar2rgb_acc']:.2%}")

        if param_stats and 'params/log_temp' in param_stats:
            print(f"\nParameters:")
            print(f"  Temperature: {param_stats['params/log_temp']:.4f}")

        grad_keys = [k for k in stats if k.startswith('grad/')]
        if grad_keys:
            print("\nGradient Flow:")
            for key in grad_keys[:3]:
                print(f"  {key[5:30]:25} {stats[key]:.2e}")
    
    # writer.flush()

def close_writer():
    writer.flush()
    writer.close()





















# def log_alignment_stats(sim_matrix, loss, step=None, prefix="SIM", 
#                        rgb_feats=None, lidar_feats=None, model=None):
#     """
#     Enhanced logging with new diagnostics
#     Args:
#         sim_matrix: [B,B] similarity matrix
#         loss: Total loss value
#         step: Global step
#         prefix: TensorBoard tag prefix
#         rgb_feats: [B,D] RGB features (for new metrics)
#         lidar_feats: [B,D] LiDAR features (for new metrics)
#         model: Model reference (for gradient/projection checks)
#     """
#     bs = sim_matrix.size(0)
#     device = sim_matrix.device
#     targets = torch.arange(bs).to(device)

#     # --- Original Metrics ---
#     diag_sim = sim_matrix.diag()
#     off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=device)]
    
#     prob_rgb2lidar = F.softmax(sim_matrix, dim=1)
#     prob_lidar2rgb = F.softmax(sim_matrix.T, dim=1)
    
#     stats = {
#         # Original metrics
#         f"{prefix}/loss": loss.item(),
#         f"{prefix}/diag_sim_mean": diag_sim.mean().item(),
#         f"{prefix}/off_diag_sim_mean": off_diag_sim.mean().item(),
#         f"{prefix}/rgb2lidar_acc": (sim_matrix.argmax(dim=1) == targets).float().mean().item(),
#         f"{prefix}/lidar2rgb_acc": (sim_matrix.argmax(dim=0) == targets).float().mean().item(),
        
#         # New feature statistics
#         f"{prefix}/rgb_feat_std": rgb_feats.std().item(),
#         f"{prefix}/lidar_feat_std": lidar_feats.std().item(),
#         f"{prefix}/cross_modal_sim": (F.normalize(rgb_feats,dim=1) * F.normalize(lidar_feats,dim=1)).sum(1).mean().item(),
#     }

#     # --- Gradient Monitoring ---
#     if model is not None:
#         grad_stats = {}
#         for name, param in model.named_parameters():
#             if param.grad is not None and ("proj" in name or "Qformer" in name):
#                 grad_stats[f"{prefix}/grad/{name.replace('.','/')}"] = param.grad.abs().mean().item()
#         stats.update(grad_stats)

#     # --- Projection Layer Health ---
#     if hasattr(model, 'vision_proj'):
#         stats.update({
#             f"{prefix}/proj/vision_weight_norm": model.vision_proj[-1].weight.norm().item(),
#             f"{prefix}/proj/lidar_weight_norm": model.lidar_proj[-1].weight.norm().item(),
#         })

#     # --- Log to TensorBoard ---
#     for k, v in stats.items():
#         writer.add_scalar(k, v, step)
    
#     # --- Console Debug ---
#     print(f"\n[Alignment Stats @ Step {step}]")
#     print(f"• Loss: {loss.item():.4f}")
#     print(f"• RGB Feat STD: {stats[f'{prefix}/rgb_feat_std']:.4f} | LiDAR Feat STD: {stats[f'{prefix}/lidar_feat_std']:.4f}")
#     print(f"• Diag/Off-diag Sim: {stats[f'{prefix}/diag_sim_mean']:.4f}/{stats[f'{prefix}/off_diag_sim_mean']:.4f}")
#     print(f"• Cross-Modal Sim: {stats[f'{prefix}/cross_modal_sim']:.4f}")
    
#     if 'grad/proj' in ''.join(stats.keys()):
#         print("\n[Gradient Flow]")
#         for k in [k for k in stats if 'grad' in k]:
#             print(f"{k.split('/')[-1]:30} {stats[k]:.2e}")

#     if 'proj/vision' in ''.join(stats.keys()):
#         print("\n[Projection Layers]")
#         print(f"Vision Weight Norm: {stats[f'{prefix}/proj/vision_weight_norm']:.4f}")
#         print(f"LiDAR Weight Norm: {stats[f'{prefix}/proj/lidar_weight_norm']:.4f}")





























# import torch
# import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime

# run_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# writer = SummaryWriter(log_dir=f"lavis/output/BLIP2/qformer_CAM_FRONT/{run_name}")



# def log_alignment_stats(sim_matrix, loss, step=None, prefix="SIM"):
#     bs = sim_matrix.size(0)
#     device = sim_matrix.device
#     targets = torch.arange(bs).to(device)

#     # Diagonal and off-diagonal similarity
#     diag_sim = sim_matrix.diag()
#     off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=device)]

#     # Softmax confidence
#     prob_rgb2lidar = F.softmax(sim_matrix, dim=1)
#     prob_lidar2rgb = F.softmax(sim_matrix.T, dim=1)

#     max_prob_rgb2lidar = prob_rgb2lidar.max(dim=1).values
#     max_prob_lidar2rgb = prob_lidar2rgb.max(dim=1).values

#     # Accuracy

#     acc_rgb2lidar = (sim_matrix.argmax(dim=1) == targets).float().mean().item()
#     acc_lidar2rgb = (sim_matrix.argmax(dim=0) == targets).float().mean().item()

#     # Log everything
#     stats = {
#         f"{prefix}/loss": loss.item(),
#         f"{prefix}/diag_sim_mean": diag_sim.mean().item(),
#         f"{prefix}/off_diag_sim_mean": off_diag_sim.mean().item(),
#         f"{prefix}/rgb2lidar_max_prob_mean": max_prob_rgb2lidar.mean().item(),
#         f"{prefix}/lidar2rgb_max_prob_mean": max_prob_lidar2rgb.mean().item(),
#         f"{prefix}/acc_rgb2lidar": acc_rgb2lidar,
#         f"{prefix}/acc_lidar2rgb": acc_lidar2rgb,
#     }

#     # print("==== Alignment Debug Stats ====")
#     # for k, v in stats.items():
#     #     print(f"{k}: {v:.4f}")
#     #     writer.add_scalar(k, v, step)
#     # print("================================")


# def close_writer():
#     writer.flush()
#     writer.close()
