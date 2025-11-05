"""
XSensorDetector — a lightweight mutual-consistency detector for camera & LiDAR BEV features.

Inputs
------
- cam_bev: [B, 256, H, W]
- lid_bev: [B, 256, H, W]

Core
----
- Flatten BEV into tokens + learned 2D positional embeddings
- Two Decoder stacks (Self→Cross→FFN): Cam→LiD and LiD→Cam
- Two heads per modality: Self-reconstruction (memory) and Cross-prediction (consistency)

Training
--------
- Benign-only (no labels). Optimize L1 reconstruction + cross-prediction + small cosine alignment bonus

Inference
---------
- Per-frame anomaly score and per-modality heatmaps (+ simple attribution masks)

Drop-in
-------
- Copy this file into your repo and import the classes/functions you need
"""
from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 0) Utilities (flatten, pos enc)
# -----------------------------

def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] → [B, N, C] with N=H*W."""
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1).contiguous()


class LearnedPosEnc2D(nn.Module):
    """Simple learned 2D positional embedding of shape [1, H*W, D]."""

    def __init__(self, H: int, W: int, D: int, std: float = 0.02):
        super().__init__()
        self.H, self.W, self.D = H, W, D
        pe = torch.zeros(1, H * W, D)
        nn.init.trunc_normal_(pe, std=std)
        self.pe = nn.Parameter(pe)  # register as parameter

    def forward(self, B: int) -> torch.Tensor:
        # Broadcast to batch
        return self.pe.expand(B, -1, -1)


# ---------------------------------------------------
# 1) Transformer Decoder Block (Self→Cross→FFN)
# ---------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, pdrop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=heads, dropout=pdrop, batch_first=True
        )
        self.drop1 = nn.Dropout(pdrop)

        self.norm2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=heads, dropout=pdrop, batch_first=True
        )
        self.drop2 = nn.Dropout(pdrop)

        self.norm3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(pdrop), nn.Linear(4 * d, d)
        )
        self.drop3 = nn.Dropout(pdrop)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q_tokens: [B, Nq, D]  — the modality being updated
            kv_tokens: [B, Nk, D] — the other modality providing context
            key_padding_mask: optional mask for attention (False/0 = keep)
        Returns:
            Updated q_tokens: [B, Nq, D]
        """
        # Self-Attn
        x = self.norm1(q_tokens)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        q = q_tokens + self.drop1(x)

        # Cross-Attn
        x = self.norm2(q)
        x, _ = self.cross_attn(x, kv_tokens, kv_tokens, key_padding_mask=key_padding_mask)
        q = q + self.drop2(x)

        # FFN
        x = self.norm3(q)
        q = q + self.drop3(self.ff(x))
        return q


# --------------------------------------------------------------
# 2) Heads: Memory (self-recon) + Consistency (cross-pred)
# --------------------------------------------------------------
class Heads(nn.Module):
    def __init__(self, d: int = 256, hidden: int | None = None):
        super().__init__()
        h = hidden or d
        self.self_head = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d))
        self.cross_head = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.self_head(tokens)
        pred = self.cross_head(tokens)
        return rec, pred


# --------------------------------------------------------------
# 3) Full module (2 decoders: Cam→LiD, LiD→Cam)
# --------------------------------------------------------------
class XSensorDetector(nn.Module):
    def __init__(
        self,
        H: int = 120,
        W: int = 120,
        d: int = 256,
        heads: int = 8,
        blocks: int = 2,
        pdrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.H, self.W, self.d = H, W, d
        self.pos_enc = LearnedPosEnc2D(H, W, d)

        # stacks of decoder blocks
        self.cam_blocks = nn.ModuleList([DecoderBlock(d, heads, pdrop) for _ in range(blocks)])
        self.lid_blocks = nn.ModuleList([DecoderBlock(d, heads, pdrop) for _ in range(blocks)])

        # heads
        self.cam_heads = Heads(d)
        self.lid_heads = Heads(d)

    def forward(self, cam_bev: torch.Tensor, lid_bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        cam_bev, lid_bev: [B, 256, H, W]
        Returns a dict of token-level outputs (flattened spatial tokens).
        """
        assert (
            cam_bev.shape[1] == lid_bev.shape[1] == self.d
        ), f"Channel mismatch: got {cam_bev.shape} and {lid_bev.shape}, expected C={self.d}"
        B = cam_bev.shape[0]

        cam = flatten_tokens(cam_bev)  # [B, N, D]
        lid = flatten_tokens(lid_bev)  # [B, N, D]
        pe = self.pos_enc(B)
        cam = cam + pe
        lid = lid + pe

        # Block 1: camera leads (cam queries LiDAR) & symmetrically update LiDAR
        cam = self.cam_blocks[0](cam, lid)
        lid = self.lid_blocks[0](lid, cam)

        # Optional refinement block
        if len(self.cam_blocks) > 1:
            cam = self.cam_blocks[1](cam, lid)
            lid = self.lid_blocks[1](lid, cam)

        # Heads
        cam_rec, cam_pred_lid = self.cam_heads(cam)
        lid_rec, lid_pred_cam = self.lid_heads(lid)

        return {
            "cam_out": cam,
            "lid_out": lid,
            "cam_rec": cam_rec,
            "lid_rec": lid_rec,
            "cam_pred_lid": cam_pred_lid,
            "lid_pred_cam": lid_pred_cam,
        }


# ---------------------------------
# 4) Losses (benign-only training)
# ---------------------------------
@torch.no_grad()
def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x.norm(dim=dim, keepdim=True).clamp_min(eps)


def xdetect_loss(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    cam_out, lid_out = outputs["cam_out"], outputs["lid_out"]
    cam_rec, lid_rec = outputs["cam_rec"], outputs["lid_rec"]
    cam_pred_lid, lid_pred_cam = outputs["cam_pred_lid"], outputs["lid_pred_cam"]

    # L1 reconstruction & cross-prediction
    L_self_cam = (cam_rec - cam_out).abs().mean()
    L_self_lid = (lid_rec - lid_out).abs().mean()
    L_cross_c2l = (cam_pred_lid - lid_out).abs().mean()
    L_cross_l2c = (lid_pred_cam - cam_out).abs().mean()

    # Cosine alignment bonus
    cam_n = cam_out / _safe_norm(cam_out)
    lid_n = lid_out / _safe_norm(lid_out)
    L_cos = (1.0 - (cam_n * lid_n).sum(-1)).mean()

    L_total = L_self_cam + L_self_lid + L_cross_c2l + L_cross_l2c + 0.1 * L_cos
    logs = {
        "self_cam": float(L_self_cam.item()),
        "self_lid": float(L_self_lid.item()),
        "cross_c2l": float(L_cross_c2l.item()),
        "cross_l2c": float(L_cross_l2c.item()),
        "cos": float(L_cos.item()),
        "total": float(L_total.item()),
    }
    return L_total, logs


# -------------------------------------------------------
# 5) Inference: frame score + attribution maps (H×W)
# -------------------------------------------------------

def _to_map(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return x.view(-1, H, W)


def anomaly_scores(outputs: Dict[str, torch.Tensor], H: int = 120, W: int = 120) -> Dict[str, torch.Tensor]:
    cam_out, lid_out = outputs["cam_out"], outputs["lid_out"]
    cam_rec, lid_rec = outputs["cam_rec"], outputs["lid_rec"]
    cam_pred_lid, lid_pred_cam = outputs["cam_pred_lid"], outputs["lid_pred_cam"]

    # token-wise scores [B, N]
    S_cam = (cam_rec - cam_out).abs().mean(dim=-1)
    S_lid = (lid_rec - lid_out).abs().mean(dim=-1)
    S_x = 0.5 * (
        (cam_pred_lid - lid_out).abs().mean(dim=-1)
        + (lid_pred_cam - cam_out).abs().mean(dim=-1)
    )

    # heatmaps
    sC, sL, sX = _to_map(S_cam, H, W), _to_map(S_lid, H, W), _to_map(S_x, H, W)

    # simple attribution (boolean masks)
    def _gt_mean(m: torch.Tensor) -> torch.Tensor:
        return m > m.mean(dim=(1, 2), keepdim=True)

    cam_only = _gt_mean(sC) & (~_gt_mean(sL)) & _gt_mean(sX)
    lid_only = _gt_mean(sL) & (~_gt_mean(sC)) & _gt_mean(sX)
    both = _gt_mean(sC) & _gt_mean(sL) & _gt_mean(sX)

    frame_score = (0.4 * sC + 0.4 * sL + 0.2 * sX).mean(dim=(1, 2))  # [B]
    return {
        "score": frame_score,
        "cam_map": sC,
        "lid_map": sL,
        "x_map": sX,
        "cam_only": cam_only,
        "lid_only": lid_only,
        "both": both,
    }


# ---------------------------------------------
# 6) Optimizer snippet (benign-only training)
# ---------------------------------------------

def make_optimizer(model: nn.Module, lr: float = 2e-4, wd: float = 0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


# ---------------------------------------------
# 7) Minimal smoke test
# ---------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 256, 120, 120
    cam = torch.randn(B, C, H, W)
    lid = torch.randn(B, C, H, W)

    model = XSensorDetector(H=H, W=W, d=C, heads=8, blocks=2, pdrop=0.0)
    out = model(cam, lid)
    loss, logs = xdetect_loss(out)
    scores = anomaly_scores(out, H=H, W=W)

    print("Loss:", float(loss))
    print("Logs:", {k: round(v, 4) for k, v in logs.items()})
    print("Score shape:", tuple(scores["score"].shape))
