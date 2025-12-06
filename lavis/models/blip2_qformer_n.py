
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer
# from lavis.models.blip_outputs import BlipOutput
# from lavis.models.base_model import BaseModel, concat_all_gather
from lavis.models.Qformer import BertConfig, BertLMHeadModel
# from lavis.runners.log_utils import log_alignment_stats
# from lavis.models.mini_qformer import MiniQFormer
from PIL import Image
import torchvision.transforms as T


import logging
import os

import numpy as np
import torch
import torch.nn as nn
from lavis.common.dist_utils import download_cached_file#, is_dist_avail_and_initialized
from lavis.common.utils import get_abs_path, is_url
from omegaconf import OmegaConf






class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            load_pretrained = cfg.get("load_pretrained", True)
            if load_pretrained:
                # load pre-trained weights
                pretrain_path = cfg.get("pretrained", None)
                assert "Found load_finetuned is False, but pretrain_path is None."
                self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)


    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)        
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
        ]                
        return optim_params
    
    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot


class BaseEncoder(nn.Module):
    """
    Base class for primitive encoders, such as ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()

    def forward_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


@torch.no_grad()
def concat_all_gather(tensor): # ----------------------------------------------------------------------------

    return tensor



"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class BlipSimilarity(ModelOutput):
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None


@dataclass
class BlipIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models.

    image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2).
    itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,)

    """

    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class BlipOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[BlipSimilarity] = None

    intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


@dataclass
class BlipOutputWithLogits(BlipOutput):
    logits: torch.FloatTensor = None
    logits_m: torch.FloatTensor = None







"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------


# ========================  for mini  ============================================
import torch
import torch.nn as nn
import torch.nn.functional as F



def inject_noise(features, noise_std=0.1):
    noise = torch.randn_like(features) * noise_std
    return features + noise






# ========================  for mini and added learnable positional embeddings for the query tokens  ============================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0.1):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5

#         self.to_q = nn.Linear(dim, dim, bias=False)
#         self.to_k = nn.Linear(dim, dim, bias=False)
#         self.to_v = nn.Linear(dim, dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context):
#         B, N, D = x.shape
#         H = self.heads

#         q = self.to_q(x).view(B, N, H, D // H).transpose(1, 2)
#         k = self.to_k(context).view(B, -1, H, D // H).transpose(1, 2)
#         v = self.to_v(context).view(B, -1, H, D // H).transpose(1, 2)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         out = attn @ v
#         out = out.transpose(1, 2).reshape(B, N, D)

#         return self.to_out(out)


# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1, cross_attn=True):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

#         self.cross_attn = CrossAttention(dim, heads, dropout) if cross_attn else None
#         self.ln2 = nn.LayerNorm(dim) if cross_attn else None

#         self.ln3 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x, context=None):
#         # Self-attention on queries
#         x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]

#         # Cross-attention (only when context is provided)
#         if self.cross_attn is not None and context is not None:
#             x = x + self.cross_attn(self.ln2(x), context)

#         # Feed-forward
#         x = x + self.mlp(self.ln3(x))
#         return x


# class MiniQFormer(nn.Module):
#     def __init__(self, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, cross_attention_freq=1):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             TransformerBlock(
#                 dim=hidden_size,
#                 heads=num_attention_heads,
#                 cross_attn=(i % cross_attention_freq == 0)
#             )
#             for i in range(num_hidden_layers)
#         ])

#     def forward(self, query_embeds, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
#         x = query_embeds  # [B, N, D]
#         for block in self.blocks:
#             x = block(x, context=encoder_hidden_states)

#         class Output:
#             def __init__(self, last_hidden_state):
#                 self.last_hidden_state = last_hidden_state

#         return Output(last_hidden_state=x)


# ========================  Blip2Base  ============================================
class Blip2Base(BaseModel):
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        hidden_size = vision_width
        Qformer = MiniQFormer(
            hidden_size=256, 
            num_hidden_layers=4, 
            num_attention_heads=4, 
            cross_attention_freq=1
            )

        query_tokens = nn.Parameter(torch.randn(1, num_query_token, hidden_size))
        return Qformer, query_tokens



# ========================  Blip2Qformer  ============================================
from torch.utils.checkpoint import checkpoint 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import os
import torch
import matplotlib.pyplot as plt

def save_bev_visualizations(cam_target, cam_rec, lid_target, lid_rec, save_dir, prefix="sample"):
    """
    cam_target: [B, C, H, W]
    cam_rec:    [B, C, H, W]
    lid_target: [B, C, H, W]
    lid_rec:    [B, C, H, W]
    """

    os.makedirs(save_dir, exist_ok=True)

    # take first batch element
    cam_t = cam_target[0].detach().cpu()
    cam_r = cam_rec[0].detach().cpu()
    lid_t = lid_target[0].detach().cpu()
    lid_r = lid_rec[0].detach().cpu()

    # visualize ONLY the first channel (C0) — easiest for debugging
    cam_t0 = cam_t[0]
    cam_r0 = cam_r[0]
    lid_t0 = lid_t[0]
    lid_r0 = lid_r[0]

    err_cam = (cam_r0 - cam_t0).abs()
    err_lid = (lid_r0 - lid_t0).abs()

    # ---- save PNGs ----
    def save_map(tensor, filename):
        plt.figure(figsize=(6, 6))
        plt.imshow(tensor, cmap="viridis")
        plt.colorbar()
        plt.title(filename)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename + ".png"))
        plt.close()

    # camera
    save_map(cam_t0,  f"{prefix}_cam_target")
    save_map(cam_r0,  f"{prefix}_cam_rec")
    save_map(err_cam, f"{prefix}_cam_error")

    # lidar
    save_map(lid_t0,  f"{prefix}_lid_target")
    save_map(lid_r0,  f"{prefix}_lid_rec")
    save_map(err_lid, f"{prefix}_lid_error")



class FusionBlock(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()

        # Pre-LN Transformer design (best for stability)
        self.norm1_rgb = nn.LayerNorm(hidden_dim)
        self.norm1_lidar = nn.LayerNorm(hidden_dim)

        self.attn_rgb_self = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.attn_lidar_self = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

        # FFN (important!)
        self.norm2_rgb = nn.LayerNorm(hidden_dim)
        self.norm2_lidar = nn.LayerNorm(hidden_dim)

        self.ffn_rgb = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.ffn_lidar = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, rgb_tokens, lidar_tokens):

        # ----- Self Attention for CAMERA -----
        rgb_norm = self.norm1_rgb(rgb_tokens)
        rgb_attn, _ = self.attn_rgb_self(rgb_norm, rgb_norm, rgb_norm, need_weights=False)
        rgb_tokens = rgb_tokens + rgb_attn

        # ----- Self Attention for LIDAR -----
        lidar_norm = self.norm1_lidar(lidar_tokens)
        lidar_attn, _ = self.attn_lidar_self(lidar_norm, lidar_norm, lidar_norm, need_weights=False)
        lidar_tokens = lidar_tokens + lidar_attn

        # ----- FFN -----
        rgb_tokens = rgb_tokens + self.ffn_rgb(self.norm2_rgb(rgb_tokens))
        lidar_tokens = lidar_tokens + self.ffn_lidar(self.norm2_lidar(lidar_tokens))

        return rgb_tokens, lidar_tokens


class BevDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim//2, out_dim, 1),   # project back to original BEV channels
        )

    def forward(self, x):   # x: [B, D, H, W]
        return self.decoder(x)



class Blip2Qformer(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
        "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
    }

    def __init__(
        self,
        vit_model=None,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=True,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=1,
        embed_dim=256,
        max_txt_len=32,
        hidden_dim=256,
        # num_layers=4,
        num_layers=4,
        heads=4,
        mask_ratio=0.2,
        block_size=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_grad_checkpoint = use_grad_checkpoint

        self.rgb_layernorm = nn.LayerNorm(hidden_dim)
        self.lidar_layernorm = nn.LayerNorm(hidden_dim)
        self.vision_proj = nn.Linear(hidden_dim, embed_dim)
        self.lidar_proj = nn.Linear(hidden_dim, embed_dim)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.dropout = nn.Dropout(p=0.1)

        self._cached_pos = None
        self.num_layers = num_layers
        self.heads = heads

        # # Stack 4 transformer-style fusion blocks
        self.fusion_layers = nn.ModuleList([
            FusionBlock(hidden_dim, heads) for _ in range(num_layers)
        ])

        # H, W = 200, 200  # replace with your actual variable names
        # self.fusion_layers = nn.ModuleList([
        #     FusionBlock(hidden_dim, heads, H, W) for _ in range(num_layers)
        # ])
        self.mask_ratio = mask_ratio
        self.block_size = block_size        
        # BEV decoders
        self.cam_decoder = BevDecoder(hidden_dim, hidden_dim)
        self.lidar_decoder = BevDecoder(hidden_dim, hidden_dim)




    def _build_2d_sincos_pos_embed(self, h, w, dim, device):
        def get_1d_sin_cos_pos_embed(embed_dim, pos):
            assert embed_dim % 2 == 0
            omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
            omega = 1. / (10000 ** (omega / (embed_dim // 2)))
            pos = pos.reshape(-1, 1)
            out = pos * omega.reshape(1, -1)
            sin = torch.sin(out)
            cos = torch.cos(out)
            return torch.cat([sin, cos], dim=1)
        grid_y = torch.arange(h, dtype=torch.float32, device=device)
        grid_x = torch.arange(w, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)
        grid = grid.reshape(2, -1).T
        emb_y = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 0])
        emb_x = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 1])
        pos = torch.cat([emb_y, emb_x], dim=1)
        return pos.unsqueeze(0)  # [1, h*w, dim]



    # def forward(self, samples, is_train=True):
    #     # print("forward============")
    #     # -----------------------------------------------------
    #     # 0. Load BEV features
    #     # -----------------------------------------------------
    #     cam_bev = samples["cam_bev"]      # [B, C, H, W]
    #     lid_bev = samples["lidar_bev"]    # [B, C, H, W]
    #     B, C, H, W = cam_bev.shape

    #     # -----------------------------------------------------
    #     # 1. Normalize BEV targets (per-sample, per-modality)
    #     # -----------------------------------------------------
    #     eps = 1e-6

    #     cam_mean = cam_bev.mean(dim=[1,2,3], keepdim=True)
    #     cam_std  = cam_bev.std(dim=[1,2,3], keepdim=True) + eps
    #     cam_target = (cam_bev - cam_mean) / cam_std

    #     lid_mean = lid_bev.mean(dim=[1,2,3], keepdim=True)
    #     lid_std  = lid_bev.std(dim=[1,2,3], keepdim=True) + eps
    #     lid_target = (lid_bev - lid_mean) / lid_std

    #     # -----------------------------------------------------
    #     # 2. Flatten to tokens + LayerNorm + pos enc
    #     # -----------------------------------------------------
    #     cam_tokens = cam_target.flatten(2).transpose(1,2)   # [B, N, C]
    #     lid_tokens = lid_target.flatten(2).transpose(1,2)

    #     # cam_tokens = self.rgb_layernorm(cam_tokens)
    #     # lid_tokens = self.lidar_layernorm(lid_tokens)

    #     # Positional embedding
    #     if self._cached_pos is None or self._cached_pos.shape[1] != (H * W):
    #         self._cached_pos = self._build_2d_sincos_pos_embed(H, W, C, cam_tokens.device)

    #     pos = self._cached_pos.to(cam_tokens.device) # [1, N, C]

    #     cam_tokens = cam_tokens + pos
    #     lid_tokens = lid_tokens + pos

    #     # -----------------------------------------------------
    #     # 3. Transformer fusion stack (self-attn only in Phase 2)
    #     # -----------------------------------------------------
    #     for layer in self.fusion_layers:
    #         cam_tokens, lid_tokens = layer(cam_tokens, lid_tokens)

    #     # -----------------------------------------------------
    #     # 4. Reshape tokens back to BEV grid
    #     # -----------------------------------------------------
    #     cam_latent = cam_tokens.transpose(1,2).view(B, C, H, W)
    #     lid_latent = lid_tokens.transpose(1,2).view(B, C, H, W)

    #     # -----------------------------------------------------
    #     # 5. CNN Decoder → reconstruct normalized BEV features
    #     # -----------------------------------------------------
    #     cam_rec = self.cam_decoder(cam_latent)     # [B, C, H, W]
    #     lid_rec = self.lidar_decoder(lid_latent)   # [B, C, H, W]

    #     # -----------------------------------------------------
    #     # 6. SmoothL1 reconstruction loss (balanced & stable)
    #     # -----------------------------------------------------
    #     loss_cam = F.smooth_l1_loss(cam_rec, cam_target)
    #     loss_lid = F.smooth_l1_loss(lid_rec, lid_target)

    #     total_loss = loss_cam + loss_lid

    #     # -----------------------------------------------------
    #     # 7. Return clean dict
    #     # -----------------------------------------------------
    #     return {
    #         "loss": total_loss,
    #         "loss_cam": loss_cam.detach(),
    #         "loss_lid": loss_lid.detach(),
    #     }
    
    

    # @torch.no_grad()
    # def forward_features(self, samples, return_maps=False):
    #     """
    #     Evaluation forward pass:
    #     - Mirrors the updated forward() exactly
    #     - Uses normalized BEV targets (same as training)
    #     - Runs fusion + CNN decoders
    #     - Computes per-token discrepancy maps and scalar anomaly scores
    #     - No gradients, no training losses returned
    #     """
    #     return_maps = True
    #     # print("forward_features=========================================")
    #     # -----------------------------------------------------
    #     # 0. Load BEV features
    #     # -----------------------------------------------------
    #     cam_bev = samples["cam_bev"]      # [B, C, H, W]
    #     lid_bev = samples["lidar_bev"]    # [B, C, H, W]
    #     B, C, H, W = cam_bev.shape

    #     # -----------------------------------------------------
    #     # 1. Normalize BEV targets (exactly as in forward())
    #     # -----------------------------------------------------
    #     eps = 1e-6
    #     cam_mean = cam_bev.mean(dim=[1,2,3], keepdim=True)
    #     cam_std  = cam_bev.std(dim=[1,2,3], keepdim=True) + eps
    #     cam_target = (cam_bev - cam_mean) / cam_std

    #     lid_mean = lid_bev.mean(dim=[1,2,3], keepdim=True)
    #     lid_std  = lid_bev.std(dim=[1,2,3], keepdim=True) + eps
    #     lid_target = (lid_bev - lid_mean) / lid_std

    #     # -----------------------------------------------------
    #     # 2. Flatten → tokens + LayerNorm + pos enc
    #     # -----------------------------------------------------
    #     cam_tokens = cam_target.flatten(2).transpose(1, 2)   # [B, N, C]
    #     lid_tokens = lid_target.flatten(2).transpose(1, 2)

    #     # cam_tokens = self.rgb_layernorm(cam_tokens)
    #     # lid_tokens = self.lidar_layernorm(lid_tokens)

    #     if self._cached_pos is None or self._cached_pos.shape[1] != (H * W):
    #         self._cached_pos = self._build_2d_sincos_pos_embed(H, W, C, cam_tokens.device)

    #     pos = self._cached_pos.to(cam_tokens.device)  # [1, N, C]
    #     cam_tokens = cam_tokens + pos
    #     lid_tokens = lid_tokens + pos

    #     # -----------------------------------------------------
    #     # 3. Transformer fusion stack
    #     # -----------------------------------------------------
    #     for layer in self.fusion_layers:
    #         cam_tokens, lid_tokens = layer(cam_tokens, lid_tokens)

    #     # -----------------------------------------------------
    #     # 4. Reshape tokens → latent BEV grids
    #     # -----------------------------------------------------
    #     cam_latent = cam_tokens.transpose(1, 2).view(B, C, H, W)
    #     lid_latent = lid_tokens.transpose(1, 2).view(B, C, H, W)

    #     # -----------------------------------------------------
    #     # 5. Decoders → normalized BEV reconstructions
    #     # -----------------------------------------------------
    #     cam_rec = self.cam_decoder(cam_latent)   # [B, C, H, W]
    #     lid_rec = self.lidar_decoder(lid_latent)

    #     # -----------------------------------------------------
    #     # 6. Per-pixel L1 discrepancy (anomaly signal)
    #     # -----------------------------------------------------
    #     cam_diff = (cam_rec - cam_target).abs().mean(1)   # [B, H, W]
    #     lid_diff = (lid_rec - lid_target).abs().mean(1)   # [B, H, W]

    #     # Scalar anomaly scores
    #     L_cam = cam_diff.mean()   # camera inconsistency score
    #     L_lid = lid_diff.mean()   # lidar inconsistency score

    #     # -----------------------------------------------------
    #     # 7. Return results
    #     # -----------------------------------------------------
    #     out = {
    #         "loss_cam": L_cam,
    #         "loss_lid": L_lid,
    #         "cam_tokens": cam_tokens,   # [B, N, C]
    #         "lid_tokens": lid_tokens,
    #     }

    #     if return_maps:
    #         # out["cam_discrepancy_map"] = cam_diff
    #         # out["lid_discrepancy_map"] = lid_diff
    #         out["cam_rec"] = cam_rec.detach().cpu()
    #         out["lid_rec"] = lid_rec.detach().cpu()
    #     return out


    # ============================================================
    #  SHARED ENCODE → FUSE → DECODE PIPELINE
    # ============================================================
    def _encode_decode(self, cam_bev, lid_bev):
        """
        Shared pipeline for both forward() and forward_features().
        Performs:
            - Normalization
            - Flattening to tokens
            - Positional embedding
            - Fusion transformer layers
            - Reshape to BEV grid
            - Decode with CNN decoders

        Returns:
            cam_rec, lid_rec,
            cam_latent, lid_latent,
            cam_target, lid_target
        """
        B, C, H, W = cam_bev.shape
        eps = 1e-6

        # ---------------------------------------------------------
        # 1. Normalize per-sample BEV features
        # ---------------------------------------------------------
        cam_mean = cam_bev.mean(dim=[1,2,3], keepdim=True)
        cam_std  = cam_bev.std(dim=[1,2,3], keepdim=True) + eps
        cam_target = (cam_bev - cam_mean) / cam_std

        lid_mean = lid_bev.mean(dim=[1,2,3], keepdim=True)
        lid_std  = lid_bev.std(dim=[1,2,3], keepdim=True) + eps
        lid_target = (lid_bev - lid_mean) / lid_std

        # ---------------------------------------------------------
        # 2. Tokens (flatten H×W → sequence)
        # ---------------------------------------------------------
        cam_tokens = cam_target.flatten(2).transpose(1,2)   # [B, N, C]
        lid_tokens = lid_target.flatten(2).transpose(1,2)

        # Positional embedding
        if self._cached_pos is None or self._cached_pos.shape[1] != (H * W):
            self._cached_pos = self._build_2d_sincos_pos_embed(H, W, C, cam_tokens.device)

        pos = self._cached_pos.to(cam_tokens.device)  # [1, N, C]
        cam_tokens = cam_tokens + pos
        lid_tokens = lid_tokens + pos

        # ---------------------------------------------------------
        # 3. Transformer Fusion (your Phase-2 design)
        # ---------------------------------------------------------
        for layer in self.fusion_layers:
            cam_tokens, lid_tokens = layer(cam_tokens, lid_tokens)

        # ---------------------------------------------------------
        # 4. Reshape tokens → latent BEV grids
        # ---------------------------------------------------------
        cam_latent = cam_tokens.transpose(1,2).view(B, C, H, W)
        lid_latent = lid_tokens.transpose(1,2).view(B, C, H, W)

        # ---------------------------------------------------------
        # 5. Decoders reconstruct normalized BEV features
        # ---------------------------------------------------------
        cam_rec = self.cam_decoder(cam_latent)
        lid_rec = self.lidar_decoder(lid_latent)

        return (
            cam_rec, lid_rec,
            cam_latent, lid_latent,
            cam_target, lid_target
        )


    # ============================================================
    #  TRAINING FORWARD
    # ============================================================
    def forward(self, samples):
        cam_bev = samples["cam_bev"]
        lid_bev = samples["lidar_bev"]

        (
            cam_rec, lid_rec,
            cam_latent, lid_latent,
            cam_target, lid_target
        ) = self._encode_decode(cam_bev, lid_bev)

        # Reconstruction losses
        loss_cam = F.smooth_l1_loss(cam_rec, cam_target)
        loss_lid = F.smooth_l1_loss(lid_rec, lid_target)
        total_loss = loss_cam + loss_lid

        return {
            "loss": total_loss,
            "loss_cam": loss_cam.detach(),
            "loss_lid": loss_lid.detach(),
        }


    # ============================================================
    #  EVALUATION FORWARD (NO GRADIENTS)
    # ============================================================
    @torch.no_grad()
    def forward_features(self, samples, return_maps=False):
        cam_bev = samples["cam_bev"]
        lid_bev = samples["lidar_bev"]

        (
            cam_rec, lid_rec,
            cam_latent, lid_latent,
            cam_target, lid_target
        ) = self._encode_decode(cam_bev, lid_bev)

        # ---------------------------------------------------------
        # Per-pixel discrepancy maps (anomaly signal)
        # ---------------------------------------------------------
        cam_diff = (cam_rec - cam_target).abs().mean(1)   # [B, H, W]
        lid_diff = (lid_rec - lid_target).abs().mean(1)

        # Global scalar inconsistency scores
        L_cam = cam_diff.mean()
        L_lid = lid_diff.mean()

        out = {
            "loss_cam": L_cam,
            "loss_lid": L_lid,
            "cam_latent": cam_latent,
            "lid_latent": lid_latent,
        }

        if return_maps:
            out["cam_rec"] = cam_rec.cpu()
            out["lid_rec"] = lid_rec.cpu()
            out["cam_diff"] = cam_diff.cpu()
            out["lid_diff"] = lid_diff.cpu()

        return out




    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)
        return model



















































