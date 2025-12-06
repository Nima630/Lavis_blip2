
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




# class Blip2Base(BaseModel):
#     @classmethod
#     def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
#         # print("[TRACE] init_Qformer in blip2.py")
#         encoder_config = BertConfig.from_pretrained("bert-base-uncased")
#         encoder_config.num_hidden_layers = 4 #6 
#         encoder_config.encoder_width = vision_width
#         encoder_config.add_cross_attention = True
#         encoder_config.cross_attention_freq = cross_attention_freq
#         encoder_config.query_length = num_query_token
#         Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
#         query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
#         query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
#         return Qformer, query_tokens







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
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        B, N, D = x.shape
        H = self.heads

        q = self.to_q(x).view(B, N, H, D // H).transpose(1, 2)
        k = self.to_k(context).view(B, -1, H, D // H).transpose(1, 2)
        v = self.to_v(context).view(B, -1, H, D // H).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1, cross_attn=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.cross_attn = CrossAttention(dim, heads, dropout) if cross_attn else None
        self.ln2 = nn.LayerNorm(dim) if cross_attn else None

        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        # Self-attention on queries
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]

        # Cross-attention (only when context is provided)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.ln2(x), context)

        # Feed-forward
        x = x + self.mlp(self.ln3(x))
        return x


class MiniQFormer(nn.Module):
    def __init__(self, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, cross_attention_freq=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                heads=num_attention_heads,
                cross_attn=(i % cross_attention_freq == 0)
            )
            for i in range(num_hidden_layers)
        ])

    def forward(self, query_embeds, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
        x = query_embeds  # [B, N, D]
        for block in self.blocks:
            x = block(x, context=encoder_hidden_states)

        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Output(last_hidden_state=x)


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
# class Blip2Qformer(Blip2Base):
#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#         "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
#     }

    # def __init__(
    #     self,
    #     vit_model=None,
    #     img_size=224,
    #     drop_path_rate=0,
    #     use_grad_checkpoint=True, #False,
    #     vit_precision="fp16",
    #     freeze_vit=True,
    #     num_query_token=32,
    #     cross_attention_freq=1,
    #     embed_dim=256,
    #     max_txt_len=32,):
    #     super().__init__()

    #     # hidden_dim = 768
    #     hidden_dim = 256
    #     self.hidden_dim = hidden_dim
    #     # self.rgb_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim) # resnet
    #     # # self.rgb_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim) # fpn
    #     # self.lidar_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim)
        
    #     self.rgb_layernorm = nn.LayerNorm(hidden_dim)
    #     self.lidar_layernorm = nn.LayerNorm(hidden_dim)

    #     # self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
    #     # qformer_lidar, query_tokens_lidar = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
    #     # self.Qformer_lidar = qformer_lidar
    #     # self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
    #     # self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)
    #     self.use_grad_checkpoint = use_grad_checkpoint
    #     self.vision_proj = nn.Linear(hidden_dim, embed_dim)
    #     self.lidar_proj = nn.Linear(hidden_dim, embed_dim)
    #     self.log_temp = nn.Parameter(torch.log(torch.tensor(0.1)))  # Learnable temp
        
    #     self.temp = nn.Parameter(0.07 * torch.ones([]))
    #     self.dropout = nn.Dropout(p=0.1)

    #     # cache to avoid re-building sincos every forward if size doesn't change
    #     self._cached_rgb_hw   = None
    #     self._cached_rgb_pos  = None
    #     self._cached_lidar_hw = None
    #     self._cached_lidar_pos= None
    #     heads = 4 #8
    #     self.heads = heads
    #     H = 125
    #     W = 200
    #     # self.relative_bias = RelativePositionBias(num_heads=heads, window_size=(H, W))
    #     self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
    #     self.ffn = nn.Sequential(
    #         nn.Linear(hidden_dim, 2 * hidden_dim),
    #         nn.GELU(),
    #         nn.Linear(2 * hidden_dim, hidden_dim)
    #     )
    #     self.norm1 = nn.LayerNorm(hidden_dim)
    #     self.norm2 = nn.LayerNorm(hidden_dim)
    #     self.norm3 = nn.LayerNorm(hidden_dim)



    # def init_feature_projection(self, in_dim, out_dim):
    #     return nn.Conv2d(in_dim, out_dim, kernel_size=1)

    # def _diversity_loss(self, features):
    #     """Penalizes collapsed feature representations.
    #     Args:
    #         features: [batch_size, feature_dim]
    #     Returns:
    #         loss: scalar tensor
    #     """
    #     # Normalize features first to get valid cosine similarity
    #     features = F.normalize(features, dim=-1)
    #     sim_matrix = torch.mm(features, features.t())  # [B,B]
    #     # print("---------------------- sim_matrix", sim_matrix)
    #     # Mask to exclude diagonal (self-similarities)
    #     mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    #     avg_sim = sim_matrix[mask].mean()
        
    #     # Target 0.0 average similarity between different samples
    #     return torch.abs(avg_sim)



#     def _build_2d_sincos_pos_embed(self, h, w, dim, device):
#         """
#         Returns [1, h*w, dim] 2D sin-cos positional embedding.
#         """
#         import math

#         def get_1d_sin_cos_pos_embed(embed_dim, pos):
#             # embed_dim must be even
#             assert embed_dim % 2 == 0
#             omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
#             omega = 1. / (10000 ** (omega / (embed_dim // 2)))
#             pos = pos.reshape(-1, 1)  # [M, 1]
#             out = pos * omega.reshape(1, -1)  # [M, embed_dim/2]
#             sin = torch.sin(out)
#             cos = torch.cos(out)
#             return torch.cat([sin, cos], dim=1)  # [M, embed_dim]

#         grid_y = torch.arange(h, dtype=torch.float32, device=device)
#         grid_x = torch.arange(w, dtype=torch.float32, device=device)
#         grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)  # [2, h, w]
#         grid = grid.reshape(2, -1).T  # [h*w, 2]

#         emb_y = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 0])
#         emb_x = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 1])
#         pos = torch.cat([emb_y, emb_x], dim=1)  # [h*w, dim]
#         return pos.unsqueeze(0)  # [1, h*w, dim]



#     # # # ======================== with positional encodings ============================================
   

#     # def forward(self, samples, is_train=True):
#     #     """
#     #     Forward pass for mutual-consistency BEV fusion model.
#     #     - Takes BEV features from camera and LiDAR.
#     #     - Applies transformer with relative positional bias.
#     #     - Computes L1 + Cosine + Local InfoNCE + MAE losses.
#     #     """

#     #     image = samples["cam_bev"]    # [B, 256, H, W] camera BEV features
#     #     lidar = samples["lidar_bev"]    # [B, 256, H, W] LiDAR BEV features
#     #     bs, c, H, W = image.shape

#     #     # === 1. Feature projection ===
#     #     # rgb_proj = self.rgb_input_proj(image)     # [B, D, H, W]
#     #     # lidar_proj = self.lidar_input_proj(lidar) # [B, D, H, W]


#     #     rgb_proj = image     # [B, D, H, W]
#     #     lidar_proj = lidar # [B, D, H, W]


#     #     # === 2. Flatten to tokens ===
#     #     rgb_tokens = rgb_proj.flatten(2).transpose(1, 2)       # [B, N, D]
#     #     lidar_tokens = lidar_proj.flatten(2).transpose(1, 2)   # [B, N, D]

#     #     rgb_tokens = self.rgb_layernorm(rgb_tokens)
#     #     lidar_tokens = self.lidar_layernorm(lidar_tokens)

#     #     # === 3. Add 2D sinusoidal position encodings ===
#     #     if not hasattr(self, "_cached_pos") or self._cached_pos.shape[1] != H * W:
#     #         self._cached_pos = self._build_2d_sincos_pos_embed(H, W, rgb_tokens.size(-1), rgb_tokens.device)
#     #     pos_embed = self._cached_pos

#     #     # rgb_tokens = rgb_tokens + pos_embed
#     #     pos_embed = self._cached_pos.to(rgb_tokens.device)
#     #     lidar_tokens = lidar_tokens + pos_embed

#     #     # === 4. Relative positional bias ===
#     #     # rel_bias = self.relative_bias()  # [heads, N, N]

#     #     # MultiheadAttention doesn’t accept bias directly → we apply it manually
#     #     # We'll do self-attn + cross-attn with bias
#     #     rgb_q = lidar_q = None  # placeholders
#     #     # # Self-attention (camera)
#     #     # q, k, v = self.norm1(rgb_tokens), self.norm1(rgb_tokens), self.norm1(rgb_tokens)
#     #     # attn_scores = torch.matmul(q @ self.attn.in_proj_weight[:self.hidden_dim, :].T,
#     #     #                         k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
#     #     # # Apply relative bias
#     #     # attn_scores = attn_scores.view(bs, self.heads, H*W, H*W) # + rel_bias.unsqueeze(0)
#     #     # attn_weights = attn_scores.softmax(dim=-1)
#     #     # rgb_self = (attn_weights @ v).reshape(bs, H*W, self.hidden_dim)

#     #     # # Self-attention (LiDAR)
#     #     # q, k, v = self.norm1(lidar_tokens), self.norm1(lidar_tokens), self.norm1(lidar_tokens)
#     #     # attn_scores = torch.matmul(q @ self.attn.in_proj_weight[:self.hidden_dim, :].T,
#     #     #                         k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
#     #     # attn_scores = attn_scores.view(bs, self.heads, H*W, H*W) # + rel_bias.unsqueeze(0)
#     #     # attn_weights = attn_scores.softmax(dim=-1)
#     #     # lidar_self = (attn_weights @ v).reshape(bs, H*W, self.hidden_dim)


#     #     # === 4. Self-Attention (camera & LiDAR) ===  # ---- change line 1
#     #     rgb_self, _ = self.attn(self.norm1(rgb_tokens),
#     #                             self.norm1(rgb_tokens),
#     #                             self.norm1(rgb_tokens),
#     #                             need_weights=False)  # ---- change line 2

#     #     lidar_self, _ = self.attn(self.norm1(lidar_tokens),
#     #                             self.norm1(lidar_tokens),
#     #                             self.norm1(lidar_tokens),
#     #                             need_weights=False)  # ---- change line 3


#     #     # === 5. Cross-attention (mutual fusion) ===
#     #     q_cam, k_lid, v_lid = self.norm2(rgb_self), self.norm2(lidar_self), self.norm2(lidar_self)
#     #     q_lid, k_cam, v_cam = self.norm2(lidar_self), self.norm2(rgb_self), self.norm2(rgb_self)

#     #     cam_fused = self.attn(q_cam, k_lid, v_lid, need_weights=False)[0]
#     #     lid_fused = self.attn(q_lid, k_cam, v_cam, need_weights=False)[0]

#     #     # === 6. Feed-forward ===
#     #     cam_out = cam_fused + self.ffn(self.norm3(cam_fused))
#     #     lid_out = lid_fused + self.ffn(self.norm3(lid_fused))

#     #     # === 7. Reconstruction heads ===
#     #     cam_rec = self.vision_proj(cam_out)
#     #     lid_rec = self.lidar_proj(lid_out)

#     #     # === 8. Loss computation ===
#     #     # Optional mask (for masked BEV training, later)
#     #     mask = torch.ones((bs, H * W), device=image.device)

#     #     L_recon_cam = F.l1_loss(cam_rec, rgb_tokens) + 0.1 * (1 - F.cosine_similarity(cam_rec, rgb_tokens, dim=-1)).mean()
#     #     L_recon_lid = F.l1_loss(lid_rec, lidar_tokens) + 0.1 * (1 - F.cosine_similarity(lid_rec, lidar_tokens, dim=-1)).mean()
#     #     # L_infonce = shuffled_infonce_loss(cam_rec, lid_rec)


#     #     # L_infonce = shuffled_infonce_with_neg_sampling(cam_rec, lid_rec)        
#     #     # L_mae = mae_loss(cam_rec, rgb_tokens, mask)

#     #     total_loss = L_recon_cam + L_recon_lid #+ 0.5 * L_infonce  #+ 0.1 * L_mae 

#     #     # === 9. Return ===
#     #     return {
#     #         "loss": total_loss,
#     #         "L_recon_cam": L_recon_cam.detach(),
#     #         "L_recon_lid": L_recon_lid.detach(),
#     #         # "L_infonce": L_infonce.detach(),
#     #         # "L_mae": L_mae.detach(),
#     #         # "cam_out": cam_out,
#     #         # "lid_out": lid_out,
#     #     }


#     # def forward(self, samples, is_train=True):
#     #     image = samples["cam_bev"]      # [B, 256, H, W]
#     #     lidar = samples["lidar_bev"]    # [B, 256, H, W]
#     #     bs, c, H, W = image.shape

#     #     # 1. (optional projection)
#     #     rgb_proj   = image      # [B, D, H, W]
#     #     lidar_proj = lidar      # [B, D, H, W]

#     #     # 2. Flatten to tokens
#     #     rgb_tokens   = rgb_proj.flatten(2).transpose(1, 2)     # [B, N, D]
#     #     lidar_tokens = lidar_proj.flatten(2).transpose(1, 2)   # [B, N, D]

#     #     rgb_tokens   = self.rgb_layernorm(rgb_tokens)
#     #     lidar_tokens = self.lidar_layernorm(lidar_tokens)

#     #     # 3. Positional encoding addition
#     #     if not hasattr(self, "_cached_pos") or self._cached_pos.shape[1] != H * W:
#     #         self._cached_pos = self._build_2d_sincos_pos_embed(
#     #             H, W, rgb_tokens.size(-1), rgb_tokens.device
#     #         )
#     #     pos_embed = self._cached_pos.to(rgb_tokens.device)

#     #     rgb_tokens = rgb_tokens + pos_embed   
#     #     lidar_tokens = lidar_tokens + pos_embed

#     #     # ---------- FUSION BLOCK STARTS HERE ----------
#     #     # checkpoint THIS part, because it's the expensive attention + FFN.

#     #     def fusion_block(rgb_in, lidar_in):
#     #         """
#     #         rgb_in:   [B, N, D]
#     #         lidar_in: [B, N, D]

#     #         returns:
#     #             cam_out [B, N, D]
#     #             lid_out [B, N, D]
#     #         """

#     #         # Self-attention (camera)
#     #         rgb_self, _ = self.attn(
#     #             self.norm1(rgb_in),
#     #             self.norm1(rgb_in),
#     #             self.norm1(rgb_in),
#     #             need_weights=False,
#     #         )

#     #         # Self-attention (LiDAR)
#     #         lidar_self, _ = self.attn(
#     #             self.norm1(lidar_in),
#     #             self.norm1(lidar_in),
#     #             self.norm1(lidar_in),
#     #             need_weights=False,
#     #         )

#     #         # Cross-attention (camera queries LiDAR)
#     #         q_cam  = self.norm2(rgb_self)
#     #         k_lid  = self.norm2(lidar_self)
#     #         v_lid  = self.norm2(lidar_self)
#     #         cam_fused, _ = self.attn(q_cam, k_lid, v_lid, need_weights=False)

#     #         # Cross-attention (LiDAR queries camera)
#     #         q_lid  = self.norm2(lidar_self)
#     #         k_cam  = self.norm2(rgb_self)
#     #         v_cam  = self.norm2(rgb_self)
#     #         lid_fused, _ = self.attn(q_lid, k_cam, v_cam, need_weights=False)

#     #         # FFN + residual for each branch
#     #         cam_out = cam_fused + self.ffn(self.norm3(cam_fused))
#     #         lid_out = lid_fused + self.ffn(self.norm3(lid_fused))

#     #         return cam_out, lid_out

#     #     if self.use_grad_checkpoint and is_train:
#     #         # save VRAM during training
#     #         cam_out, lid_out = checkpoint(fusion_block, 
#     #                                       rgb_tokens, 
#     #                                       lidar_tokens, 
#     #                                       use_reentrant=False,
#     #                                       )
#     #     else:
#     #         # normal path (eval or no checkpoint)
#     #         cam_out, lid_out = fusion_block(rgb_tokens, lidar_tokens)

#     #     # ---------- FUSION BLOCK ENDS HERE ----------

#     #     # 4. Reconstruction heads
#     #     cam_rec = self.vision_proj(cam_out)   # [B, N, embed_dim]
#     #     lid_rec = self.lidar_proj(lid_out)    # [B, N, embed_dim]

#     #     # 5. Losses
#     #     L_recon_cam = (
#     #         F.l1_loss(cam_rec, rgb_tokens)
#     #         + 0.1 * (1 - F.cosine_similarity(cam_rec, rgb_tokens, dim=-1)).mean()
#     #     )
#     #     L_recon_lid = (
#     #         F.l1_loss(lid_rec, lidar_tokens)
#     #         + 0.1 * (1 - F.cosine_similarity(lid_rec, lidar_tokens, dim=-1)).mean()
#     #     )

#     #     total_loss = L_recon_cam + L_recon_lid
#     #     total_loss_for_dp = total_loss.view(1)  
#     #     # 6. Return (with cam_out / lid_out dropped to save VRAM lifetime)
#     #     return {
#     #         "loss": total_loss_for_dp,
#     #         "L_recon_cam": L_recon_cam.detach(),
#     #         "L_recon_lid": L_recon_lid.detach(),
#     #         # "cam_out": cam_out,         # keep commented if you don't need them
#     #         # "lid_out": lid_out,
#     #     }

#     def forward(self, samples, is_train=True):
#         image = samples["cam_bev"]      # [B, 256, H, W]
#         lidar = samples["lidar_bev"]    # [B, 256, H, W]
#         bs, c, H, W = image.shape

#         # 1. (optional projection)
#         rgb_proj   = image      # [B, D, H, W]
#         lidar_proj = lidar      # [B, D, H, W]

#         # 2. Flatten to tokens
#         rgb_tokens   = rgb_proj.flatten(2).transpose(1, 2)     # [B, N, D]
#         lidar_tokens = lidar_proj.flatten(2).transpose(1, 2)   # [B, N, D]

#         rgb_tokens   = self.rgb_layernorm(rgb_tokens)
#         lidar_tokens = self.lidar_layernorm(lidar_tokens)

#         # 3. Positional encoding addition (applied to both)
#         if not hasattr(self, "_cached_pos") or self._cached_pos.shape[1] != H * W:
#             self._cached_pos = self._build_2d_sincos_pos_embed(
#                 H, W, rgb_tokens.size(-1), rgb_tokens.device
#             )
#         pos_embed = self._cached_pos.to(rgb_tokens.device)

#         rgb_tokens = rgb_tokens + pos_embed   
#         lidar_tokens = lidar_tokens + pos_embed

#         # ---------- FUSION BLOCK STARTS HERE ----------
#         def fusion_block(rgb_in, lidar_in):
#             # Self-attention (camera)
#             rgb_self, _ = self.attn(
#                 self.norm1(rgb_in),
#                 self.norm1(rgb_in),
#                 self.norm1(rgb_in),
#                 need_weights=False,
#             )
#             # Self-attention (LiDAR)
#             lidar_self, _ = self.attn(
#                 self.norm1(lidar_in),
#                 self.norm1(lidar_in),
#                 self.norm1(lidar_in),
#                 need_weights=False,
#             )
#             # Cross-attention (camera queries LiDAR)
#             q_cam  = self.norm2(rgb_self)
#             k_lid  = self.norm2(lidar_self)
#             v_lid  = self.norm2(lidar_self)
#             cam_fused, _ = self.attn(q_cam, k_lid, v_lid, need_weights=False)
#             # Cross-attention (LiDAR queries camera)
#             q_lid  = self.norm2(lidar_self)
#             k_cam  = self.norm2(rgb_self)
#             v_cam  = self.norm2(rgb_self)
#             lid_fused, _ = self.attn(q_lid, k_cam, v_cam, need_weights=False)
#             # FFN + residual for each branch
#             cam_out = cam_fused + self.ffn(self.norm3(cam_fused))
#             lid_out = lid_fused + self.ffn(self.norm3(lid_fused))
#             return cam_out, lid_out

#         if self.use_grad_checkpoint and is_train:
#             cam_out, lid_out = checkpoint(
#                 fusion_block, 
#                 rgb_tokens, 
#                 lidar_tokens, 
#                 use_reentrant=False,
#             )
#         else:
#             cam_out, lid_out = fusion_block(rgb_tokens, lidar_tokens)

#         # ---------- FUSION BLOCK ENDS HERE ----------

#         # 4. Cross-reconstruction heads
#         # Predict LiDAR from Camera and Camera from LiDAR
#         lidar_rec = self.lidar_proj(cam_out)   # camera predicts LiDAR features [B, N, D]
#         cam_rec   = self.vision_proj(lid_out)  # LiDAR predicts camera features [B, N, D]

#         # 5. Discrepancy maps (element-wise L1 loss for each token)
#         # Use mean over channels for each BEV cell
#         lid_discrepancy_map = (lidar_rec - lidar_tokens).abs().mean(-1)   # [B, N]
#         cam_discrepancy_map = (cam_rec - rgb_tokens).abs().mean(-1)       # [B, N]

#         # 6. Scalar losses (mean over all tokens and batch, for training)
#         L_cross_lid = lid_discrepancy_map.mean()
#         L_cross_cam = cam_discrepancy_map.mean()
#         total_loss = L_cross_lid + L_cross_cam

#         return {
#             "loss": total_loss.view(1),
#             "L_cross_lid": L_cross_lid.detach(),
#             "L_cross_cam": L_cross_cam.detach(),
#             # "lid_discrepancy_map": lid_discrepancy_map.detach(),  # [B, N], for heatmap
#             # "cam_discrepancy_map": cam_discrepancy_map.detach(),  # [B, N], for heatmap
#             # "cam_out": cam_out,         # keep commented if you don't need them
#             # "lid_out": lid_out,
#         }



#     @classmethod
#     def from_config(cls, cfg):
#         vit_model = cfg.get("vit_model", "eva_clip_g")
#         img_size = cfg.get("image_size")
#         num_query_token = cfg.get("num_query_token")
#         cross_attention_freq = cfg.get("cross_attention_freq", 2)
#         drop_path_rate = cfg.get("drop_path_rate", 0)
#         use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
#         vit_precision = cfg.get("vit_precision", "fp16")
#         freeze_vit = cfg.get("freeze_vit", True)
#         max_txt_len = cfg.get("max_txt_len", 32)

#         model = cls(
#             vit_model=vit_model,
#             img_size=img_size,
#             drop_path_rate=drop_path_rate,
#             use_grad_checkpoint=use_grad_checkpoint,
#             vit_precision=vit_precision,
#             freeze_vit=freeze_vit,
#             num_query_token=num_query_token,
#             cross_attention_freq=cross_attention_freq,
#             max_txt_len=max_txt_len,
#         )
#         model.load_checkpoint_from_config(cfg)
#         return model





# ###################### no duplicate parameters ######################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint

# old, no partition and no duplicated parameters or residuals 
# class FusionBlock(nn.Module):
#     def __init__(self, hidden_dim, heads):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(hidden_dim)
#         self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, 2 * hidden_dim),
#             nn.GELU(),
#             nn.Linear(2 * hidden_dim, hidden_dim)
#         )
#     def forward(self, rgb_in, lidar_in):
#         # Self-attention
#         rgb_self, _ = self.attn(self.norm1(rgb_in), self.norm1(rgb_in), self.norm1(rgb_in), need_weights=False)
#         lidar_self, _ = self.attn(self.norm1(lidar_in), self.norm1(lidar_in), self.norm1(lidar_in), need_weights=False)
#         # Cross-attention
#         q_cam  = self.norm2(rgb_self)
#         k_lid  = self.norm2(lidar_self)
#         v_lid  = self.norm2(lidar_self)
#         cam_fused, _ = self.attn(q_cam, k_lid, v_lid, need_weights=False)
#         q_lid  = self.norm2(lidar_self)
#         k_cam  = self.norm2(rgb_self)
#         v_cam  = self.norm2(rgb_self)
#         lid_fused, _ = self.attn(q_lid, k_cam, v_cam, need_weights=False)
#         # FFN + residual
#         cam_out = cam_fused + self.ffn(self.norm3(cam_fused))
#         lid_out = lid_fused + self.ffn(self.norm3(lid_fused))
#         return cam_out, lid_out



# # class Blip2Qformer(nn.Module):  # (Blip2Base):
# class Blip2Qformer(Blip2Base):
#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#         "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
#     }

#     def __init__(
#         self,
#         vit_model=None,
#         img_size=224,
#         drop_path_rate=0,
#         use_grad_checkpoint=True,
#         vit_precision="fp16",
#         freeze_vit=True,
#         num_query_token=32,
#         cross_attention_freq=1,
#         embed_dim=256,
#         max_txt_len=32,
#         hidden_dim=256,
#         # num_layers=4,
#         num_layers=2,
#         heads=4,
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.use_grad_checkpoint = use_grad_checkpoint

#         self.rgb_layernorm = nn.LayerNorm(hidden_dim)
#         self.lidar_layernorm = nn.LayerNorm(hidden_dim)
#         self.vision_proj = nn.Linear(hidden_dim, embed_dim)
#         self.lidar_proj = nn.Linear(hidden_dim, embed_dim)
#         self.log_temp = nn.Parameter(torch.log(torch.tensor(0.1)))
#         self.temp = nn.Parameter(0.07 * torch.ones([]))
#         self.dropout = nn.Dropout(p=0.1)

#         self._cached_pos = None
#         self.num_layers = num_layers
#         self.heads = heads

#         # Stack 4 transformer-style fusion blocks
#         self.fusion_layers = nn.ModuleList([
#             FusionBlock(hidden_dim, heads) for _ in range(num_layers)
#         ])

#     def _build_2d_sincos_pos_embed(self, h, w, dim, device):
#         def get_1d_sin_cos_pos_embed(embed_dim, pos):
#             assert embed_dim % 2 == 0
#             omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
#             omega = 1. / (10000 ** (omega / (embed_dim // 2)))
#             pos = pos.reshape(-1, 1)
#             out = pos * omega.reshape(1, -1)
#             sin = torch.sin(out)
#             cos = torch.cos(out)
#             return torch.cat([sin, cos], dim=1)
#         grid_y = torch.arange(h, dtype=torch.float32, device=device)
#         grid_x = torch.arange(w, dtype=torch.float32, device=device)
#         grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)
#         grid = grid.reshape(2, -1).T
#         emb_y = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 0])
#         emb_x = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 1])
#         pos = torch.cat([emb_y, emb_x], dim=1)
#         return pos.unsqueeze(0)  # [1, h*w, dim]

#     def forward(self, samples, is_train=True):
#         image = samples["cam_bev"]      # [B, 256, H, W]
#         lidar = samples["lidar_bev"]    # [B, 256, H, W]
#         bs, c, H, W = image.shape

#         # 1. (optional projection)
#         rgb_proj   = image      # [B, D, H, W]
#         lidar_proj = lidar      # [B, D, H, W]

#         # 2. Flatten to tokens
#         rgb_tokens   = self.rgb_layernorm(rgb_proj.flatten(2).transpose(1, 2))
#         lidar_tokens = self.lidar_layernorm(lidar_proj.flatten(2).transpose(1, 2))

#         # 3. Positional encoding addition (applied to both)
#         if self._cached_pos is None or self._cached_pos.shape[1] != H * W:
#             self._cached_pos = self._build_2d_sincos_pos_embed(
#                 H, W, rgb_tokens.size(-1), rgb_tokens.device
#             )
#         pos_embed = self._cached_pos.to(rgb_tokens.device)
#         rgb_tokens = rgb_tokens + pos_embed   
#         lidar_tokens = lidar_tokens + pos_embed

#         # 4. Run 4 fusion layers
#         cam_out, lid_out = rgb_tokens, lidar_tokens
#         def fusion_stack(cam_in, lid_in):
#             for layer in self.fusion_layers:
#                 cam_in, lid_in = layer(cam_in, lid_in)
#             return cam_in, lid_in

#         if self.use_grad_checkpoint and is_train:
#             cam_out, lid_out = checkpoint(
#                 fusion_stack, cam_out, lid_out, use_reentrant=False
#             )
#         else:
#             cam_out, lid_out = fusion_stack(cam_out, lid_out)

#         # 5. Cross-reconstruction heads
#         lidar_rec = self.lidar_proj(cam_out)
#         cam_rec   = self.vision_proj(lid_out)

#         # 6. Discrepancy maps (element-wise L1 loss for each token)
#         lid_discrepancy_map = (lidar_rec - lidar_tokens).abs().mean(-1)
#         cam_discrepancy_map = (cam_rec - rgb_tokens).abs().mean(-1)

#         # 7. Scalar losses (mean over all tokens and batch, for training)
#         L_cross_lid = lid_discrepancy_map.mean()
#         L_cross_cam = cam_discrepancy_map.mean()
#         total_loss = L_cross_lid + L_cross_cam

#         return {
#             "loss": total_loss.view(1),
#             "L_cross_lid": L_cross_lid.detach(),
#             "L_cross_cam": L_cross_cam.detach(),
#             # "lid_discrepancy_map": lid_discrepancy_map.detach(),
#             # "cam_discrepancy_map": cam_discrepancy_map.detach(),
#         }



#     @classmethod
#     def from_config(cls, cfg):
#         vit_model = cfg.get("vit_model", "eva_clip_g")
#         img_size = cfg.get("image_size")
#         num_query_token = cfg.get("num_query_token")
#         cross_attention_freq = cfg.get("cross_attention_freq", 2)
#         drop_path_rate = cfg.get("drop_path_rate", 0)
#         use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
#         vit_precision = cfg.get("vit_precision", "fp16")
#         freeze_vit = cfg.get("freeze_vit", True)
#         max_txt_len = cfg.get("max_txt_len", 32)

#         model = cls(
#             vit_model=vit_model,
#             img_size=img_size,
#             drop_path_rate=drop_path_rate,
#             use_grad_checkpoint=use_grad_checkpoint,
#             vit_precision=vit_precision,
#             freeze_vit=freeze_vit,
#             num_query_token=num_query_token,
#             cross_attention_freq=cross_attention_freq,
#             max_txt_len=max_txt_len,
#         )
#         model.load_checkpoint_from_config(cfg)
#         return model



































import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint





# # # with  duplicated parameters and residuals but no partition 
# class FusionBlock(nn.Module):
#     def __init__(self, hidden_dim, heads):
#         super().__init__()
#         # self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm1_rgb = nn.LayerNorm(hidden_dim)
#         self.norm1_lidar = nn.LayerNorm(hidden_dim)


#         self.norm2_rgb = nn.LayerNorm(hidden_dim)
#         self.norm2_lidar = nn.LayerNorm(hidden_dim)
#         self.norm3_rgb = nn.LayerNorm(hidden_dim)
#         self.norm3_lidar = nn.LayerNorm(hidden_dim)

#         self.norm3 = nn.LayerNorm(hidden_dim)
#         # self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)


#         self.attn_rgb_self = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
#         self.attn_lidar_self = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
#         self.attn_cam_to_lidar = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
#         self.attn_lidar_to_cam = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)


#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, 2 * hidden_dim),
#             nn.GELU(),
#             nn.Linear(2 * hidden_dim, hidden_dim)
#         )
#     def forward(self, rgb_in, lidar_in, cam_mask=None, lidar_mask=None):
#         # Self-attention
#         rgb_self, _ = self.attn_rgb_self(self.norm1_rgb(rgb_in), self.norm1_rgb(rgb_in), self.norm1_rgb(rgb_in), need_weights=False)
#         rgb_self = rgb_in + rgb_self 
        
#         lidar_self, _ = self.attn_lidar_self(self.norm1_lidar(lidar_in), self.norm1_lidar(lidar_in), self.norm1_lidar(lidar_in), need_weights=False)
#         lidar_self = lidar_in + lidar_self 


#         # Cross-attention
#         q_cam  = self.norm2_rgb(rgb_self)
#         q_lid  = self.norm2_lidar(lidar_self)

#         # === apply random masking on features (zero out masked tokens) ===
#         if lidar_mask is not None:
#             # lidar_mask: [B, N] → [B, N, 1] for broadcasting
#             lidar_self = lidar_self.masked_fill(lidar_mask.unsqueeze(-1), 0.0)
#         if cam_mask is not None:
#             rgb_self = rgb_self.masked_fill(cam_mask.unsqueeze(-1), 0.0)

#         # commente the cross-attnion and ffn parts to test only self-attention   

#         # # Cross-attention
#         # k_lid  = self.norm2_lidar(lidar_self)
#         # v_lid  = self.norm2_lidar(lidar_self)
#         # cam_fused, _ = self.attn_cam_to_lidar(q_cam, k_lid, v_lid, need_weights=False)
#         # cam_fused = rgb_self + cam_fused
        
#         # # and for the reverse path:
#         # k_cam  = self.norm2_rgb(rgb_self)
#         # v_cam  = self.norm2_rgb(rgb_self)
#         # lid_fused, _ = self.attn_lidar_to_cam(q_lid, k_cam, v_cam, need_weights=False)
#         # lid_fused = lidar_self + lid_fused 

        
#         # # FFN + residual
#         # cam_out = cam_fused + self.ffn(self.norm3_rgb(cam_fused))
#         # lid_out = lid_fused + self.ffn(self.norm3_lidar(lid_fused))
        
#         # return cam_out, lid_out


#         return rgb_self, lidar_self



# # class Blip2Qformer(nn.Module):  # (Blip2Base):
# class Blip2Qformer(Blip2Base):
#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#         "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
#     }

#     def __init__(
#         self,
#         vit_model=None,
#         img_size=224,
#         drop_path_rate=0,
#         use_grad_checkpoint=True,
#         vit_precision="fp16",
#         freeze_vit=True,
#         num_query_token=32,
#         cross_attention_freq=1,
#         embed_dim=256,
#         max_txt_len=32,
#         hidden_dim=256,
#         # num_layers=4,
#         num_layers=4,
#         heads=4,
#         mask_ratio=0.2,
#         block_size=2,
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.use_grad_checkpoint = use_grad_checkpoint

#         self.rgb_layernorm = nn.LayerNorm(hidden_dim)
#         self.lidar_layernorm = nn.LayerNorm(hidden_dim)
#         self.vision_proj = nn.Linear(hidden_dim, embed_dim)
#         self.lidar_proj = nn.Linear(hidden_dim, embed_dim)
#         self.log_temp = nn.Parameter(torch.log(torch.tensor(0.1)))
#         self.temp = nn.Parameter(0.07 * torch.ones([]))
#         self.dropout = nn.Dropout(p=0.1)

#         self._cached_pos = None
#         self.num_layers = num_layers
#         self.heads = heads

#         # # Stack 4 transformer-style fusion blocks
#         self.fusion_layers = nn.ModuleList([
#             FusionBlock(hidden_dim, heads) for _ in range(num_layers)
#         ])

#         # H, W = 200, 200  # replace with your actual variable names
#         # self.fusion_layers = nn.ModuleList([
#         #     FusionBlock(hidden_dim, heads, H, W) for _ in range(num_layers)
#         # ])
#         self.mask_ratio = mask_ratio
#         self.block_size = block_size        

#     def _build_2d_sincos_pos_embed(self, h, w, dim, device):
#         def get_1d_sin_cos_pos_embed(embed_dim, pos):
#             assert embed_dim % 2 == 0
#             omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
#             omega = 1. / (10000 ** (omega / (embed_dim // 2)))
#             pos = pos.reshape(-1, 1)
#             out = pos * omega.reshape(1, -1)
#             sin = torch.sin(out)
#             cos = torch.cos(out)
#             return torch.cat([sin, cos], dim=1)
#         grid_y = torch.arange(h, dtype=torch.float32, device=device)
#         grid_x = torch.arange(w, dtype=torch.float32, device=device)
#         grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)
#         grid = grid.reshape(2, -1).T
#         emb_y = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 0])
#         emb_x = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 1])
#         pos = torch.cat([emb_y, emb_x], dim=1)
#         return pos.unsqueeze(0)  # [1, h*w, dim]



#     def make_block_mask(self, batch_size, H, W, block_size, mask_ratio, device):
#         """
#         Returns a bool mask of shape [B, H*W], where True = masked token.
#         Mask is made of random square blocks of size block_size x block_size.
#         """
#         mask = torch.zeros(batch_size, H, W, dtype=torch.bool, device=device)

#         # how many blocks per sample (rough approximation)
#         area = H * W
#         block_area = block_size * block_size
#         num_blocks = max(1, int(mask_ratio * area / block_area))

#         for b in range(batch_size):
#             for _ in range(num_blocks):
#                 y0 = torch.randint(0, max(1, H - block_size + 1), (1,), device=device).item()
#                 x0 = torch.randint(0, max(1, W - block_size + 1), (1,), device=device).item()
#                 mask[b, y0:y0+block_size, x0:x0+block_size] = True

#         return mask.view(batch_size, -1)  # [B, H*W]



#     def forward(self, samples, is_train=True):
#         # TEMP SAVE TEST (remove later)
#         # print("hhhhhhhhhhhhhhhhhhhhhhhhhhh")
#         save_dir = "/home/draiman/Desktop/debug_cam_features"
#         os.makedirs(save_dir, exist_ok=True)
#         test_path = os.path.join(save_dir, "test_save.pt")
#         torch.save(torch.tensor([1,2,3]), test_path)

#         # -------------------------------------
#         # 1. Load BEV feature maps [B, C, H, W]
#         # -------------------------------------
#         image = samples["cam_bev"]       # [B, 256, H, W]
#         lidar = samples["lidar_bev"]     # [B, 256, H, W]
#         bs, c, H, W = image.shape


        
        

#         # First sample only (to avoid massive files)
#         lidar_feat_to_save = lidar[0].detach().cpu()

#         save_path = os.path.join(save_dir, "lidar_features.pt")
#         torch.save(lidar_feat_to_save, save_path)

#         # First sample only (to avoid massive files)
#         cam_feat_to_save = image[0].detach().cpu()

#         save_path = os.path.join(save_dir, "cam_features.pt")
#         torch.save(cam_feat_to_save, save_path)


#         print(f"[DEBUG] Saved camera & lidar BEV features to: {save_path}")
#         # ==========================================


#         # -------------------------------------
#         # 2. Flatten into tokens -> [B, N, D]
#         # -------------------------------------
#         # raw features (before positional encoding)
#         rgb_feat_raw   = image.flatten(2).transpose(1, 2)     # [B, N, D]
#         lidar_feat_raw = lidar.flatten(2).transpose(1, 2)     # [B, N, D]

#         # layernorm the raw features
#         rgb_tokens_raw   = self.rgb_layernorm(rgb_feat_raw)
#         lidar_tokens_raw = self.lidar_layernorm(lidar_feat_raw)

#         N = rgb_tokens_raw.size(1)

#         # -------------------------------------
#         # 3. Add 2D positional embedding
#         # -------------------------------------
#         if self._cached_pos is None or self._cached_pos.shape[1] != H * W:
#             self._cached_pos = self._build_2d_sincos_pos_embed(
#                 H, W, rgb_tokens_raw.size(-1), rgb_tokens_raw.device
#             )
#         pos_embed = self._cached_pos.to(rgb_tokens_raw.device)   # [1, N, D]

#         rgb_tokens   = rgb_tokens_raw   + pos_embed
#         lidar_tokens = lidar_tokens_raw + pos_embed

#         # -------------------------------------
#         # 4. No masking in the baseline
#         # -------------------------------------
#         cam_mask = None
#         lidar_mask = None

#         # -------------------------------------
#         # 5. Run ONLY self-attention layers
#         # -------------------------------------
#         def fusion_stack(cam_in, lid_in):
#             for layer in self.fusion_layers:
#                 # self-attention only (cross-attn removed inside FusionBlock)
#                 cam_in, lid_in = layer(cam_in, lid_in, cam_mask, lidar_mask)
#             return cam_in, lid_in

#         if self.use_grad_checkpoint and is_train:
#             cam_out, lid_out = checkpoint(
#                 fusion_stack, rgb_tokens, lidar_tokens, use_reentrant=False
#             )
#         else:
#             cam_out, lid_out = fusion_stack(rgb_tokens, lidar_tokens)

#         # -------------------------------------
#         # 6. Feature reconstruction heads
#         # -------------------------------------
#         # Project back to original feature dimension
#         lidar_rec = self.lidar_proj(lid_out)  # LiDAR reconstructs LiDAR features
#         cam_rec   = self.vision_proj(cam_out) # Camera reconstructs camera features

#         # -------------------------------------
#         # 7. Reconstruction loss: MSE on FEATURES
#         # -------------------------------------
#         mse = torch.nn.MSELoss()

#         L_recon_lidar = mse(lidar_rec, lidar_feat_raw)   # LiDAR → LiDAR
#         L_recon_cam   = mse(cam_rec, rgb_feat_raw)       # Camera → Camera

#         total_loss = L_recon_lidar + L_recon_cam







#     # ================================
#     # DEBUG PRINT (only first iter/epoch)
#     # ================================

#         def stats(name, x):
#             print(f"[STATS] {name}: min={x.min().item():.4f}, "
#                 f"max={x.max().item():.4f}, "
#                 f"mean={x.mean().item():.4f}, "
#                 f"std={x.std().item():.4f}")

#         print("\n" + "="*80)
#         print("[DEBUG] Feature Reconstruction Check")

#         # ---- shapes ----
#         print("[SHAPES]")
#         print("  cam_rec:", cam_rec.shape)
#         print("  cam_raw:", rgb_feat_raw.shape)
#         print("  lidar_rec:", lidar_rec.shape)
#         print("  lidar_raw:", lidar_feat_raw.shape)



        
#         # # Choose save directory
#         # save_dir = "/home/draiman/Desktop/debug_cam_features"
#         # os.makedirs(save_dir, exist_ok=True)
#         # save_path = os.path.join(save_dir, f"cam_features.pt")



#         # ---- first 5 tokens × first 5 dims ----
#         print("\n[CAMERA RAW (first 5 tokens × 5 dims)]")
#         print(rgb_feat_raw[0, :5, :5])

#         print("\n[CAMERA REC (first 5 tokens × 5 dims)]")
#         print(cam_rec[0, :5, :5])

#         print("\n[LIDAR RAW (first 5 tokens × 5 dims)]")
#         print(lidar_feat_raw[0, :5, :5])

#         print("\n[LIDAR REC (first 5 tokens × 5 dims)]")
#         print(lidar_rec[0, :5, :5])

#         # ---- per-token MAE ----
#         cam_mae = (cam_rec[0] - rgb_feat_raw[0]).abs().mean(dim=-1)
#         lid_mae = (lidar_rec[0] - lidar_feat_raw[0]).abs().mean(dim=-1)

#         print("\n[Per-token MAE CAMERA (first 10 tokens)]")
#         print(cam_mae[:10])

#         print("\n[Per-token MAE LIDAR (first 10 tokens)]")
#         print(lid_mae[:10])

#         # ---- stats ----
#         print("\n[STATISTICS]")
#         stats("cam_raw", rgb_feat_raw[0])
#         stats("cam_rec", cam_rec[0])
#         stats("lidar_raw", lidar_feat_raw[0])
#         stats("lidar_rec", lidar_rec[0])

#         print("="*80 + "\n")
#     # ================================








#         # -------------------------------------
#         # 8. Return clean dictionary
#         # -------------------------------------
#         return {
#             "loss": total_loss.view(1),
#             "L_recon_lidar": L_recon_lidar.detach(),
#             "L_recon_cam": L_recon_cam.detach(),
#         }


#     @torch.no_grad()
#     def forward_features(self, samples, return_maps=False):
#         print(">>> Using forward_features()  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#         return_maps=True
#         """
#         Evaluation / testing forward:
#         - NO gradient checkpointing
#         - NO training loss
#         - Returns discrepancy maps + scalar scores you can threshold / log.

#         Args:
#             samples: dict with
#                 samples["cam_bev"]   -> [B, D, H, W]
#                 samples["lidar_bev"] -> [B, D, H, W]
#             return_maps (bool):
#                 If True, also return [B, H, W] discrepancy maps.

#         Returns:
#             dict with keys:
#                 "L_cross_lid" : scalar tensor (mean L1 recon error for LiDAR)
#                 "L_cross_cam" : scalar tensor (mean L1 recon error for camera)
#                 "lid_discrepancy_map": [B, H, W] (if return_maps)
#                 "cam_discrepancy_map": [B, H, W] (if return_maps)
#                 "cam_tokens":  [B, H*W, D] fused camera tokens
#                 "lid_tokens":  [B, H*W, D] fused lidar tokens
#         """
#         image = samples["cam_bev"]      # [B, D, H, W]
#         lidar = samples["lidar_bev"]    # [B, D, H, W]
#         bs, c, H, W = image.shape

#         # 1. (optional projection) – here you use BEV features directly
#         rgb_proj   = image      # [B, D, H, W]
#         lidar_proj = lidar      # [B, D, H, W]

#         # 2. Flatten to tokens + LayerNorm
#         rgb_tokens_raw   = self.rgb_layernorm(rgb_proj.flatten(2).transpose(1, 2))   # [B, H*W, D]
#         lidar_tokens_raw = self.lidar_layernorm(lidar_proj.flatten(2).transpose(1, 2))  # [B, H*W, D]

#         # 3. Positional encoding (same as in forward)
#         if self._cached_pos is None or self._cached_pos.shape[1] != H * W:
#             self._cached_pos = self._build_2d_sincos_pos_embed(
#                 H, W, rgb_tokens_raw.size(-1), rgb_tokens_raw.device
#             )
#         pos_embed = self._cached_pos.to(rgb_tokens_raw.device)   # [1, H*W, D]

#         rgb_tokens   = rgb_tokens_raw + pos_embed
#         lidar_tokens = lidar_tokens_raw + pos_embed

#         # 4. Run fusion stack (NO checkpointing in eval)
#         cam_out, lid_out = rgb_tokens, lidar_tokens
#         for layer in self.fusion_layers:
#             cam_out, lid_out = layer(cam_out, lid_out)   # [B, H*W, D] each

#         # 5. Cross-reconstruction heads
#         lidar_rec = self.lidar_proj(cam_out)   # [B, H*W, embed_dim]
#         cam_rec   = self.vision_proj(lid_out)  # [B, H*W, embed_dim]

#         # 6. Discrepancy maps (per-token L1)
#         lid_discrepancy_map_flat = (lidar_rec - lidar_tokens_raw).abs().mean(-1)  # [B, H*W]
#         cam_discrepancy_map_flat = (cam_rec   - rgb_tokens_raw  ).abs().mean(-1)  # [B, H*W]

#         # 7. Scalar scores (mean over tokens)
#         L_cross_lid = lid_discrepancy_map_flat.mean()  # scalar
#         L_cross_cam = cam_discrepancy_map_flat.mean()  # scalar

#         out = {
#             "L_cross_lid": L_cross_lid,
#             "L_cross_cam": L_cross_cam,
#             "cam_tokens": cam_out,     # [B, H*W, D]
#             "lid_tokens": lid_out,     # [B, H*W, D]
#         }

#         if return_maps:
#             lid_discrepancy_map = lid_discrepancy_map_flat.view(bs, H, W)  # [B, H, W]
#             cam_discrepancy_map = cam_discrepancy_map_flat.view(bs, H, W)  # [B, H, W]
#             out["lid_discrepancy_map"] = lid_discrepancy_map
#             out["cam_discrepancy_map"] = cam_discrepancy_map

#         return out


#     @classmethod
#     def from_config(cls, cfg):
#         vit_model = cfg.get("vit_model", "eva_clip_g")
#         img_size = cfg.get("image_size")
#         num_query_token = cfg.get("num_query_token")
#         cross_attention_freq = cfg.get("cross_attention_freq", 2)
#         drop_path_rate = cfg.get("drop_path_rate", 0)
#         use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
#         vit_precision = cfg.get("vit_precision", "fp16")
#         freeze_vit = cfg.get("freeze_vit", True)
#         max_txt_len = cfg.get("max_txt_len", 32)

#         model = cls(
#             vit_model=vit_model,
#             img_size=img_size,
#             drop_path_rate=drop_path_rate,
#             use_grad_checkpoint=use_grad_checkpoint,
#             vit_precision=vit_precision,
#             freeze_vit=freeze_vit,
#             num_query_token=num_query_token,
#             cross_attention_freq=cross_attention_freq,
#             max_txt_len=max_txt_len,
#         )
#         model.load_checkpoint_from_config(cfg)
#         return model


















































# -------------------------------
# Enhanced Loss Functions
# -------------------------------
def l1_cosine_loss(pred, target):
    """Combine L1 + Cosine similarity losses."""
    l1 = F.l1_loss(pred, target)
    cosine = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return l1 + 0.1 * cosine

def local_infonce_loss(features1, features2, temperature=0.1):
    """InfoNCE on local BEV tokens."""
    B, N, D = features1.shape
    f1 = F.normalize(features1, dim=-1)
    f2 = F.normalize(features2, dim=-1)
    sim = torch.matmul(f1, f2.transpose(1, 2)) / temperature  # [B, N, N]
    labels = torch.arange(N, device=features1.device)
    loss = F.cross_entropy(sim, labels.expand(B, -1))
    return loss.mean()

def mae_loss(recon, target, mask):
    """Masked Autoencoder loss."""
    diff = (recon - target) ** 2
    return (diff * mask.unsqueeze(-1)).sum() / mask.sum()

# def shuffled_infonce_loss(features1, features2, temperature=0.1):
#     """
#     InfoNCE compares each token in features1 to a randomly shuffled token in features2.
#     Instead of matching by spatial index, we randomly permute the correspondence
#     for each batch.
#     """
#     B, N, D = features1.shape
#     f1 = F.normalize(features1, dim=-1)
#     f2 = F.normalize(features2, dim=-1)
#     sim = torch.matmul(f1, f2.transpose(1, 2)) / temperature  # [B, N, N]
#     losses = []

#     for b in range(B):
#         # Get a random permutation for each batch
#         perm = torch.randperm(N, device=features1.device)
#         # Each token in features1 is now paired with permuted token in features2
#         labels = perm
#         # Cross-entropy expects shape [N, N] and labels [N]
#         loss = F.cross_entropy(sim[b], labels)
#         losses.append(loss)

#     return torch.stack(losses).mean()

def shuffled_infonce_with_neg_sampling(features1, features2, temperature=0.1, num_negatives=32):
    B, N, D = features1.shape
    f1 = F.normalize(features1, dim=-1)
    f2 = F.normalize(features2, dim=-1)
    losses = []
    for b in range(B):
        # Random permutation for positives
        perm = torch.randperm(N, device=features1.device)
        pos_sim = torch.sum(f1[b] * f2[b][perm], dim=-1) / temperature  # [N]
        logits_list = []
        for i in range(N):
            # Sample random negative indices, excluding the positive index
            negatives = [idx for idx in range(N) if idx != perm[i]]
            neg_indices = torch.tensor(np.random.choice(negatives, size=num_negatives, replace=False), device=f1.device)
            neg_sim = torch.matmul(f1[b][i].unsqueeze(0), f2[b][neg_indices].transpose(0, 1)).squeeze(0) / temperature  # [num_negatives]
            logits = torch.cat([pos_sim[i].unsqueeze(0), neg_sim], dim=0)  # [1 + num_negatives]
            logits_list.append(logits)
        logits_all = torch.stack(logits_list, dim=0)
        labels = torch.zeros(N, dtype=torch.long, device=f1.device)  # Positive is always first in logits
        loss = F.cross_entropy(logits_all, labels)
        losses.append(loss)
    return torch.stack(losses).mean()


# -------------------------------
# Example Training Step
# -------------------------------
def compute_total_loss(pred_cam, pred_lid, target_cam, target_lid, mask=None):
    """Full loss: L1 + Cosine + Local InfoNCE + MAE"""
    L_recon_cam = l1_cosine_loss(pred_cam, target_cam)
    L_recon_lid = l1_cosine_loss(pred_lid, target_lid)
    L_infonce = local_infonce_loss(pred_cam, pred_lid)
    L_mae = mae_loss(pred_cam, target_cam, mask) if mask is not None else 0.0

    total = L_recon_cam + L_recon_lid + 0.5 * L_infonce + 0.1 * L_mae
    return total
























































# Main differences with previous version:
# - no masking implemented yet
# - changed: using L1 loss to using MAE
# - changed: reconstruction heads to predict projected features instead of tokens
# - changed: removed cross-attention and FFN from FusionBlock, only self-attention remains
# - changed: fusion_layers from 2 layers to 4 layers








































































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





# class Blip2Qformer(nn.Module):  # (Blip2Base):
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



    def make_block_mask(self, batch_size, H, W, block_size, mask_ratio, device):
        """
        Returns a bool mask of shape [B, H*W], where True = masked token.
        Mask is made of random square blocks of size block_size x block_size.
        """
        mask = torch.zeros(batch_size, H, W, dtype=torch.bool, device=device)

        # how many blocks per sample (rough approximation)
        area = H * W
        block_area = block_size * block_size
        num_blocks = max(1, int(mask_ratio * area / block_area))

        for b in range(batch_size):
            for _ in range(num_blocks):
                y0 = torch.randint(0, max(1, H - block_size + 1), (1,), device=device).item()
                x0 = torch.randint(0, max(1, W - block_size + 1), (1,), device=device).item()
                mask[b, y0:y0+block_size, x0:x0+block_size] = True

        return mask.view(batch_size, -1)  # [B, H*W]

    def forward(self, samples, is_train=True):
        cam_bev = samples["cam_bev"]   # [B, C, H, W]
        lid_bev = samples["lidar_bev"] # [B, C, H, W]
        B, C, H, W = cam_bev.shape

        # -----------------------------------------------------
        # 1. Flatten to tokens + add positional encoding
        # -----------------------------------------------------
        cam_tokens = cam_bev.flatten(2).transpose(1,2)  # [B, N, D]
        lid_tokens = lid_bev.flatten(2).transpose(1,2)

        cam_tokens = self.rgb_layernorm(cam_tokens)
        lid_tokens = self.lidar_layernorm(lid_tokens)

        # Positional embedding (same shape for both)
        if self._cached_pos is None or self._cached_pos.shape[1] != (H*W):
            self._cached_pos = self._build_2d_sincos_pos_embed(H, W, C, cam_tokens.device)
        pos = self._cached_pos

        cam_tokens = cam_tokens + pos
        lid_tokens = lid_tokens + pos

        # -----------------------------------------------------
        # 2. Transformer stack (self-attn only in Phase 2)
        # -----------------------------------------------------
        for layer in self.fusion_layers:
            cam_tokens, lid_tokens = layer(cam_tokens, lid_tokens)

        # -----------------------------------------------------
        # 3. Reshape tokens back to BEV grid
        # -----------------------------------------------------
        cam_latent = cam_tokens.transpose(1,2).view(B, C, H, W)
        lid_latent = lid_tokens.transpose(1,2).view(B, C, H, W)

        # -----------------------------------------------------
        # 4. CNN Decoder
        # -----------------------------------------------------
        cam_rec = self.cam_decoder(cam_latent)
        lid_rec = self.lidar_decoder(lid_latent)

        # -----------------------------------------------------
        # 5. Reconstruction Loss in BEV space
        # -----------------------------------------------------
        loss_cam = F.mse_loss(cam_rec, cam_bev)
        loss_lid = F.mse_loss(lid_rec, lid_bev)
        total_loss = loss_cam + loss_lid

        return {
            "loss": total_loss,
            "loss_cam": loss_cam.detach(),
            "loss_lid": loss_lid.detach(),
        }







    # @torch.no_grad()
    # def forward_features(self, samples, return_maps=False):
    #     print(">>> Using forward_features()  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     return_maps=True
    #     """
    #     Evaluation / testing forward:
    #     - NO gradient checkpointing
    #     - NO training loss
    #     - Returns discrepancy maps + scalar scores you can threshold / log.

    #     Args:
    #         samples: dict with
    #             samples["cam_bev"]   -> [B, D, H, W]
    #             samples["lidar_bev"] -> [B, D, H, W]
    #         return_maps (bool):
    #             If True, also return [B, H, W] discrepancy maps.

    #     Returns:
    #         dict with keys:
    #             "L_cross_lid" : scalar tensor (mean L1 recon error for LiDAR)
    #             "L_cross_cam" : scalar tensor (mean L1 recon error for camera)
    #             "lid_discrepancy_map": [B, H, W] (if return_maps)
    #             "cam_discrepancy_map": [B, H, W] (if return_maps)
    #             "cam_tokens":  [B, H*W, D] fused camera tokens
    #             "lid_tokens":  [B, H*W, D] fused lidar tokens
    #     """
    #     image = samples["cam_bev"]      # [B, D, H, W]
    #     lidar = samples["lidar_bev"]    # [B, D, H, W]
    #     bs, c, H, W = image.shape

    #     # 1. (optional projection) – here you use BEV features directly
    #     rgb_proj   = image      # [B, D, H, W]
    #     lidar_proj = lidar      # [B, D, H, W]

    #     # 2. Flatten to tokens + LayerNorm
    #     rgb_tokens_raw   = self.rgb_layernorm(rgb_proj.flatten(2).transpose(1, 2))   # [B, H*W, D]
    #     lidar_tokens_raw = self.lidar_layernorm(lidar_proj.flatten(2).transpose(1, 2))  # [B, H*W, D]

    #     # 3. Positional encoding (same as in forward)
    #     if self._cached_pos is None or self._cached_pos.shape[1] != H * W:
    #         self._cached_pos = self._build_2d_sincos_pos_embed(
    #             H, W, rgb_tokens_raw.size(-1), rgb_tokens_raw.device
    #         )
    #     pos_embed = self._cached_pos.to(rgb_tokens_raw.device)   # [1, H*W, D]

    #     rgb_tokens   = rgb_tokens_raw + pos_embed
    #     lidar_tokens = lidar_tokens_raw + pos_embed

    #     # 4. Run fusion stack (NO checkpointing in eval)
    #     cam_out, lid_out = rgb_tokens, lidar_tokens
    #     for layer in self.fusion_layers:
    #         cam_out, lid_out = layer(cam_out, lid_out)   # [B, H*W, D] each

    #     # 5. Cross-reconstruction heads
    #     lidar_rec = self.lidar_proj(cam_out)   # [B, H*W, embed_dim]
    #     cam_rec   = self.vision_proj(lid_out)  # [B, H*W, embed_dim]

    #     # 6. Discrepancy maps (per-token L1)
    #     lid_discrepancy_map_flat = (lidar_rec - lidar_tokens_raw).abs().mean(-1)  # [B, H*W]
    #     cam_discrepancy_map_flat = (cam_rec   - rgb_tokens_raw  ).abs().mean(-1)  # [B, H*W]

    #     # 7. Scalar scores (mean over tokens)
    #     L_cross_lid = lid_discrepancy_map_flat.mean()  # scalar
    #     L_cross_cam = cam_discrepancy_map_flat.mean()  # scalar

    #     out = {
    #         "L_cross_lid": L_cross_lid,
    #         "L_cross_cam": L_cross_cam,
    #         "cam_tokens": cam_out,     # [B, H*W, D]
    #         "lid_tokens": lid_out,     # [B, H*W, D]
    #     }

    #     if return_maps:
    #         lid_discrepancy_map = lid_discrepancy_map_flat.view(bs, H, W)  # [B, H, W]
    #         cam_discrepancy_map = cam_discrepancy_map_flat.view(bs, H, W)  # [B, H, W]
    #         out["lid_discrepancy_map"] = lid_discrepancy_map
    #         out["cam_discrepancy_map"] = cam_discrepancy_map

    #     return out


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



