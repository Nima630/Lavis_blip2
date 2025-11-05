
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

#         q = self.to_q(x).view(B, N, H, D // H).transpose(1, 2)  # [B, H, N, D/H]
#         k = self.to_k(context).view(B, -1, H, D // H).transpose(1, 2)  # [B, H, S, D/H]
#         v = self.to_v(context).view(B, -1, H, D // H).transpose(1, 2)  # [B, H, S, D/H]

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, S]
#         attn = attn.softmax(dim=-1)

#         out = attn @ v  # [B, H, N, D/H]
#         out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
#         return self.to_out(out)

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
#         self.ln2 = nn.LayerNorm(dim)
#         self.cross_attn = CrossAttention(dim, heads, dropout)
#         self.ln3 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x, context):
#         x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
#         x = x + self.cross_attn(self.ln2(x), context)
#         x = x + self.mlp(self.ln3(x))
#         return x


# class MiniQFormerDecoder(nn.Module):
#     def __init__(self, hidden_size=256, num_hidden_layers=3, num_attention_heads=4, cross_attention_freq=2):
#         super().__init__()

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=num_attention_heads,
#             dim_feedforward=hidden_size * 4,
#             batch_first=True,
#         )

#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
#         self.cross_attention_freq = cross_attention_freq

#     def forward(self, query_embeds, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
#         # Simple concat
#         x = torch.cat([query_embeds, encoder_hidden_states], dim=1)
#         x = self.encoder(x)

#         class Output:
#             def __init__(self, last_hidden_state):
#                 self.last_hidden_state = last_hidden_state

#         return Output(last_hidden_state=x[:, :query_embeds.shape[1]])




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
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=1,
        embed_dim=256,
        max_txt_len=32,):
        super().__init__()

        # hidden_dim = 768
        hidden_dim = 256
        self.rgb_input_proj = self.init_feature_projection(in_dim=2048, out_dim=hidden_dim) # resnet
        # self.rgb_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim) # fpn
        self.lidar_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim)
        
        self.rgb_layernorm = nn.LayerNorm(hidden_dim)
        self.lidar_layernorm = nn.LayerNorm(hidden_dim)

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
        qformer_lidar, query_tokens_lidar = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
        self.Qformer_lidar = qformer_lidar
        self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
        self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)

        self.vision_proj = nn.Linear(hidden_dim, embed_dim)
        self.lidar_proj = nn.Linear(hidden_dim, embed_dim)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.1)))  # Learnable temp
        # print(self.Qformer)
        # print(self.Qformer_lidar)
        # print("sanity check")


        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.dropout = nn.Dropout(p=0.1)

    def init_feature_projection(self, in_dim, out_dim):
        return nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def _diversity_loss(self, features):
        """Penalizes collapsed feature representations.
        Args:
            features: [batch_size, feature_dim]
        Returns:
            loss: scalar tensor
        """
        # Normalize features first to get valid cosine similarity
        features = F.normalize(features, dim=-1)
        sim_matrix = torch.mm(features, features.t())  # [B,B]
        # print("---------------------- sim_matrix", sim_matrix)
        # Mask to exclude diagonal (self-similarities)
        mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        avg_sim = sim_matrix[mask].mean()
        
        # Target 0.0 average similarity between different samples
        return torch.abs(avg_sim)


    # # forward mini
    def forward(self, samples, is_train=True):
        
        shuffle_prob = 1 #0.5
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        
        # projection
        rgb_proj = self.rgb_input_proj(image)

        lidar_feat = lidar.squeeze(1)
        lidar_proj = self.lidar_input_proj(lidar_feat)


        # if is_train:
        #     rgb_proj = self.dropout(rgb_proj)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)





        # if is_train:
        #     lidar_proj = self.dropout(lidar_proj)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)


        rgb_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

        lidar_output = self.Qformer_lidar(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)


        rgb_avg = rgb_feats.mean(dim=1)  # [B, D]
        lidar_avg = lidar_feats.mean(dim=1)  # [B, D]

        if is_train and shuffle_prob > 0: 
            device = image.device
            perm = torch.randperm(bs, device=device)
            keep_mask = torch.rand(bs, device=device) > shuffle_prob

            # print(f"[DEBUG] perm: {perm.tolist()}")
            # print(f"[DEBUG] keep_mask: {keep_mask.tolist()}")
            
            lidar_avg_shuffled = lidar_avg.clone()
            lidar_avg_shuffled[~keep_mask] = lidar_avg[perm[~keep_mask]]

            targets = torch.arange(bs, device=device)
            targets[~keep_mask] = perm[~keep_mask]

            # print(f"[DEBUG] targets: {targets.tolist()}")
            
            num_mismatched = (~keep_mask).sum().item()
            # print(f"[DEBUG] Total mismatches introduced: {num_mismatched}")

            
        else:
            lidar_avg_shuffled = lidar_avg
            targets = torch.arange(bs, device=image.device)
        # print(f"--------------------------------------------------------------------")

        # # === Compute contrastive loss ===
        sim_matrix = (rgb_avg @ lidar_avg_shuffled.T) * torch.exp(self.log_temp)
        # print("sim_matrix shape:  sim_matrix = (rgb_avg @ lidar_avg.T) * torch.exp(self.log_temp)", sim_matrix.shape)

        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2
        # print("[DEBUG] loss_contrastive = ", loss_contrastive)

        return BlipOutput(loss=loss_contrastive)



    def _build_2d_sincos_pos_embed(self, h, w, dim, device):
        """
        Returns [1, h*w, dim] 2D sin-cos positional embedding.
        """
        import math

        def get_1d_sin_cos_pos_embed(embed_dim, pos):
            # embed_dim must be even
            assert embed_dim % 2 == 0
            omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
            omega = 1. / (10000 ** (omega / (embed_dim // 2)))
            pos = pos.reshape(-1, 1)  # [M, 1]
            out = pos * omega.reshape(1, -1)  # [M, embed_dim/2]
            sin = torch.sin(out)
            cos = torch.cos(out)
            return torch.cat([sin, cos], dim=1)  # [M, embed_dim]

        grid_y = torch.arange(h, dtype=torch.float32, device=device)
        grid_x = torch.arange(w, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)  # [2, h, w]
        grid = grid.reshape(2, -1).T  # [h*w, 2]

        emb_y = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 0])
        emb_x = get_1d_sin_cos_pos_embed(dim // 2, grid[:, 1])
        pos = torch.cat([emb_y, emb_x], dim=1)  # [h*w, dim]
        return pos.unsqueeze(0)  # [1, h*w, dim]


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

