
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
from lavis.runners.log_utils import log_alignment_stats
# from lavis.models.mini_qformer import MiniQFormer



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



    def forward(self, samples, is_train=True):
        print("forward   ")
        
        shuffle_prob = 0.5
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        print('image shape', image.shape)
        print('lidar shape', lidar.shape)
       
        rgb_proj = self.rgb_input_proj(image)
        print('rgb_proj shape from: rgb_proj = self.rgb_input_proj(rgb_feat) ', rgb_proj.shape)

        # if is_train:
        #     rgb_proj = self.dropout(rgb_proj)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        print('rgb_embeds shape from: rgb_embeds = rgb_proj.flatten(2).transpose(1, 2) ', rgb_embeds.shape)

        rgb_embeds = self.rgb_layernorm(rgb_embeds)
        print('rgb_embeds shape from: rgb_embeds = self.rgb_layernorm(rgb_embeds) ', rgb_embeds.shape)

        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        print('rgb_atts shape from: rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device) ', rgb_atts.shape)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        print('query_tokens shape from: query_tokens = self.query_tokens.expand(bs, -1, -1) ', query_tokens.shape)

        rgb_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )

        print("rgb_output = self.Qformer(..) shape: rgb_output.last_hidden_state.shape ", rgb_output.last_hidden_state.shape)


        rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)
        print("rgb_feats shape: rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1) ", rgb_feats.shape)

        lidar_feat = lidar.squeeze(1)
        print('lidar_feat shape from: lidar_feat = lidar.squeeze(1)', lidar_feat.shape)
        
        lidar_proj = self.lidar_input_proj(lidar_feat)
        print('lidar_proj shape from: lidar_proj = self.lidar_input_proj(lidar_feat) ', lidar_proj.shape)

        # if is_train:
        #     lidar_proj = self.dropout(lidar_proj)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        print('lidar_embeds shape from: lidar_embeds = lidar_proj.flatten(2).transpose(1, 2) ', lidar_embeds.shape)

        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        print('lidar_embeds shape from: lidar_embeds = self.lidar_layernorm(lidar_embeds) ', lidar_embeds.shape)
        
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        print('lidar_atts shape from: lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device) ', lidar_atts.shape)

        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)
        print('query_tokens_lidar shape from: query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1) ', query_tokens_lidar.shape)

        lidar_output = self.Qformer_lidar(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        print("lidar_output.last_hidden_state shape: lidar_output = self.Qformer_lidar(..)", lidar_output.last_hidden_state.shape)

        lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)
        print("lidar_feats shape: lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1) ", lidar_feats.shape)


        rgb_avg = rgb_feats.mean(dim=1)  # [B, D]
        print("rgb_avg shape: rgb_avg = rgb_feats.mean(dim=1)", rgb_avg.shape)

        lidar_avg = lidar_feats.mean(dim=1)  # [B, D]
        print("lidar_avg shape: lidar_avg = lidar_feats.mean(dim=1) ", lidar_avg.shape)


        # print("[DEBUG] rgb_output stats before proj:", rgb_output.last_hidden_state.mean().item(), rgb_output.last_hidden_state.std().item())
        # rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)
        # rgb_feats = self.vision_proj(rgb_output.last_hidden_state)
        # print("[DEBUG] rgb_feats stats after proj:", rgb_feats.mean().item(), rgb_feats.std().item())

        # print("[DEBUG] lidar_output stats before proj:", lidar_output.last_hidden_state.mean().item(), lidar_output.last_hidden_state.std().item())
        # lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)
        # lidar_feats = self.lidar_proj(lidar_output.last_hidden_state)
        # print("[DEBUG] lidar_feats stats after proj:", lidar_feats.mean().item(), lidar_feats.std().item())


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
        print(f"--------------------------------------------------------------------")

        # # === Compute contrastive loss ===
        sim_matrix = (rgb_avg @ lidar_avg_shuffled.T) * torch.exp(self.log_temp)
        print("sim_matrix shape:  sim_matrix = (rgb_avg @ lidar_avg.T) * torch.exp(self.log_temp)", sim_matrix.shape)


        # diag_sim = sim_matrix.diag()
        # true_pos_indices = keep_mask.nonzero(as_tuple=True)[0]
        # mean_true_diag_sim = diag_sim[true_pos_indices].mean().item()
        # if len(true_pos_indices) > 0:
        #     mean_true_diag_sim = diag_sim[true_pos_indices].mean().item()
        # else:
        #     mean_true_diag_sim = float("nan")
        # print(f"[DEBUG] True match diag sim: {mean_true_diag_sim:.4f}")

    

        # if torch.isnan(rgb_output.last_hidden_state).any():
        #     print("[DEBUG] NaNs in rgb_output")
        # if torch.isnan(lidar_output.last_hidden_state).any():
        #     print("[DEBUG] NaNs in lidar_output")

        # if torch.isnan(rgb_feats).any() or torch.isnan(lidar_feats).any():
        #     print("[DEBUG] NaNs in pooled features")

        # sim_matrix = F.cosine_similarity(rgb_avg.unsqueeze(1), lidar_avg.unsqueeze(0), dim=-1)

        # if torch.isnan(sim_matrix).any():
        #     print("[DEBUG] NaNs in sim_matrix")


        # if torch.isnan(rgb_avg).any() or torch.isnan(lidar_avg).any():
        #     print("[DEBUG] NaN detected in rgb_avg or lidar_avg")
        # if torch.isnan(sim_matrix).any():
        #     print("[DEBUG] NaN detected in sim_matrix")
        # print(f"[DEBUG] log_temp: {self.log_temp.item()}, exp(log_temp): {torch.exp(self.log_temp).item()}")
        print("targets", targets)


        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2
        # print("[DEBUG] loss_contrastive = ", loss_contrastive)

        return BlipOutput(loss=loss_contrastive)


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







    def forward_features(self, samples, is_train=True):
        print("forward_features")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)

        # === RGB branch ===
        rgb_proj = self.rgb_input_proj(image)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        rgb_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

        # === LiDAR branch ===
        lidar_feat = lidar.squeeze(1)
        lidar_proj = self.lidar_input_proj(lidar_feat)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)
        lidar_output = self.Qformer_lidar(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

        # === Pool and similarity ===
        rgb_avg = rgb_feats.mean(dim=1)
        lidar_avg = lidar_feats.mean(dim=1)
        print("torch.exp(self.log_temp)")
        print(torch.exp(self.log_temp))
        sim_matrix = (rgb_avg @ lidar_avg.T) * torch.exp(self.log_temp)
        targets = torch.arange(bs, device=image.device)

        # === Contrastive loss ===
        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) +
            F.cross_entropy(sim_matrix.T, targets)) / 2

        print(f"\n[DEBUG] CE(sim_matrix): {F.cross_entropy(sim_matrix, targets)}")
        print(f"\n[DEBUG] CE(sim_matrix.T): {F.cross_entropy(sim_matrix.T, targets)}")


        print("sim_matrix")
        # print(sim_matrix)
        # print(f"[DEBUG] sim_matrix sample row (0): {sim_matrix[5][:8]}")

        vals, idx = sim_matrix.max(dim=1)
        print("idx", idx)
        # print(f"[DEBUG] ffffffffffffffff sim_matrix sample row (0): {sim_matrix[5][:8]}")
        
        # === Diversity loss ===
        if bs > 1:
            loss_diversity = (self._diversity_loss(rgb_avg) + self._diversity_loss(lidar_avg)) / 2
        else:
            loss_diversity = torch.tensor(0.0, device=rgb_avg.device)
        print("loss_contrastive = ",loss_contrastive)
        print("loss_diversity = ",loss_diversity)
        # === Diagnostics ===
        prob_rgb2lidar = F.softmax(sim_matrix, dim=1)
        prob_lidar2rgb = F.softmax(sim_matrix.T, dim=1)
        acc_rgb2lidar = (prob_rgb2lidar.argmax(dim=1) == targets).float().mean()
        acc_lidar2rgb = (prob_lidar2rgb.argmax(dim=1) == targets).float().mean()
        rgb_feat_norm = torch.norm(rgb_feats, dim=-1).mean()
        lidar_feat_norm = torch.norm(lidar_feats, dim=-1).mean()
        diag_sim = sim_matrix.diag().mean()
        off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=sim_matrix.device)].mean()

        # Return dictionary
        return {
            "rgb_feats": rgb_feats,  # [B, N, D]
            "lidar_feats": lidar_feats,  # [B, N, D]
            "diagnostics": {
                "similarity": {
                    "matrix": sim_matrix,
                    "diag_mean": diag_sim,
                    "off_diag_mean": off_diag_sim
                },
                "accuracy": (acc_rgb2lidar + acc_lidar2rgb) / 2,
                "feature_norms": (rgb_feat_norm, lidar_feat_norm)
            }
        }


    # def forward_features_shuffled(self, samples, shuffle_prob=0.5, is_train=True):
    #     print("forward_features_shuffled")

    #     """
    #     Same as forward_features, but applies the same lidar shuffling scheme used in forward().
    #     Useful to (a) retrieve features, (b) compute recall/MMR, and (c) verify that the
    #     shuffling pipeline behaves as expected at eval time.

    #     Returns:
    #         dict with:
    #         - rgb_feats, lidar_feats: [B, N, D]
    #         - sim_matrix: [B, B] (after shuffling)
    #         - targets: the (possibly permuted) gt indices per row
    #         - keep_mask, perm: to inspect which samples were shuffled
    #         - loss_contrastive: scalar tensor
    #         - gt_indices, match_labels: for the ROC/AUC + shuffled-retrieval path
    #         - diagnostics: assorted stats
    #     """
    #     image = samples["image"]
    #     lidar = samples["lidar"]
    #     bs = image.size(0)

    #     print("\n================ forward_features_shuffled ================")
    #     print(f"[FF_SHUF] batch size = {bs}, shuffle_prob = {shuffle_prob}, is_train = {is_train}")
    #     print(f"[FF_SHUF] exp(log_temp) = {torch.exp(self.log_temp).item():.4f}")

    #     # === RGB branch ===
    #     rgb_proj = self.rgb_input_proj(image)
    #     rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
    #     rgb_embeds = self.rgb_layernorm(rgb_embeds)
    #     rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long, device=image.device)
    #     query_tokens = self.query_tokens.expand(bs, -1, -1)

    #     rgb_output = self.Qformer(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=rgb_embeds,
    #         encoder_attention_mask=rgb_atts,
    #         return_dict=True,
    #     )
    #     rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

    #     # === LiDAR branch ===
    #     lidar_feat = lidar.squeeze(1)
    #     lidar_proj = self.lidar_input_proj(lidar_feat)
    #     lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
    #     lidar_embeds = self.lidar_layernorm(lidar_embeds)
    #     lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long, device=lidar.device)
    #     query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

    #     lidar_output = self.Qformer_lidar(
    #         query_embeds=query_tokens_lidar,
    #         encoder_hidden_states=lidar_embeds,
    #         encoder_attention_mask=lidar_atts,
    #         return_dict=True,
    #     )
    #     lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

    #     # === Pool ===
    #     rgb_avg = rgb_feats.mean(dim=1)      # [B, D]
    #     lidar_avg = lidar_feats.mean(dim=1)  # [B, D]

    #     # === Shuffle just like in forward() ===
    #     device = image.device
    #     if is_train and shuffle_prob > 0:
    #         perm = torch.randperm(bs, device=device)
    #         keep_mask = torch.rand(bs, device=device) > shuffle_prob  # True = keep (no shuffle)

    #         lidar_avg_shuffled = lidar_avg.clone()
    #         lidar_avg_shuffled[~keep_mask] = lidar_avg[perm[~keep_mask]]

    #         targets = torch.arange(bs, device=device)
    #         targets_shuffled = targets.clone()
    #         targets_shuffled[~keep_mask] = perm[~keep_mask]

    #         num_mismatched = (~keep_mask).sum().item()

    #         # === Verbose prints ===
    #         print(f"[FF_SHUF] perm          : {perm.tolist()}")
    #         print(f"[FF_SHUF] keep_mask     : {keep_mask.tolist()}")
    #         print(f"[FF_SHUF] mismatched ct : {num_mismatched}/{bs}")
    #         if num_mismatched > 0:
    #             print("[FF_SHUF] (idx -> perm[idx]) for mismatched:")
    #             mismatched_idx = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    #             for i in mismatched_idx.tolist():
    #                 print(f"   {i} -> {perm[i].item()}")

    #         gt_indices = targets_shuffled  # what column is considered GT for each row
    #         match_labels = keep_mask.long()  # 1 if true match kept, 0 if mismatched
    #     else:
    #         perm = torch.arange(bs, device=device)
    #         keep_mask = torch.ones(bs, dtype=torch.bool, device=device)
    #         lidar_avg_shuffled = lidar_avg
    #         targets_shu_







    def forward_features_shuffled(self, samples, shuffle_prob=0.5, is_train=True):
        """
        Same as forward_features, but applies the same lidar shuffling scheme used in forward().
        Useful to (a) retrieve features, (b) compute recall/MMR, and (c) verify that the
        shuffling pipeline behaves as expected at eval time.

        Returns:
            dict with:
            - rgb_feats, lidar_feats: [B, N, D]
            - sim_matrix: [B, B] (after shuffling)
            - targets: the (possibly permuted) gt indices per row
            - keep_mask, perm: to inspect which samples were shuffled
            - loss_contrastive: scalar tensor
            - gt_indices, match_labels: for the ROC/AUC + shuffled-retrieval path
            - diagnostics: assorted stats
        """
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        shuffle_prob = 0
        print("\n================ forward_features_shuffled ================")
        print(f"[FF_SHUF] batch size = {bs}, shuffle_prob = {shuffle_prob}, is_train = {is_train}")
        print(f"[FF_SHUF] exp(log_temp) = {torch.exp(self.log_temp).item():.4f}")

        # === RGB branch ===
        rgb_proj = self.rgb_input_proj(image)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long, device=image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

        # === LiDAR branch ===
        lidar_feat = lidar.squeeze(1)
        lidar_proj = self.lidar_input_proj(lidar_feat)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long, device=lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

        lidar_output = self.Qformer_lidar(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

        # === Pool ===
        rgb_avg = rgb_feats.mean(dim=1)      # [B, D]
        lidar_avg = lidar_feats.mean(dim=1)  # [B, D]

        # === Shuffle just like in forward() ===
        device = image.device
        if is_train and shuffle_prob > 0:
            print("shuffle_prob =", shuffle_prob)
            perm = torch.randperm(bs, device=device)
            keep_mask = torch.rand(bs, device=device) > shuffle_prob  # True = keep (no shuffle)

            lidar_avg_shuffled = lidar_avg.clone()
            lidar_avg_shuffled[~keep_mask] = lidar_avg[perm[~keep_mask]]

            targets = torch.arange(bs, device=device)
            targets_shuffled = targets.clone()
            targets_shuffled[~keep_mask] = perm[~keep_mask]

            num_mismatched = (~keep_mask).sum().item()

            # === Verbose prints ===
            print(f"[FF_SHUF] perm          : {perm.tolist()}")
            print(f"[FF_SHUF] keep_mask     : {keep_mask.tolist()}")
            print(f"[FF_SHUF] mismatched ct : {num_mismatched}/{bs}")
            if num_mismatched > 0:
                print("[FF_SHUF] (idx -> perm[idx]) for mismatched:")
                mismatched_idx = torch.nonzero(~keep_mask, as_tuple=False).flatten()
                for i in mismatched_idx.tolist():
                    print(f"   {i} -> {perm[i].item()}")

            gt_indices = targets_shuffled  # what column is considered GT for each row
            match_labels = keep_mask.long()  # 1 if true match kept, 0 if mismatched
        else:
            print("shuffle_prob =", shuffle_prob)
            perm = torch.arange(bs, device=device)
            keep_mask = torch.ones(bs, dtype=torch.bool, device=device)
            lidar_avg_shuffled = lidar_avg
            targets_shuffled = torch.arange(bs, device=device)
            gt_indices = targets_shuffled
            match_labels = keep_mask.long()

            print("[FF_SHUF] (no shuffle applied)")

        # === Similarity + loss ===
        sim_matrix = (rgb_avg @ lidar_avg_shuffled.T) * torch.exp(self.log_temp)

        loss_i2t = F.cross_entropy(sim_matrix, targets_shuffled)
        loss_t2i = F.cross_entropy(sim_matrix.T, targets_shuffled)
        loss_contrastive = (loss_i2t + loss_t2i) / 2

        print(f"[FF_SHUF] loss_i2t = {loss_i2t.item():.4f}")
        print(f"[FF_SHUF] loss_t2i = {loss_t2i.item():.4f}")
        print(f"[FF_SHUF] contrastive_loss = {loss_contrastive.item():.4f}")
        print(f"[FF_SHUF] sim_matrix[0][:8] = {sim_matrix[0][:8]}")
        print("============================================================\n")

        # === Diagnostics (optional) ===
        prob_rgb2lidar = F.softmax(sim_matrix, dim=1)
        prob_lidar2rgb = F.softmax(sim_matrix.T, dim=1)
        acc_rgb2lidar = (prob_rgb2lidar.argmax(dim=1) == targets_shuffled).float().mean()
        acc_lidar2rgb = (prob_lidar2rgb.argmax(dim=1) == targets_shuffled).float().mean()
        diag_sim = sim_matrix.diag().mean()
        off_diag_sim = sim_matrix[~torch.eye(bs, dtype=torch.bool, device=sim_matrix.device)].mean()

        return {
            "rgb_feats": rgb_feats,                  # [B, N, D]
            "lidar_feats": lidar_feats,              # [B, N, D]
            "sim_matrix": sim_matrix,                # [B, B] (post-shuffle)
            "targets": targets_shuffled,             # [B]
            "perm": perm,                            # [B]
            "keep_mask": keep_mask,                  # [B] (True=kept, False=shuffled)
            "gt_indices": gt_indices,                # [B] (for your shuffled-eval path)
            "match_labels": match_labels,            # [B] (1 if kept, 0 if shuffled)
            "loss_contrastive": loss_contrastive,    # scalar
            "diagnostics": {
                "accuracy": (acc_rgb2lidar + acc_lidar2rgb) / 2,
                "diag_sim": diag_sim,
                "off_diag_sim": off_diag_sim,
            },
        }



























































































































































#--------------------------------------------------------------------------------------------------------------------
