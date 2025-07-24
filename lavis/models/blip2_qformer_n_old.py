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
    # """
    # Performs all_gather operation on the provided tensors.
    # *** Warning ***: torch.distributed.all_gather has no gradient.
    # """
    # # if use distributed training
    # # if not is_dist_avail_and_initialized():
    # #     return tensor

    # tensors_gather = [
    #     torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    # ]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # output = torch.cat(tensors_gather, dim=0)
    # return output
    return tensor


# def tile(x, dim, n_tile):
#     init_dim = x.size(dim)
#     repeat_idx = [1] * x.dim()
#     repeat_idx[dim] = n_tile
#     x = x.repeat(*(repeat_idx))
#     order_index = torch.LongTensor(
#         np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
#     )
#     return torch.index_select(x, dim, order_index.to(x.device))













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


# @dataclass
# class BlipOutputFeatures(ModelOutput):
#     """
#     Data class of features from BlipFeatureExtractor.

#     Args:
#         image_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional
#         image_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional
#         text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional
#         text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional

#         The first embedding or feature is for the [CLS] token.

#         Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
#     """

#     image_embeds: Optional[torch.FloatTensor] = None
#     image_embeds_proj: Optional[torch.FloatTensor] = None

#     text_embeds: Optional[torch.FloatTensor] = None
#     text_embeds_proj: Optional[torch.FloatTensor] = None

#     multimodal_embeds: Optional[torch.FloatTensor] = None






















class Blip2Base(BaseModel):
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        # print("[TRACE] init_Qformer in blip2.py")
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = 4 #6 
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
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
        max_txt_len=32,
    ):
        super().__init__()

        hidden_dim = 768
        self.rgb_input_proj = self.init_feature_projection(in_dim=2048, out_dim=hidden_dim)
        self.lidar_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim)

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
        qformer_lidar, query_tokens_lidar = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
        self.Qformer_lidar = qformer_lidar
        self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
        self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)

        self.vision_proj = nn.Linear(hidden_dim, embed_dim)
        self.lidar_proj = nn.Linear(hidden_dim, embed_dim)

        self.matching_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def init_feature_projection(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, samples):
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        print('batch size in blip2_qformer', bs)
        print('image shape', image.shape)
        print('lidar shape', lidar.shape)
        # print("[MODEL DEBUG] Forward input keys:", list(samples.keys()))
        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f" - {k}: {tuple(v.shape)}")

        print('-------------------------------------')

        rgb_feat = image.squeeze(1)
        print('image shape after', image.shape)

        rgb_proj = self.rgb_input_proj(rgb_feat)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

        lidar_feat = lidar.squeeze(1)
        lidar_proj = self.lidar_input_proj(lidar_feat)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

        lidar_output = self.Qformer_lidar.bert(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

        # rgb_feats_all = concat_all_gather(rgb_feats)
        # lidar_feats_all = concat_all_gather(lidar_feats)

        B, N, D = rgb_feats.shape

        dot_products = torch.einsum('bnd,tmd->btmn', rgb_feats, lidar_feats)
        sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)
        sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)
        targets = torch.arange(B).to(sim_rgb2lidar.device)

        loss_contrastive = (
            F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
        ) / 2

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
















































































































# """
#  Copyright (c) 2023, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """
# import logging

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from torch.cuda.amp import autocast as autocast
# from torch.nn import functional as F
# from lavis.models.blip_outputs import BlipOutput, BlipOutputFeatures
# from lavis.models.base_model import concat_all_gather #, all_gather_with_grad

# from lavis.models.blip2_models.blip2 import (
#     Blip2Base,
#     # compute_sim_matrix,
#     # disabled_train,
# )




# class Blip2Qformer(Blip2Base):
#     """
#     BLIP2 Q-former variant for precomputed features (e.g., NuScenes).
#     No vision encoder is used; instead, 1x1 conv projections map feature maps to Q-former input size.
#     """

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
#         use_grad_checkpoint=False,
#         vit_precision="fp16",
#         freeze_vit=True,
#         num_query_token=32,
#         cross_attention_freq=2,
#         embed_dim=256,
#         max_txt_len=32,
#     ):
#         super().__init__()

#         # ===== Projection for precomputed features =====
#         hidden_dim = 768  # Must match Qformer.config.hidden_size
#         self.rgb_input_proj = self.init_feature_projection(in_dim=2048, out_dim=hidden_dim)
#         self.lidar_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim)

#         # ===== Q-Formers =====
#         self.Qformer, self.query_tokens = self.init_Qformer(
#             num_query_token, hidden_dim, cross_attention_freq
#         )
#         qformer_lidar, query_tokens_lidar = self.init_Qformer(
#             num_query_token, hidden_dim, cross_attention_freq
#         )
#         self.Qformer_lidar = qformer_lidar
#         self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
#         self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)

#         # ===== Projection for contrastive loss =====
#         self.vision_proj = nn.Linear(hidden_dim, embed_dim)
#         self.lidar_proj = nn.Linear(hidden_dim, embed_dim)

#         # ===== Matching head =====
#         self.matching_head = nn.Sequential(
#             nn.Linear(embed_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#         self.temp = nn.Parameter(0.07 * torch.ones([]))

#     def init_feature_projection(self, in_dim, out_dim):
#         return nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, kernel_size=1),
#             nn.BatchNorm2d(out_dim)  # Better for spatial feature maps
#         )


#     def forward(self, samples):
#         image = samples["image"]  # [B, 1, C, H, W]
#         lidar = samples["lidar"]  # [B, 1, C, H, W]
#         bs = image.size(0)

#         # === Camera features ===
#         rgb_feat = image.squeeze(1)  # [B, C, H, W]
#         rgb_proj = self.rgb_input_proj(rgb_feat)  # [B, D, H, W]
#         rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)  # [B, N, D]
#         rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
#         query_tokens = self.query_tokens.expand(bs, -1, -1)

#         rgb_output = self.Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=rgb_embeds,
#             encoder_attention_mask=rgb_atts,
#             return_dict=True,
#         )
#         rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

#         # === LiDAR features ===
#         lidar_feat = lidar.squeeze(1)  # [B, C, H, W]
#         lidar_proj = self.lidar_input_proj(lidar_feat)  # [B, D, H, W]
#         lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)  # [B, N, D]
#         lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
#         query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

#         lidar_output = self.Qformer_lidar.bert(
#             query_embeds=query_tokens_lidar,
#             encoder_hidden_states=lidar_embeds,
#             encoder_attention_mask=lidar_atts,
#             return_dict=True,
#         )
#         lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

#         # === Gather across GPUs (if using DDP) ===
#         rgb_feats_all = concat_all_gather(rgb_feats)
#         lidar_feats_all = concat_all_gather(lidar_feats)
#         B, N, D = rgb_feats_all.shape

#         # === Contrastive Loss ===
#         dot_products = torch.einsum('bnd,tmd->btmn', rgb_feats_all, lidar_feats_all)
#         sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)
#         sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)
#         targets = torch.arange(B).to(sim_rgb2lidar.device)

#         loss_contrastive = (
#             F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
#             F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
#         ) / 2

#         return BlipOutput(loss=loss_contrastive)


#     @classmethod
#     def from_config(cls, cfg): # ----------------------------------------------------------------------------
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

#     # def compute_sim_matrix(self, data_loader, task_cfg):
#     #     """
#     #     Compute similarity i2t, t2i matrix for the given data loader.
#     #     """
#     #     k_test = task_cfg.k_test

#     #     return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)








































# blip2.py code 

