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




from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Create one writer globally
run_name = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = f"lavis/output/BLIP2/qformer_muli_cam/{run_name}"
writer = SummaryWriter(log_dir=log_dir)

def log_alignment_stats(sim_rgb2lidar, sim_lidar2rgb, step=None, prefix="SIM"):
    stats = {
        f"{prefix}/rgb2lidar_min": sim_rgb2lidar.min().item(),
        f"{prefix}/rgb2lidar_max": sim_rgb2lidar.max().item(),
        f"{prefix}/rgb2lidar_mean": sim_rgb2lidar.mean().item(),
        f"{prefix}/lidar2rgb_min": sim_lidar2rgb.min().item(),
        f"{prefix}/lidar2rgb_max": sim_lidar2rgb.max().item(),
        f"{prefix}/lidar2rgb_mean": sim_lidar2rgb.mean().item(),
    }

    print("==== Alignment Stats ====")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
        writer.add_scalar(k, v, step)
    print("=========================")




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
        # self.rgb_input_proj = self.init_feature_projection(in_dim=2048, out_dim=hidden_dim) # resnet
        self.rgb_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim) # fpn
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




    def forward(self, samples, is_train=True):
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["bev_image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        # print("[MODEL DEBUG] Forward input keys:", list(samples.keys()))
        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f" - {k}: {tuple(v.shape)}")


        rgb_feat = image.squeeze(1)
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


        rgb_avg = rgb_feats.mean(dim=1)  # [B, D]
        lidar_avg = lidar_feats.mean(dim=1)  # [B, D]

        sim_matrix = torch.matmul(rgb_avg, lidar_avg.T)  # [B, B]
        temperature = 0.1
        sim_matrix = sim_matrix / temperature

        targets = torch.arange(bs).to(sim_matrix.device)

        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2

        return BlipOutput(loss=loss_contrastive)
        # =================================================================================================








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










    def forward(self, samples, is_train=True):
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["bev_image"]
        lidar = samples["lidar"]
        bs = image.size(0)
        # print("[MODEL DEBUG] Forward input keys:", list(samples.keys()))
        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f" - {k}: {tuple(v.shape)}")
    def forward(self, samples, is_train=True):

        image = samples["cam_f"]
        image2 = samples["cam_fr"]
        bs = image.size(0)

        rgb1_feat = image.squeeze(1)
        rgb1_proj = self.image_input_proj(rgb1_feat)
        rgb1_embeds = rgb1_proj.flatten(2).transpose(1, 2)
        rgb1_atts = torch.ones(rgb1_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb1_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb1_embeds,
            encoder_attention_mask=rgb1_atts,
            return_dict=True,
        )
        rgb1_feats = F.normalize(self.vision_proj(rgb1_output.last_hidden_state), dim=-1)

        rgb2_feats = image2.squeeze(1)
        rgb2_proj = self.image2_input_proj(rgb2_feats)
        rgb2_embeds = rgb2_proj.flatten(2).transpose(1, 2)
        rgb2_atts = torch.ones(rgb2_embeds.size()[:-1], dtype=torch.long).to(image2.device)
        query_tokens_2 = self.query_tokens_2.expand(bs, -1, -1)

        rgb2_output = self.Qformer_image2.bert(
            query_embeds=query_tokens_2,
            encoder_hidden_states=rgb2_embeds,
            encoder_attention_mask=rgb2_atts,
            return_dict=True,
        )
        rgb2_feats = F.normalize(self.rgb2_proj(rgb2_output.last_hidden_state), dim=-1)


        rgb1_avg = rgb1_feats.mean(dim=1)  # [B, D]
        rgb2_avg = rgb2_feats.mean(dim=1)  # [B, D]

        sim_matrix = torch.matmul(rgb1_avg, rgb2_avg.T)  # [B, B]
        temperature = 0.1
        sim_matrix = sim_matrix / temperature

        targets = torch.arange(bs).to(sim_matrix.device)

        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2

        return BlipOutput(loss=loss_contrastive)
        # =================================================================================================



