
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

        hidden_dim = 768
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




    def forward_features(self, samples, is_train=True):
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
    

        rgb_feat = image.squeeze(1)
        rgb_proj = self.rgb_input_proj(rgb_feat)
        # if is_train:
        #     rgb_proj = self.dropout(rgb_proj)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
       
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
        # if is_train:
        #     lidar_proj = self.dropout(lidar_proj)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        
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
        # sim_matrix = sim_matrix / temperature
        sim_matrix = (rgb_avg @ lidar_avg.T) * torch.exp(self.log_temp)  #----------------------------- (added temp)                     
        diag_sim = sim_matrix.diag()
        off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=sim_matrix.device)]

        targets = torch.arange(bs).to(sim_matrix.device)

        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2
      
        if rgb_avg.shape[0] > 1:  # Check batch size using either modality's features
            loss_diversity = (self._diversity_loss(rgb_avg) + 
                            self._diversity_loss(lidar_avg)) / 2
        else:
            loss_diversity = torch.tensor(0.0, device=rgb_avg.device)


        # 1. Softmax analysis
        prob_rgb2lidar = F.softmax(sim_matrix, dim=1)
        prob_lidar2rgb = F.softmax(sim_matrix.T, dim=1)
        
        # 2. Prediction accuracy
        targets = torch.arange(bs).to(image.device)
        acc_rgb2lidar = (prob_rgb2lidar.argmax(dim=1) == targets).float().mean()
        acc_lidar2rgb = (prob_lidar2rgb.argmax(dim=1) == targets).float().mean()
        
        # 3. Feature statistics
        rgb_feat_norm = torch.norm(rgb_feats, dim=-1).mean()
        lidar_feat_norm = torch.norm(lidar_feats, dim=-1).mean()
        
        # 4. Similarity range
        diag_sim = sim_matrix.diag().mean()
        off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool)].mean()
        # Combined loss

        total_loss = loss_contrastive + 0.3 * loss_diversity  # Weight hyperparameter
        # print('loss_contrastive', loss_contrastive)
        # print('loss_diversity', loss_diversity)
        # print('total_loss', total_loss)        
        return {
            "rgb_feats": rgb_feats,      # [B, N, D] (original)
            "lidar_feats": lidar_feats,   # [B, N, D] (original)
            "diagnostics": {              # New: Additional metrics
                "similarity": {
                    "matrix": sim_matrix,
                    "diag_mean": diag_sim,
                    "off_diag_mean": off_diag_sim
                },
                "accuracy": (acc_rgb2lidar + acc_lidar2rgb)/2,
                "feature_norms": (rgb_feat_norm, lidar_feat_norm)
            }}



    def forward(self, samples, is_train=True):
        shuffle_prob = 0.5
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
       
        rgb_feat = image.squeeze(1)
        rgb_proj = self.rgb_input_proj(rgb_feat)
        if is_train:
            rgb_proj = self.dropout(rgb_proj)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
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
        if is_train:
            lidar_proj = self.dropout(lidar_proj)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        
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



        if is_train and shuffle_prob > 0: 
            device = image.device
            perm = torch.randperm(bs, device=device)
            keep_mask = torch.rand(bs, device=device) > shuffle_prob

            print(f"[DEBUG] perm: {perm.tolist()}")
            print(f"[DEBUG] keep_mask: {keep_mask.tolist()}")
            
            lidar_avg_shuffled = lidar_avg.clone()
            lidar_avg_shuffled[~keep_mask] = lidar_avg[perm[~keep_mask]]

            targets = torch.arange(bs, device=device)
            targets[~keep_mask] = perm[~keep_mask]

            print(f"[DEBUG] targets: {targets.tolist()}")
            
            num_mismatched = (~keep_mask).sum().item()
            print(f"[DEBUG] Total mismatches introduced: {num_mismatched}")
        else:
            lidar_avg_shuffled = lidar_avg
            targets = torch.arange(bs, device=image.device)
        print(f"--------------------------------------------------------------------")

        # # === Compute contrastive loss ===
        sim_matrix = (rgb_avg @ lidar_avg_shuffled.T) * torch.exp(self.log_temp)

        diag_sim = sim_matrix.diag()
        true_pos_indices = keep_mask.nonzero(as_tuple=True)[0]
        mean_true_diag_sim = diag_sim[true_pos_indices].mean().item()
        print(f"[DEBUG] True match diag sim: {mean_true_diag_sim:.4f}")


        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2

        return BlipOutput(loss=loss_contrastive)



    def forward_features_with_shuffling(self, samples, shuffle_prob=0.5):
        image = samples["image"]
        lidar = samples["lidar"]
        B = image.size(0)

        def process(feat, proj, norm, query_tokens, qformer, proj_head):
            feat = feat.squeeze(1)
            feat_proj = proj(feat)
            feat_embeds = norm(feat_proj.flatten(2).transpose(1, 2))
            attn = torch.ones(feat_embeds.size()[:-1], dtype=torch.long).to(feat.device)
            output = qformer(query_embeds=query_tokens.expand(B, -1, -1),
                            encoder_hidden_states=feat_embeds,
                            encoder_attention_mask=attn,
                            return_dict=True)
            return F.normalize(proj_head(output.last_hidden_state), dim=-1).mean(dim=1)

        rgb_vecs = process(image, self.rgb_input_proj, self.rgb_layernorm,
                        self.query_tokens, self.Qformer.bert, self.vision_proj)

        lidar_vecs_raw = process(lidar, self.lidar_input_proj, self.lidar_layernorm,
                                self.query_tokens_lidar, self.Qformer_lidar.bert, self.lidar_proj)

        device = image.device
        perm = torch.randperm(B, device=device)
        keep_mask = torch.rand(B, device=device) > shuffle_prob

        shuffled_lidar_vecs = lidar_vecs_raw.clone()
        shuffled_lidar_vecs[~keep_mask] = lidar_vecs_raw[perm[~keep_mask]]

        gt_indices = torch.arange(B, device=device)
        gt_indices[~keep_mask] = perm[~keep_mask]

        labels = torch.ones(B, dtype=torch.long, device=device)
        labels[~keep_mask] = 0

        # === Add DEBUG logs ===
        print("\n[DEBUG] forward_features_with_shuffling:")
        print(f"  shuffle_prob = {shuffle_prob}")
        print(f"  keep_mask (1 = keep match): {keep_mask.tolist()}")
        print(f"  permuted indices: {perm.tolist()}")
        print(f"  gt_indices: {gt_indices.tolist()}")
        print(f"  labels: {labels.tolist()}")
        mismatches = (~keep_mask).nonzero(as_tuple=True)[0].tolist()
        print(f"  Total mismatches introduced: {len(mismatches)} at indices {mismatches}")

        # Optional: Check if shuffling actually changed feature vector
        if len(mismatches) > 0:
            idx = mismatches[0]
            original = lidar_vecs_raw[idx]
            shuffled = shuffled_lidar_vecs[idx]
            cosine_sim = F.cosine_similarity(original.unsqueeze(0), shuffled.unsqueeze(0)).item()
            print(f"  [DEBUG] Cosine sim before vs after shuffle at index {idx}: {cosine_sim:.4f}")

        sim_matrix = (rgb_vecs @ shuffled_lidar_vecs.T) * torch.exp(self.log_temp)

        return {
            "sim_matrix": sim_matrix,
            "gt_indices": gt_indices,
            "match_labels": labels,
            "rgb_vecs": rgb_vecs,
            "lidar_vecs": shuffled_lidar_vecs
        }





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



    def forward_4_chnages(self, samples, is_train=True):
        # print("start of Qformer ------------------------------------------------------------")
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)
    

        rgb_feat = image.squeeze(1)
        rgb_proj = self.rgb_input_proj(rgb_feat)
        # if is_train:
        #     rgb_proj = self.dropout(rgb_proj)
        rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
        rgb_embeds = self.rgb_layernorm(rgb_embeds)
        # print(f"Pre-LN mean/std: {rgb_embeds.mean().item():.2f} ± {rgb_embeds.std().item():.2f}")
        # print(f"Post-LN mean/std: {self.rgb_layernorm(rgb_embeds).mean().item():.2f} ± {self.rgb_layernorm(rgb_embeds).std().item():.2f}")


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
        # if is_train:
        #     lidar_proj = self.dropout(lidar_proj)
        lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
        lidar_embeds = self.lidar_layernorm(lidar_embeds)
        
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
        # sim_matrix = sim_matrix / temperature
        sim_matrix = (rgb_avg @ lidar_avg.T) * torch.exp(self.log_temp)  #----------------------------- (added temp)                     
        diag_sim = sim_matrix.diag()
        off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=sim_matrix.device)]

        targets = torch.arange(bs).to(sim_matrix.device)

        loss_contrastive = (
            F.cross_entropy(sim_matrix, targets) + 
            F.cross_entropy(sim_matrix.T, targets)) / 2
        # print(f"[DEBUG] Contrastive Loss: {loss_contrastive.item():.4f}")
        # if is_train:
            # log_alignment_stats(sim_matrix, loss_contrastive, step=samples.get("global_step", 0))

        
        if rgb_avg.shape[0] > 1:  # Check batch size using either modality's features
            loss_diversity = (self._diversity_loss(rgb_avg) + 
                            self._diversity_loss(lidar_avg)) / 2
        else:
            loss_diversity = torch.tensor(0.0, device=rgb_avg.device)



        # Combined loss

        total_loss = loss_contrastive + 0.3 * loss_diversity  # Weight hyperparameter
        print('loss_contrastive', loss_contrastive)
        print('loss_diversity', loss_diversity)
        print('total_loss', total_loss)        



        if is_train:
                # Collect parameter statistics
                param_stats = {}
                
                # 1. Temperature parameter
                param_stats["params/log_temp"] = self.log_temp.item()
                
                # 2. Projection layer weights
                for proj_name in ['rgb_input_proj', 'lidar_input_proj', 'vision_proj', 'lidar_proj']:
                    if hasattr(self, proj_name):
                        proj = getattr(self, proj_name)
                        for name, param in proj.named_parameters():
                            if param.requires_grad:
                                key = f"params/{proj_name}/{name.replace('.', '/')}"
                                param_stats[f"{key}_mean"] = param.data.mean().item()
                                param_stats[f"{key}_std"] = param.data.std().item()
                                if param.grad is not None:
                                    param_stats[f"{key}_grad"] = param.grad.abs().mean().item()
                
                # 3. Qformer main parameters (sample a few)
                qformer_params = list(self.Qformer.named_parameters())[:5]  # First 5 params
                for name, param in qformer_params:
                    if param.requires_grad:
                        key = f"params/Qformer/{name.replace('.', '/')}"
                        param_stats[f"{key}_mean"] = param.data.mean().item()
                        if param.grad is not None:
                            param_stats[f"{key}_grad"] = param.grad.abs().mean().item()

                # Enhanced logging with all metrics
                log_alignment_stats(
                    sim_matrix=sim_matrix,
                    contrastive_loss=loss_contrastive,
                    diversity_loss=loss_diversity,
                    total_loss=total_loss,
                    step=samples.get("global_step", 0), 
                    rgb_feats=rgb_avg,
                    lidar_feats=lidar_avg,
                    model=self,
                    param_stats=param_stats  # Pass the collected parameter stats
                )

        return BlipOutput(loss=total_loss)

























































































































































































#--------------------------------------------------------------------------------------------------------------------


# """
#  Copyright (c) 2023, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from transformers import BertTokenizer
# # from lavis.models.blip_outputs import BlipOutput
# # from lavis.models.base_model import BaseModel, concat_all_gather
# from lavis.models.Qformer import BertConfig, BertLMHeadModel
# from lavis.runners.log_utils import log_alignment_stats



# import logging
# import os

# import numpy as np
# import torch
# import torch.nn as nn
# from lavis.common.dist_utils import download_cached_file#, is_dist_avail_and_initialized
# from lavis.common.utils import get_abs_path, is_url
# from omegaconf import OmegaConf




# class BaseModel(nn.Module):
#     """Base class for models."""

#     def __init__(self):
#         super().__init__()

#     @property
#     def device(self):
#         return list(self.parameters())[0].device

#     def load_checkpoint(self, url_or_filename):
#         """
#         Load from a finetuned checkpoint.

#         This should expect no mismatch in the model keys and the checkpoint keys.
#         """

#         if is_url(url_or_filename):
#             cached_file = download_cached_file(
#                 url_or_filename, check_hash=False, progress=True
#             )
#             checkpoint = torch.load(cached_file, map_location="cpu")
#         elif os.path.isfile(url_or_filename):
#             checkpoint = torch.load(url_or_filename, map_location="cpu")
#         else:
#             raise RuntimeError("checkpoint url or path is invalid")

#         if "model" in checkpoint.keys():
#             state_dict = checkpoint["model"]
#         else:
#             state_dict = checkpoint

#         msg = self.load_state_dict(state_dict, strict=False)

#         logging.info("Missing keys {}".format(msg.missing_keys))
#         logging.info("load checkpoint from %s" % url_or_filename)

#         return msg


#     def load_checkpoint_from_config(self, cfg, **kwargs):
#         """
#         Load checkpoint as specified in the config file.

#         If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
#         When loading the pretrained model, each task-specific architecture may define their
#         own load_from_pretrained() method.
#         """
#         load_finetuned = cfg.get("load_finetuned", True)
#         if load_finetuned:
#             finetune_path = cfg.get("finetuned", None)
#             assert (
#                 finetune_path is not None
#             ), "Found load_finetuned is True, but finetune_path is None."
#             self.load_checkpoint(url_or_filename=finetune_path)
#         else:
#             load_pretrained = cfg.get("load_pretrained", True)
#             if load_pretrained:
#                 # load pre-trained weights
#                 pretrain_path = cfg.get("pretrained", None)
#                 assert "Found load_finetuned is False, but pretrain_path is None."
#                 self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)


#     def get_optimizer_params(self, weight_decay, lr_scale=1):
#         p_wd, p_non_wd = [], []
#         for n, p in self.named_parameters():
#             if not p.requires_grad:
#                 continue  # frozen weights
#             if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
#                 p_non_wd.append(p)
#             else:
#                 p_wd.append(p)        
#         optim_params = [
#             {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
#             {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
#         ]                
#         return optim_params
    
#     def before_evaluation(self, **kwargs):
#         pass

#     def show_n_params(self, return_str=True):
#         tot = 0
#         for p in self.parameters():
#             w = 1
#             for x in p.shape:
#                 w *= x
#             tot += w
#         if return_str:
#             if tot >= 1e6:
#                 return "{:.1f}M".format(tot / 1e6)
#             else:
#                 return "{:.1f}K".format(tot / 1e3)
#         else:
#             return tot


# class BaseEncoder(nn.Module):
#     """
#     Base class for primitive encoders, such as ViT, TimeSformer, etc.
#     """

#     def __init__(self):
#         super().__init__()

#     def forward_features(self, samples, **kwargs):
#         raise NotImplementedError

#     @property
#     def device(self):
#         return list(self.parameters())[0].device


# @torch.no_grad()
# def concat_all_gather(tensor): # ----------------------------------------------------------------------------

#     return tensor



# """
#  Copyright (c) 2022, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """

# from dataclasses import dataclass
# from typing import Optional

# import torch
# from transformers.modeling_outputs import (
#     ModelOutput,
#     BaseModelOutputWithPoolingAndCrossAttentions,
#     CausalLMOutputWithCrossAttentions,
# )


# @dataclass
# class BlipSimilarity(ModelOutput):
#     sim_i2t: torch.FloatTensor = None
#     sim_t2i: torch.FloatTensor = None

#     sim_i2t_m: Optional[torch.FloatTensor] = None
#     sim_t2i_m: Optional[torch.FloatTensor] = None

#     sim_i2t_targets: Optional[torch.FloatTensor] = None
#     sim_t2i_targets: Optional[torch.FloatTensor] = None


# @dataclass
# class BlipIntermediateOutput(ModelOutput):
#     """
#     Data class for intermediate outputs of BLIP models.

#     image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
#     text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

#     image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
#     text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

#     encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder.
#     encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs.

#     decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder.
#     decoder_labels (torch.LongTensor): labels for the captioning loss.

#     itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2).
#     itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,)

#     """

#     # uni-modal features
#     image_embeds: torch.FloatTensor = None
#     text_embeds: Optional[torch.FloatTensor] = None

#     image_embeds_m: Optional[torch.FloatTensor] = None
#     text_embeds_m: Optional[torch.FloatTensor] = None

#     # intermediate outputs of multimodal encoder
#     encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
#     encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

#     itm_logits: Optional[torch.FloatTensor] = None
#     itm_labels: Optional[torch.LongTensor] = None

#     # intermediate outputs of multimodal decoder
#     decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
#     decoder_labels: Optional[torch.LongTensor] = None


# @dataclass
# class BlipOutput(ModelOutput):
#     # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
#     sims: Optional[BlipSimilarity] = None

#     intermediate_output: BlipIntermediateOutput = None

#     loss: Optional[torch.FloatTensor] = None

#     loss_itc: Optional[torch.FloatTensor] = None

#     loss_itm: Optional[torch.FloatTensor] = None

#     loss_lm: Optional[torch.FloatTensor] = None


# @dataclass
# class BlipOutputWithLogits(BlipOutput):
#     logits: torch.FloatTensor = None
#     logits_m: torch.FloatTensor = None




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
#         use_grad_checkpoint=False,
#         vit_precision="fp16",
#         freeze_vit=True,
#         num_query_token=32,
#         cross_attention_freq=1,
#         embed_dim=256,
#         max_txt_len=32,
#     ):
#         super().__init__()

#         hidden_dim = 768
#         self.rgb_input_proj = self.init_feature_projection(in_dim=2048, out_dim=hidden_dim) # resnet
#         # self.rgb_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim) # fpn
#         self.lidar_input_proj = self.init_feature_projection(in_dim=256, out_dim=hidden_dim)
#         self.rgb_layernorm = nn.LayerNorm(hidden_dim)
#         self.lidar_layernorm = nn.LayerNorm(hidden_dim)

#         self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
#         qformer_lidar, query_tokens_lidar = self.init_Qformer(num_query_token, hidden_dim, cross_attention_freq)
#         self.Qformer_lidar = qformer_lidar
#         self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
#         self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)

#         self.vision_proj = nn.Linear(hidden_dim, embed_dim)
#         self.lidar_proj = nn.Linear(hidden_dim, embed_dim)

#         self.matching_head = nn.Sequential(
#             nn.Linear(embed_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#         self.temp = nn.Parameter(0.07 * torch.ones([]))
#         self.dropout = nn.Dropout(p=0.1)




#     def init_feature_projection(self, in_dim, out_dim):
#         # return nn.Sequential(
#         #     nn.Conv2d(in_dim, out_dim, kernel_size=1),
#         #     nn.BatchNorm2d(out_dim)
#         # )
#         return nn.Conv2d(in_dim, out_dim, kernel_size=1)




#     def forward(self, samples, is_train=True):
#         # print("start of Qformer ------------------------------------------------------------")
#         image = samples["image"]
#         lidar = samples["lidar"]
#         bs = image.size(0)
#         # print("[MODEL DEBUG] Forward input keys:", list(samples.keys()))
#         # for k, v in samples.items():
#         #     if isinstance(v, torch.Tensor):
#         #         print(f" - {k}: {tuple(v.shape)}")


#         rgb_feat = image.squeeze(1)
#         rgb_proj = self.rgb_input_proj(rgb_feat)
#         if is_train:
#             rgb_proj = self.dropout(rgb_proj)
#         rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)
#         rgb_embeds = self.rgb_layernorm(rgb_embeds)
#         rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
#         query_tokens = self.query_tokens.expand(bs, -1, -1)

#         rgb_output = self.Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=rgb_embeds,
#             encoder_attention_mask=rgb_atts,
#             return_dict=True,
#         )
#         rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)

#         lidar_feat = lidar.squeeze(1)
#         lidar_proj = self.lidar_input_proj(lidar_feat)
#         if is_train:
#             lidar_proj = self.dropout(lidar_proj)
#         lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
#         lidar_embeds = self.lidar_layernorm(lidar_embeds)
        
#         lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
#         query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

#         lidar_output = self.Qformer_lidar.bert(
#             query_embeds=query_tokens_lidar,
#             encoder_hidden_states=lidar_embeds,
#             encoder_attention_mask=lidar_atts,
#             return_dict=True,
#         )
#         lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)


#         rgb_avg = rgb_feats.mean(dim=1)  # [B, D]
#         lidar_avg = lidar_feats.mean(dim=1)  # [B, D]

#         sim_matrix = torch.matmul(rgb_avg, lidar_avg.T)  # [B, B]
#         temperature = 0.1
#         sim_matrix = sim_matrix / temperature

#         diag_sim = sim_matrix.diag()
#         off_diag_sim = sim_matrix[~torch.eye(bs, dtype=bool, device=sim_matrix.device)]

#         # print(f"[DEBUG] Diagonal similarities (pos pairs): {diag_sim.mean().item():.4f}")
#         # print(f"[DEBUG] Off-diagonal similarities (neg pairs): {off_diag_sim.mean().item():.4f}")
#         # prob = F.softmax(sim_matrix, dim=1)
#         # print(f"[DEBUG] Max softmax prob per sample (row): {prob.max(dim=1).values}")
#         # preds_rgb_to_lidar = sim_matrix.argmax(dim=1)
#         # acc_rgb_to_lidar = (preds_rgb_to_lidar == targets).float().mean().item()

#         # preds_lidar_to_rgb = sim_matrix.argmax(dim=0)
#         # acc_lidar_to_rgb = (preds_lidar_to_rgb == targets).float().mean().item()

#         # print(f"[DEBUG] RGB→LiDAR Acc: {acc_rgb_to_lidar:.2%}, LiDAR→RGB Acc: {acc_lidar_to_rgb:.2%}")



#         targets = torch.arange(bs).to(sim_matrix.device)

#         loss_contrastive = (
#             F.cross_entropy(sim_matrix, targets) + 
#             F.cross_entropy(sim_matrix.T, targets)) / 2
#         # print(f"[DEBUG] Contrastive Loss: {loss_contrastive.item():.4f}")
#         # if is_train:
#         #     log_alignment_stats(sim_matrix, loss_contrastive, step=samples.get("global_step", 0))

#         return BlipOutput(loss=loss_contrastive)
#         # =================================================================================================


#     @torch.no_grad()
#     def forward_features(self, samples):
#         image = samples["image"]
#         lidar = samples["lidar"]
#         bs = image.size(0)

#         # === Encode RGB ===
#         rgb_feat = image.squeeze(1)  # [B, C, H, W]
#         rgb_proj = self.rgb_input_proj(rgb_feat)  # [B, H, W, hidden_dim]
#         rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
#         rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
#         query_tokens = self.query_tokens.expand(bs, -1, -1)

#         rgb_output = self.Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=rgb_embeds,
#             encoder_attention_mask=rgb_atts,
#             return_dict=True,
#         )
#         rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)  # [B, N, D]

#         # === Encode LiDAR ===
#         lidar_feat = lidar.squeeze(1)
#         lidar_proj = self.lidar_input_proj(lidar_feat)
#         lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
#         lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
#         query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

#         lidar_output = self.Qformer_lidar.bert(
#             query_embeds=query_tokens_lidar,
#             encoder_hidden_states=lidar_embeds,
#             encoder_attention_mask=lidar_atts,
#             return_dict=True,
#         )
#         lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

#         return {
#             "rgb_feats": rgb_feats,         # [B, N, D]
#             "lidar_feats": lidar_feats      # [B, N, D]
#         }


#     # @torch.no_grad()
#     # def forward_features(self, samples):
#     #     image = samples["image"]
#     #     lidar = samples["lidar"]
#     #     bs = image.size(0)

#     #     # === Encode RGB ===
#     #     rgb_feat = image.squeeze(1)  # [B, C, H, W]
#     #     rgb_proj = self.rgb_input_proj(rgb_feat)  # [B, H, W, hidden_dim]
#     #     rgb_embeds = rgb_proj.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
#     #     rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
#     #     query_tokens = self.query_tokens.expand(bs, -1, -1)

#     #     rgb_output = self.Qformer.bert(
#     #         query_embeds=query_tokens,
#     #         encoder_hidden_states=rgb_embeds,
#     #         encoder_attention_mask=rgb_atts,
#     #         return_dict=True,
#     #     )
#     #     rgb_feats = F.normalize(self.vision_proj(rgb_output.last_hidden_state), dim=-1)  # [B, N, D]

#     #     # === Encode LiDAR ===
#     #     lidar_feat = lidar.squeeze(1)
#     #     lidar_proj = self.lidar_input_proj(lidar_feat)
#     #     lidar_embeds = lidar_proj.flatten(2).transpose(1, 2)
#     #     lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
#     #     query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

#     #     lidar_output = self.Qformer_lidar.bert(
#     #         query_embeds=query_tokens_lidar,
#     #         encoder_hidden_states=lidar_embeds,
#     #         encoder_attention_mask=lidar_atts,
#     #         return_dict=True,
#     #     )
#     #     lidar_feats = F.normalize(self.lidar_proj(lidar_output.last_hidden_state), dim=-1)

#     #     return {
#     #         "rgb_feats": rgb_feats,         # [B, N, D]
#     #         "lidar_feats": lidar_feats      # [B, N, D]
#     #     }




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







