"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

# from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.lidar_encoders.pointnet_encoder import PointNetEncoder


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
        "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
    }

    # def __init__(
    #     self,
    #     vit_model="eva_clip_g",
    #     img_size=224,
    #     drop_path_rate=0,  # not used
    #     use_grad_checkpoint=False,  # not used
    #     vit_precision="fp16",  # not used
    #     freeze_vit=True,
    #     num_query_token=32,
    #     cross_attention_freq=2,
    #     embed_dim=256,
    #     max_txt_len=32,  # not used
    # ):
    #     super().__init__()

    #     # === RGB Encoder ===
    #     self.visual_encoder, self.ln_vision = self.init_vision_encoder(
    #         vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
    #     )
    #     if freeze_vit:
    #         for name, param in self.visual_encoder.named_parameters():
    #             param.requires_grad = False
    #         self.visual_encoder = self.visual_encoder.eval()
    #         self.visual_encoder.train = disabled_train
    #         logging.info("freeze RGB vision encoder")

    #     # === LiDAR Encoder (identical structure) ===
    #     self.lidar_encoder, self.ln_lidar = self.init_vision_encoder(
    #         vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
    #     )

    #     if freeze_vit:
    #         for name, param in self.lidar_encoder.named_parameters():
    #             param.requires_grad = False
    #         self.lidar_encoder = self.lidar_encoder.eval()
    #         self.lidar_encoder.train = disabled_train
    #         logging.info("freeze LiDAR encoder")

    #     # === Shared Q-Former for RGB and LiDAR ===
    #     self.Qformer, self.query_tokens = self.init_Qformer(
    #         num_query_token, self.visual_encoder.num_features, cross_attention_freq
    #     )
    #     # self.Qformer_lidar, self.query_tokens_lidar = self.init_Qformer(
    #     #     num_query_token, self.lidar_encoder.num_features, cross_attention_freq
    #     # )
    #     qformer_lidar, query_tokens_lidar = self.init_Qformer(
    #         num_query_token, self.lidar_encoder.num_features, cross_attention_freq)
    #     self.Qformer_lidar = qformer_lidar
    #     self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
    #     self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)


    #     # === Projection layers for contrastive loss ===
    #     self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
    #     self.lidar_proj = nn.Linear(self.Qformer_lidar.config.hidden_size, embed_dim)

    #     # === Matching head (concatenated RGB + LiDAR → binary match) ===
    #     self.matching_head = nn.Sequential(
    #         nn.Linear(embed_dim * 2, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 1)
    #     )

    #     # === Learnable temperature for contrastive loss ===
    #     self.temp = nn.Parameter(0.07 * torch.ones([]))

    # def forward(self, samples):
    #     image = samples["image"]
    #     lidar = samples["lidar"]
        
    #     bs = image.size(0)

    #     if dist.is_available() and dist.is_initialized():
    #         rank = dist.get_rank()
    #     else:
    #         rank = 0  # or None if rank isn’t needed for your logic

    #     # rank = dist.get_rank()

    #     # === Encode Camera (RGB) ===
    #     rgb_embeds = self.ln_vision(self.visual_encoder(image))
    #     rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
    #     query_tokens = self.query_tokens.expand(bs, -1, -1)

    #     rgb_query_output = self.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=rgb_embeds,
    #         encoder_attention_mask=rgb_atts,
    #         return_dict=True,
    #     )
    #     rgb_feats = F.normalize(self.vision_proj(rgb_query_output.last_hidden_state), dim=-1)

    #     # === Encode LiDAR ===
    #     lidar_embeds = self.ln_lidar(self.lidar_encoder(lidar))
    #     lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
    #     query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)
        
    #     lidar_query_output = self.Qformer_lidar.bert(
    #         query_embeds=query_tokens_lidar,
    #         encoder_hidden_states=lidar_embeds,
    #         encoder_attention_mask=lidar_atts,
    #         return_dict=True,
    #     )
    #     lidar_feats = F.normalize(self.lidar_proj(lidar_query_output.last_hidden_state), dim=-1)
    #     rgb_feats_all = concat_all_gather(rgb_feats)
    #     lidar_feats_all = concat_all_gather(lidar_feats)

    #     B, N, D = rgb_feats_all.shape  # [B, N, D]

    #     # Flatten for einsum
    #     rgb_feats_flat = rgb_feats_all  # [B, N, D]
    #     lidar_feats_flat = lidar_feats_all  # [B, N, D]


    #     # Output shape: [B, B, N, N]
    #     dot_products = torch.einsum('bnd,tmd->btmn', rgb_feats_flat, lidar_feats_flat)
    #     # RGB→LiDAR
    #     sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)  # shape [B, B]

    #     # LiDAR→RGB
    #     sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)  # shape [B, B]
    #     targets = torch.arange(B).to(sim_rgb2lidar.device)  # [0, 1, ..., B-1]

    #     loss_contrastive = (
    #         F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
    #         F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
    #     ) / 2
    #     return BlipOutput(
    #         loss=loss_contrastive,
            
    #     )


    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,  # not used
        use_grad_checkpoint=False,  # not used
        vit_precision="fp16",  # not used
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,  # not used
    ):
        super().__init__()

        # === RGB Encoder ===
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze RGB vision encoder")

        # === LiDAR Encoder (PointNet instead of ViT) ===
        self.lidar_encoder, self.ln_lidar = self.init_lidar_encoder(input_dim=5, output_dim=embed_dim)


        if freeze_vit:
            for name, param in self.lidar_encoder.named_parameters():
                param.requires_grad = False
            self.lidar_encoder = self.lidar_encoder.eval()
            self.lidar_encoder.train = disabled_train
            logging.info("freeze LiDAR PointNet encoder")

        # === Q-Former blocks ===
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer_lidar, query_tokens_lidar = self.init_Qformer(
            num_query_token, embed_dim, cross_attention_freq
        )
        self.query_tokens_lidar = nn.Parameter(query_tokens_lidar.data.clone())
        self.register_parameter("query_tokens_lidar", self.query_tokens_lidar)

        # === Projection heads ===
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.lidar_proj = nn.Linear(self.Qformer_lidar.config.hidden_size, embed_dim)

        # === Matching head ===
        self.matching_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # === Learnable temperature ===
        self.temp = nn.Parameter(0.07 * torch.ones([]))


        
    def forward(self, samples):
        image = samples["image"]
        lidar = samples["lidar"]  # [B, N, 5]
        print(f"[DEBUG] LiDAR shape: {lidar.shape}") 
        bs = image.size(0)

        # === Encode Camera (RGB) ===
        rgb_embeds = self.ln_vision(self.visual_encoder(image))
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_query_output.last_hidden_state), dim=-1)

        # === Encode LiDAR using frozen PointNet ===
        lidar_feats_raw = self.lidar_encoder(lidar)         # [B, D]
        lidar_feats_raw = lidar_feats_raw.unsqueeze(1)      # [B, 1, D]
        lidar_embeds = self.ln_lidar(lidar_feats_raw)       # [B, 1, D]
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)

        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)
        lidar_query_output = self.Qformer_lidar.bert(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_query_output.last_hidden_state), dim=-1)

        # === Contrastive loss ===
        rgb_feats_all = concat_all_gather(rgb_feats)
        lidar_feats_all = concat_all_gather(lidar_feats)
        
        print("rgb_feats_all", rgb_feats_all)
        print("lidar_feats_all", lidar_feats_all)

        B, N, D = rgb_feats_all.shape
        dot_products = torch.einsum('bnd,tmd->btmn', rgb_feats_all, lidar_feats_all)
        sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)
        sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)

        targets = torch.arange(B).to(sim_rgb2lidar.device)
        loss_contrastive = (
            F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
        ) / 2

        print("sim_rgb2lidar", sim_rgb2lidar)

        print("sim_lidar2rgb", sim_lidar2rgb)

        print("loss_contrastive", loss_contrastive)
        return BlipOutput(loss=loss_contrastive)


    def forward_(self, samples):
        # print("Samples keys:", samples.keys())
        image = samples["image"]
        lidar = samples["lidar"]
        # print("Image shape:", image.shape)
        # print("LiDAR shape:", lidar.shape)
        
        bs = image.size(0)
        # print("bs = image.size(0)", bs)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0  # or None if rank isn’t needed for your logic

        # rank = dist.get_rank()

        # === Encode Camera (RGB) ===
        rgb_embeds = self.ln_vision(self.visual_encoder(image))
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        # print("shape of query_tokens", query_tokens.shape)

        rgb_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_query_output.last_hidden_state), dim=-1)

        # === Encode LiDAR ===
        lidar_embeds = self.ln_lidar(self.lidar_encoder(lidar))
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)
        # query_tokens_lidar = self.query_tokens_lidar.repeat(bs, -1, -1)
        # print("shape of query_tokens_lidar", query_tokens_lidar.shape)
        
        lidar_query_output = self.Qformer_lidar.bert(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_query_output.last_hidden_state), dim=-1)
        # print("shape of rgb_feats", rgb_feats.shape)
        # print("shape of lidar_feats", lidar_feats.shape)
        # === Contrastive Loss ===
        rgb_feats_all = concat_all_gather(rgb_feats)
        lidar_feats_all = concat_all_gather(lidar_feats)

        # print("shape of rgb_feats_all", rgb_feats_all.shape)
        # print("shape of lidar_feats_all", lidar_feats_all.shape)


        B, N, D = rgb_feats_all.shape  # [B, N, D]

        # Flatten for einsum
        rgb_feats_flat = rgb_feats_all  # [B, N, D]
        lidar_feats_flat = lidar_feats_all  # [B, N, D]


        # Output shape: [B, B, N, N]
        dot_products = torch.einsum('bnd,tmd->btmn', rgb_feats_flat, lidar_feats_flat)
        # RGB→LiDAR
        sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)  # shape [B, B]

        # LiDAR→RGB
        sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)  # shape [B, B]



        targets = torch.arange(B).to(sim_rgb2lidar.device)  # [0, 1, ..., B-1]


        # print("targets = ", targets)
        # print("sim_rgb2lidar = ", sim_rgb2lidar.shape)
        # print("sim_rgb2lidar value = ", sim_rgb2lidar)

        # print("sim_lidar2rgb = ", sim_lidar2rgb.shape)
        # print("sim_lidar2rgb value = ", sim_lidar2rgb)

        loss_contrastive = (
            F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
        ) / 2


        # B_all, N, D = rgb_feats_all.shape

        # # Reshape for batched dot products
        # rgb_feats_flat   = rgb_feats_all.view(B_all, N, 1, D)       # [B, N, 1, D]
        # lidar_feats_flat = lidar_feats_all.view(1, B_all, N, D)     # [1, B, N, D]

        # # Compute pairwise dot products between queries
        # # Result: [B, B, N, N] → sim(i,j)[k,l] = q_k ⋅ l_l
        # dot_products = torch.einsum('bnik,btjk->bnti', rgb_feats_flat, lidar_feats_flat)  # or use broadcasting

        # # RGB→LiDAR: max over lidar queries for each RGB query
        # sim_rgb2lidar = dot_products.max(dim=-1).values.mean(dim=-1)  # [B, B]

        # # LiDAR→RGB: max over RGB queries for each LiDAR query
        # sim_lidar2rgb = dot_products.max(dim=-2).values.mean(dim=-1)  # [B, B]
        
        # sim_score = sim_rgb2lidar.mean(dim=1)  # shape = [B]

        # B = sim_rgb2lidar.size(0)  # get batch size
        # targets = torch.arange(B).to(sim_rgb2lidar.device)  # [0, 1, 2, ..., B-1]

        # print("targets = ", targets)
        # print("sim_rgb2lidar = ", sim_rgb2lidar.shape)
        # print("sim_rgb2lidar value = ", sim_rgb2lidar)

        # print("sim_lidar2rgb = ", sim_lidar2rgb.shape)
        # print("sim_lidar2rgb value = ", sim_lidar2rgb)

        # loss_contrastive = (
        #     F.cross_entropy(sim_rgb2lidar, targets, label_smoothing=0.1) +
        #     F.cross_entropy(sim_lidar2rgb, targets, label_smoothing=0.1)
        # ) / 2




        # sim_r2l = torch.matmul(rgb_feats.mean(dim=1), lidar_feats_all.mean(dim=1).T) / self.temp
        # sim_l2r = torch.matmul(lidar_feats.mean(dim=1), rgb_feats_all.mean(dim=1).T) / self.temp
        # print("passed the similarity check -----------------------------------------------------------------------")
        # targets = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long).to(image.device)

        # loss_contrastive = (
        #     F.cross_entropy(sim_r2l, targets, label_smoothing=0.1) +
        #     F.cross_entropy(sim_l2r, targets, label_smoothing=0.1)
        # ) / 2
        # print("passed the loss_contrastive -----------------------------------------------------------------------")
        
        
        
        # === Matching Loss (Like ITM in BLIP2) ===

        # # Sample mismatched lidar (negatives)
        # lidar_feats_neg = []
        # for i in range(bs):
        #     j = (i + 1) % bs  # Simple shift
        #     lidar_feats_neg.append(lidar_feats[j])
        # lidar_feats_neg = torch.stack(lidar_feats_neg, dim=0)
        # print("passed the lidar_feats_neg -----------------------------------------------------------------------")
        # # Build fused features: [pos, neg]
        # rgb_all = torch.cat([rgb_feats, rgb_feats], dim=0)         # [2*bs, Nq, D]
        # print("passed the rgb_all -----------------------------------------------------------------------")

        # lidar_all = torch.cat([lidar_feats, lidar_feats_neg], dim=0)  # [2*bs, Nq, D]
        # print("passed the lidar_all -----------------------------------------------------------------------")

        # fused_feats = torch.cat([rgb_all.mean(dim=1), lidar_all.mean(dim=1)], dim=-1)
        # print("passed the fused_feats -----------------------------------------------------------------------")

        # logits_match = self.matching_head(fused_feats).squeeze()
        # print("passed the logits_match -----------------------------------------------------------------------")
        # labels_match = torch.cat([
        #     torch.ones(bs),    # positive pairs
        #     torch.zeros(bs),   # negative pairs
        # ]).to(image.device)
        # print("passed the labels_match -----------------------------------------------------------------------")
        # loss_match = F.binary_cross_entropy_with_logits(logits_match, labels_match)
        # print("passed the loss_match -----------------------------------------------------------------------")





        # print("passed the total_loss -----------------------------------------------------------------------")
        return BlipOutput(
            loss=loss_contrastive,
            # loss_itc=loss_contrastive,
        )


    @torch.no_grad()
    def forward_features(self, samples):
        image = samples["image"]
        lidar = samples["lidar"]
        bs = image.size(0)

        # === Encode RGB ===
        rgb_embeds = self.ln_vision(self.visual_encoder(image))
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_query_output.last_hidden_state), dim=-1)

        # === Encode LiDAR ===
        lidar_embeds = self.ln_lidar(self.lidar_encoder(lidar))
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        query_tokens_lidar = self.query_tokens_lidar.expand(bs, -1, -1)

        lidar_query_output = self.Qformer_lidar.bert(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_query_output.last_hidden_state), dim=-1)
        print("----------------   lidar_feats shape", lidar_feats.shape)
        return {
            "rgb_feats": rgb_feats,         # [B, N, D]
            "lidar_feats": lidar_feats      # [B, N, D]
        }

    @classmethod
    def from_config(cls, cfg): # ----------------------------------------------------------------------------
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

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

