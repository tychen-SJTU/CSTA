# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import os
import numpy as np
from torch.nn.functional import kl_div

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from timesformer.models.cosine_linear import CosineLinear, SplitCosineLinear
from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class DotProductAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(DotProductAttention, self).__init__()
        self.input_dim = input_dim

    def forward(self, query, keys, values):
        # 计算分数向量
        scores = torch.matmul(query, keys.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.input_dim, dtype=torch.float32))

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 对值向量进行加权求和
        output = torch.matmul(attention_weights, values)

        return output


class Classifier(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(embed_dim, output_dim)
        for p in self.head.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        self.head = self.head.cuda()
        x = self.head(x)
        return x


class T_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

        for p in self.D_fc1.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.D_fc2.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def freeze_parameters(m, requires_grad=False):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        return self.freeze_parameters(self)

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # 在时间维度上，将输入张量限制在 [-1, 1] 的范围内
        # grid = torch.stack(torch.meshgrid(
        #     torch.linspace(-1, 1, x.size(2)),
        #     torch.linspace(-1, 1, 1),
        # )).unsqueeze(0).repeat(x.size(0), 1, 1, 1).transpose(1, -1).cuda()
        # xs = F.grid_sample(x.unsqueeze(2), grid, mode='bilinear', padding_mode='border').squeeze(2)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x


class SAdapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=1.0, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        for p in self.qkv.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Attention_via_tasks(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.norm = nn.LayerNorm(dim)
        if self.with_qkv:
            self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        for p in self.q.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.kv.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def freeze_parameters(m, requires_grad=False):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        return self.freeze_parameters(self)

    def forward(self, q, kv):

        B, N, C = q.shape
        _, M, _ = kv.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        x = self.norm(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time', age=0,
                 cfg=None):
        super().__init__()
        self.spatial_euclidean_dist = None
        self.temporal_euclidean_dist = None
        self.kv_temporal = None
        self.kv_spatial = None
        self.pearson_corr = None
        self.div_spatial = None
        self.div_temporal = None
        self.attention_type = attention_type
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])
        self.cfg = cfg
        self.norm1 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.age = age
        self.Spatial_adapter = {}
        self.Temporal_adapter = {}

        if self.cfg.ADAPTER:
            if self.age > 0:
                for j in range(0, self.age):
                    self.Spatial_adapter[j] = T_Adapter(dim)
                    self.Temporal_adapter[j] = T_Adapter(dim)
                    self.Spatial_attn_via_tasks = Attention_via_tasks(dim)
                    self.Temporal_attn_via_tasks = Attention_via_tasks(dim)
                    self.Spatial_attn_via_tasks_old = Attention_via_tasks(dim)
                    self.Temporal_attn_via_tasks_old = Attention_via_tasks(dim)
                    # load trained in last tasks
                    if j < self.age - 1:
                        self.Spatial_adapter[j].load_state_dict(torch.load(os.path.join(cfg.DIR,
                            'model_Spatial_adapter{}.pt'.format(j))))
                        self.Temporal_adapter[j].load_state_dict(torch.load(os.path.join(cfg.DIR,
                            'model_Temporal_adapter{}.pt'.format(j))))

                        self.Spatial_attn_via_tasks_old.load_state_dict(torch.load(os.path.join(cfg.DIR,
                            'model_Spatial_attn_via_tasks{}.pt'.format(self.age - 2))))
                        self.Temporal_attn_via_tasks_old.load_state_dict(torch.load(os.path.join(cfg.DIR,
                            'model_Temporal_attn_via_tasks{}.pt'.format(self.age - 2))))
                        # freeze trained in last tasks

                    self.Spatial_adapter[j] = self.Spatial_adapter[j].cuda()
                    self.Temporal_adapter[j] = self.Temporal_adapter[j].cuda()
                    self.Spatial_attn_via_tasks = self.Spatial_attn_via_tasks.cuda()
                    self.Temporal_attn_via_tasks = self.Temporal_attn_via_tasks.cuda()
                    self.Spatial_attn_via_tasks_old = self.Spatial_attn_via_tasks_old.cuda()
                    self.Temporal_attn_via_tasks_old = self.Temporal_attn_via_tasks_old.cuda()

    def freeze_parameters(m, requires_grad=False):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        return self.freeze_parameters(self)

    def forward(self, x, B, T, W, signal_flag):

        global res_temporal_out, res_spatial_out
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:, 1:, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            if self.cfg.ADAPTER:
                # task 1-9 train/val时
                if signal_flag == 0 or signal_flag == 2:
                    if self.age > 0:
                        temporal_res_spatial_ada = []
                        for i in range(0, self.age):
                            # calculate att for every task, every task is ada from all tasks before
                            temporal_res_spatial_ada.append(self.Temporal_adapter[i](res_temporal))
                            # concat current task and task before's information and cal att_via_task
                            # b = torch.cat(temporal_res_spatial_ada, dim=1).squeeze(0)
                        kv = torch.cat((res_temporal, torch.cat(temporal_res_spatial_ada, dim=1).squeeze(0)), dim=1)
                        self.kv_temporal = self.Temporal_attn_via_tasks(
                            temporal_res_spatial_ada[self.age - 1], kv)
                        # add new task's information
                        res_temporal = self.kv_temporal * self.cfg.PATH_WEIGHT + res_temporal

                else:
                    if self.age > 1:
                        temporal_res_spatial_ada = []
                        for i in range(0, self.age - 1):
                            # calculate att for every task, every task is ada from all tasks before
                            temporal_res_spatial_ada.append(self.Temporal_adapter[i](res_temporal))
                            # concat current task and task before's information and cal att_via_task

                        kv = torch.cat((res_temporal, torch.cat(temporal_res_spatial_ada, dim=1).squeeze(0)), dim=1)
                        self.kv_temporal = self.Temporal_attn_via_tasks_old(
                            temporal_res_spatial_ada[self.age - 2], kv)
                        # add new task's information
                        res_temporal = self.kv_temporal * self.cfg.PATH_WEIGHT + res_temporal

            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)

            xt = x[:, 1:, :] + res_temporal

            ## Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            if self.cfg.ADAPTER:
                # # task 1-9 train/val时
                if signal_flag == 0 or signal_flag == 1:
                    if self.age > 0:
                        spatial_res_spatial_ada = []
                        for i in range(0, self.age):
                            # calculate att for every task, every task is ada from all tasks before
                            spatial_res_spatial_ada.append(self.Spatial_adapter[i](res_spatial))
                            # concat current task and task before's information and cal att_via_task
                        kv = torch.cat((res_spatial, torch.cat(spatial_res_spatial_ada, dim=1).squeeze(0)), dim=1)
                        self.kv_spatial = self.Spatial_attn_via_tasks(
                            spatial_res_spatial_ada[self.age - 1], kv)
                        # add new task's information
                        res_spatial = self.kv_spatial * self.cfg.PATH_WEIGHT + res_spatial
                else:
                    if self.age > 1:
                        spatial_res_spatial_ada = []
                        for i in range(0, self.age - 1):
                            # calculate att for every task, every task is ada from all tasks before
                            spatial_res_spatial_ada.append(self.Spatial_adapter[i](res_spatial))
                            # concat current task and task before's information and cal att_via_task
                        kv = torch.cat((res_spatial, torch.cat(spatial_res_spatial_ada, dim=1).squeeze(0)), dim=1)
                        self.kv_spatial = self.Spatial_attn_via_tasks_old(
                            spatial_res_spatial_ada[self.age - 2], kv)
                        # add new task's information
                        res_spatial = self.kv_spatial * self.cfg.PATH_WEIGHT + res_spatial

                # Taking care of CLS token

            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)

            x = x + self.drop_path(self.mlp(self.norm2(x)))

            res_spatial = torch.cat((cls_token, res), 1)
            res_spatial = res_spatial + self.drop_path(self.mlp(self.norm2(res_spatial)))

            # res_temporal = torch.cat((cls_token, res_temporal), 1)
            res_temporal = res_temporal + self.drop_path(self.mlp(self.norm2(res_temporal)))

            return x, res_spatial, res_temporal


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', dropout=0., age=0, training=True, cfg=None):
        super().__init__()
        self.head_o = None
        self.feat = None
        self.age = age
        self.attention_type = attention_type
        self.depth = depth
        self.training = training
        self.cfg = cfg
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type, age=age, cfg=self.cfg)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        self.Ada_mlp = SAdapter2(embed_dim)
        # for cosine classifier
        #  head
        if cfg.HEAD_TYPE == 'cosine':
            if age == 0 or (age == 1 and self.training):
                self.head = CosineLinear(embed_dim, num_classes, is_train=self.training)
            else:
                self.head = SplitCosineLinear(embed_dim, num_classes - len(cfg.CURRENT_TASK), len(cfg.CURRENT_TASK),
                                              is_train=self.training)
            print("cosine classifier")
        else:
            if self.age>0:
                self.head = Classifier(embed_dim, self.cfg.NUM_OLD_HEAD)
            else:
                self.head = Classifier(embed_dim, self.cfg.LEN_lIST[0])
            self.new_head = Classifier(embed_dim, self.cfg.LEN_lIST[self.age])

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def freeze_parameters(m, requires_grad=False):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        return self.freeze_parameters(self)

    # for cosine classifier
    def increment_head(self, inc_class, age):
        in_features = self.head.in_features
        out_features = self.head.out_features

        new_head = SplitCosineLinear(in_features, out_features, inc_class - out_features, is_train=self.training)

        if age == 1:
            new_head.fc1.weight.data = self.head.weight.data.clone()

            new_head.sigma.data = self.head.sigma.data.clone()

            new_head.eta.data = self.head.eta.data.clone()
        else:
            out_features1 = self.head.fc1.out_features

            new_head.fc1.weight.data[:out_features1] = self.head.fc1.weight.data.clone()
            new_head.fc1.weight.data[out_features1:] = self.head.fc2.weight.data.clone()

            new_head.sigma.data = self.head.sigma.data.clone()

            new_head.eta.data = self.head.eta.data.clone()

        del self.head
        self.head = new_head

    def increment_head_linear(self):

        old_head = Classifier(self.embed_dim, self.cfg.NUM_OLD_HEAD)
        old_head.head.weight.data[:self.cfg.LEN_lIST[0]] = \
            self.head.head.weight.data.clone()
        old_head.head.bias.data[:self.cfg.LEN_lIST[0]] = \
            self.head.head.bias.data.clone()
        head_length = 0
        for i in range(1, self.age):
            head_length += self.cfg.LEN_lIST[i - 1]
            self.head_o = Classifier(self.embed_dim, self.cfg.LEN_lIST[i])
            self.head_o.load_state_dict(torch.load(os.path.join(self.cfg.DIR,
                'model_head{}.pt'.format(i))))
            old_head.head.weight.data[head_length:head_length + self.cfg.LEN_lIST[i]] = \
                self.head_o.head.weight.data.clone()
            old_head.head.bias.data[head_length:head_length + self.cfg.LEN_lIST[i]] = \
                self.head_o.head.bias.data.clone()
            del self.head_o
        self.head = old_head

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, flag3):
        global res_spatial, res_temporal, xs
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        #   # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        #   # Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            #   # Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        x3 = x
        #  Attention blocks  x:T&S, X1:T, X2:S, X3:None
        if flag3 == 0:
            # age > 0, while training
            for blk in self.blocks:
                x, res_spatial, res_temporal = blk(x, B, T, W, 0)
                x3, _, _ = blk(x3, B, T, W, 3)
            x = self.norm(x)
            x3 = self.norm(x3)
            res_spatial = self.norm(res_spatial)
            res_temporal = self.norm(res_temporal)
            #  feature  x:T&S, X1:T, X2:S, X3:None
            return x[:, 0], x3[:, 0], res_spatial[:, 0], res_temporal[:, 0]

        elif flag3 == 1:
            # while inference and age 0's training
            for blk in self.blocks:
                x, _, _ = blk(x, B, T, W, 0)
            #   # Predictions for space-only baseline
            x = self.norm(x)
            return x[:, 0]

        elif flag3 == 2:
            # while mixing S
            xt = x
            for blk in self.blocks:
                x, _, _ = blk(x, B, T, W, 1)
                xt, _, _ = blk(xt, B, T, W, 2)
            xs = self.norm(x)
            xt = self.norm(xt)
            return xs[:, 0], xt[:, 0]

        elif flag3 == 3:
            # while mixing T
            for blk in self.blocks:
                x, _, _ = blk(x, B, T, W, 2)
            xt = self.norm(x)
            return xt[:, 0]

        elif flag3 == 4:
            # while mixing delta temporal & spatial
            for blk in self.blocks:
                x, res_spatial, res_temporal = blk(x, B, T, W, 0)
            x = self.norm(x)
            res_spatial = self.norm(res_spatial)
            res_temporal = self.norm(res_temporal)
            return x[:, 0], res_spatial[:, 0], res_temporal[:, 0]

        else:
            # while calculating similarity
            for blk in self.blocks:
                x, res_spatial, res_temporal = blk(x, B, T, W, 3)
            x = self.norm(x)
            res_spatial = self.norm(res_spatial)
            res_temporal = self.norm(res_temporal)
            return x[:, 0], res_spatial[:, 0], res_temporal[:, 0]

    def forward(self, x, flag3):
        if flag3 == 0:
            # age > 0, while training
            xf, x3f, res_spatial, res_temporal = self.forward_features(x, flag3)
            self.feat = xf
            x = self.new_head(xf)
            x3 = self.new_head(x3f)
            logits_spatial = self.new_head(res_spatial)
            logits_temporal = self.new_head(res_temporal)
            logits_spatial0 = self.head(res_spatial)
            logits_temporal0 = self.head(res_temporal)
            logits_spatial = torch.cat((logits_spatial0, logits_spatial), dim=1)
            logits_temporal = torch.cat((logits_temporal0, logits_temporal), dim=1)
            x0 = self.head(xf)
            x03 = self.head(x3f)
            #  Logits  x:T&S, X1:S, X2:T, X3:None
            x = torch.cat((x0, x), dim=1)
            x3 = torch.cat((x03, x3), dim=1)
            return x, self.feat, x3, logits_spatial, logits_temporal

        elif flag3 == 1:
            # while inference and age 0's training
            xf = self.forward_features(x, flag3)
            self.feat = xf
            if self.cfg.HEAD_TYPE != 'cosine':
                if self.age > 0:
                    x = self.new_head(xf)
                    x0 = self.head(xf)
                    x = torch.cat((x0, x), dim=1)
                else:
                    x = self.head(xf)
            else:
                x = self.head(xf, self.training)
            return x, self.feat

        elif flag3 == 2:
            # while mixing spatial

            xsf, xtf = self.forward_features(x, flag3)
            xs = self.new_head(xsf)
            xs0 = self.head(xsf)
            xs = torch.cat((xs0, xs), dim=1)
            xt = self.new_head(xtf)
            xt0 = self.head(xtf)
            xt = torch.cat((xt0, xt), dim=1)
            return xs, xt

        elif flag3 == 3:
            # while mixing temporal

            xsf = self.forward_features(x, flag3)
            xs = self.new_head(xsf)
            xs0 = self.head(xsf)
            xs = torch.cat((xs0, xs), dim=1)
            return xs

        elif flag3 == 4:
            # while mixing delta temporal & spatial
            x, res_spatial, res_temporal = self.forward_features(x, flag3)
            self.feat = x
            x = self.head(x)
            logits_spatial0 = self.head(res_spatial)
            logits_temporal0 = self.head(res_temporal)
            logits_spatial = self.new_head(res_spatial)
            logits_temporal = self.new_head(res_temporal)
            logits_spatial = torch.cat((logits_spatial0, logits_spatial), dim=1)
            logits_temporal = torch.cat((logits_temporal0, logits_temporal), dim=1)
            return x, self.feat, logits_spatial, logits_temporal

        else:
            # while calculating similarity
            x, res_spatial, res_temporal = self.forward_features(x, flag3)
            self.feat = x
            x = self.head(x)
            logits_spatial = self.head(res_spatial)
            logits_temporal = self.head(res_temporal)
            return x, self.feat, logits_spatial, logits_temporal


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.training = True
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.NUM_HEAD,
                                       patch_size=patch_size,
                                       embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                       attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES,
                                       attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, age=cfg.AGE,
                                       training=self.training,cfg=cfg, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.cfg = cfg
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            print("load pretrain")
            load_pretrained(self.model, num_classes=cfg.NUM_HEAD, in_chans=kwargs.get('in_chans', 3),
                            filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches,
                            attention_type=self.attention_type, pretrained_model=pretrained_model)

    def increment_head_linear(self):
        return self.model.increment_head_linear()

    def increment_head(self, head, age):
        return self.model.increment_head(head, age)

    def forward(self, x, flag3):
        #  Logits  x:T&S, X1:T, X2:S, X3:None flag3: 0:training 1:inference 2:mixing 3.sim
        if flag3 == 0:
            if self.cfg.AGE > 0:
                x, feat, x3, logits_spatial, logits_temporal = self.model(x, flag3)
                return x, feat, x3, logits_spatial, logits_temporal
        elif flag3 == 1:
            x, feat = self.model(x, flag3)
            return x, feat
        elif flag3 == 2:
            if self.cfg.AGE > 0:
                xs, xt = self.model(x, flag3)
                return xs, xt
        elif flag3 == 3:
            if self.cfg.AGE > 0:
                xs = self.model(x, flag3)
                return xs
        else:
            x, feat, logits_spatial, logits_temporal = self.model(x, flag3)
            return x, feat, logits_spatial, logits_temporal


@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',
                 pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained = True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768,
                                       depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type,
                                       **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch' + str(patch_size) + '_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3),
                            filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames,
                            num_patches=self.num_patches, attention_type=self.attention_type,
                            pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x