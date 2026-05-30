import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np

from .style import EqualLinear, StyleLayer, StyleLayer_norm_scale_shift
from functools import (partial, reduce)
import operator

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head 
        # b * pixels * features +coord -> b * pixels *rgb

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):

        return self.conv(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class MMAttentionEncoderv1(nn.Module):
    def __init__(self,
                in_dim:int=1,
                out_dim:int=1,
                hidden_dim:int=64,
                in_module_depth:int=4,
                out_module_depth:int=4
                ):
        super(MMAttentionEncoderv1, self).__init__()        
        self.first_layer = StyleLayer(in_dim, hidden_dim, 3, bias=True, activate=True)
        self.input_channels = in_dim

        in_module = nn.ModuleList()

        for _ in range(in_module_depth):
            in_module.append(
                ResnetBlocWithAttn(hidden_dim, 
                    hidden_dim, 
                    noise_level_emb_dim=hidden_dim, 
                    norm_groups=hidden_dim//4, 
                    dropout=0.2, 
                    with_attn=True)
            )

        self.in_module = nn.Sequential(*in_module)

        self.hin_embedding = nn.Sequential(
            PositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        out_module = nn.ModuleList()

        for _ in range(out_module_depth):
            out_module.append(
                ResnetBlocWithAttn(hidden_dim, 
                    hidden_dim, 
                    noise_level_emb_dim=hidden_dim, 
                    norm_groups=hidden_dim//4, 
                    dropout=0.2, 
                    with_attn=True)
            )

        self.out_module = nn.Sequential(*out_module)

        self.hout_embedding = nn.Sequential(
            PositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.last_layer = StyleLayer(hidden_dim, out_dim, 3, bias=True, activate=True)
                
    def forward(self, x, h_in, h_out):
        x = self.first_layer(x)
    
        hin_embed = self.hout_embedding(h_in)
        for layer in self.in_module:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, hin_embed)

        hout_embed = self.hout_embedding(h_out)
        for layer in self.out_module:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, hout_embed)

        x = self.last_layer(x)
        return x
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class MMAttentionEncoderv2(nn.Module):
    def __init__(self,
                in_dim:int=1,
                out_dim:int=1,
                hidden_dim:int=64,
                module_depth:int=16,
                ):
        super(MMAttentionEncoderv2, self).__init__()        
        self.first_layer = StyleLayer(in_dim, hidden_dim, 3, bias=True, activate=True)
        self.input_channels = in_dim

        mid_module = nn.ModuleList()

        for _ in range(module_depth):
            mid_module.append(
                ResnetBlocWithAttn(hidden_dim, 
                    hidden_dim, 
                    noise_level_emb_dim=hidden_dim, 
                    norm_groups=hidden_dim//8, 
                    dropout=0.1, 
                    with_attn=True)
            )

        self.mid_module = nn.Sequential(*mid_module)

        self.height_embedding = nn.Sequential(
            PositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.last_layer = StyleLayer(hidden_dim, out_dim, 3, bias=True, activate=True)
                
    def forward(self, x, h_in, h_out):
        x = self.first_layer(x)
    
        h_embed = self.height_embedding(torch.log(h_out/h_in))
        for layer in self.mid_module:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, h_embed)

        x = self.last_layer(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

if __name__ == "__main__":
    batch_size = 4
    device = torch.device('cuda')
    model = MMAttentionEncoderv1(in_dim=1, out_dim=1, hidden_dim=64).to(device)
    x = torch.Tensor(batch_size, 1, 15, 20).to(device)

    h_in = torch.Tensor(batch_size, 1).to(device)
    h_out = torch.Tensor(batch_size, 1).to(device)

    y = model(x, h_in, h_out)

    print(y.shape)
    model._count_params()