import math
from multiprocessing.spawn import is_forking
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
# from data.util import *
from functools import (partial, reduce)
import operator

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

# def resize_fn(img, size):
#     return transforms.Resize(size, interpolation=InterpolationMode.BICUBIC)(img)

# import torch.distributed as dist
from einops import rearrange
from .style import EqualLinear, StyleLayer, StyleLayer_norm_scale_shift
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
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

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, shape):
        return self.conv(F.interpolate(x, shape))

class idm(nn.Module):
    def __init__(self, dim,feat_unfold=False,
        local_ensemble=False,
        cell_decode=False,
        include_scale=True):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.include_scale = include_scale
        if self.include_scale:
            self.style = StyleLayer_norm_scale_shift(
                        dim,
                        dim,
                        kernel_size=3,
                        num_style_feat=512,
                        demodulate=True,
                        sample_mode=None,
                        resample_kernel=(1, 3, 3, 1))
        else:
            self.style = StyleLayer(
                        in_channels=dim+dim,
                        out_channels=dim,
                        kernel_size=3,
                        downsample=False,
                        bias=True,
                        activate=False
            )
        if self.cell_decode:
            self.imnet = nn.Sequential(nn.Linear(dim + 2 + 2 , 256),nn.Linear(256, dim))
        else:
            self.imnet = nn.Sequential(nn.Linear(dim + 2, 256),nn.Linear(256, dim))
    def forward(self, x, shape, scale1, scale2, shift):
        coord = make_coord(shape).repeat(x.shape[0], 1, 1).to(x.device)
        cell = torch.ones_like(coord).to(x.device)
        cell[:, 0] *= 2 / shape[-2]
        cell[:, 1] *= 2 / shape[-1]
        return self.query_rgb(x, scale1, scale2, shift, coord, cell)

    def query_rgb(self, x_feat, scale1, scale2, shift, coord, cell=None):
        if self.include_scale:
            # print("befor: ", x_feat.shape, shift.shape)
            feat = self.style(x_feat, noise=None, scale1=scale1, scale2=scale2, shift=shift)
            # print("after: ", feat.shape, shift.shape)
        else:
            feat = self.style(torch.cat([x_feat, shift], dim=1))
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                # print(rel_coord)
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):

        return self.conv(x)


# building block modules


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

class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value, mean_offset_init=0, eps=1e-8):
        super().__init__()
        if not isinstance(norm_axis, int):
            raise ValueError("norm_axis must be an integer.")
        if num_heads <= 0 or not isinstance(num_heads, int):
            raise ValueError("num_heads must be a positive integer.")
        if num_gaussians <= 0 or not isinstance(num_gaussians, int):
            raise ValueError("num_gaussians must be a positive integer.")

        self.norm_axis = norm_axis
        self.eps = eps
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.num_gaussians = num_gaussians

        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians, dtype=torch.float))
        self.c = nn.Parameter(torch.randn(num_gaussians, dtype=torch.float))

    def forward(self, x, return_attention_details=False):
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {x.dim()}.")
        if self.norm_axis >= x.dim() or self.norm_axis < -x.dim():
            raise ValueError(f"norm_axis {self.norm_axis} is out of bounds for input tensor with {x.dim()} dimensions.")

        mask = x != self.padding_value if self.padding_value is not None else None
        x_masked = torch.where(mask, x, torch.zeros_like(x)) if mask is not None else x

        mean = x_masked.mean(dim=self.norm_axis, keepdim=True)
        var = x_masked.var(dim=self.norm_axis, keepdim=True) + self.eps

        mixture = 1
        for i in range(self.num_gaussians):
            adjusted_mean = mean + self.mean_offsets[i]
            y_norm = (x - adjusted_mean) / torch.sqrt(var)
            gaussian = torch.exp(-((y_norm ** 2) / (2.0 * (self.c[i] ** 2)))) / torch.sqrt(2 * torch.pi * (self.c[i] ** 2))
            mixture *= gaussian

        mixture /= mixture.sum(dim=self.norm_axis, keepdim=True).clamp(min=self.eps)

        if return_attention_details:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture, mixture.detach()
        else:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture
            
            
class MultiHeadGaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            GaussianAdaptiveAttention(norm_axis, num_heads, num_gaussians, padding_value, eps)
            for _ in range(num_heads)
        ])

    def forward(self, x, return_attention_details=False):
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        if chunk_size == 0:
            raise ValueError(f"Input tensor size along norm_axis ({self.norm_axis}) must be larger than the number of heads ({self.num_heads}).")

        outputs, attention_details_ = [], []
        for i in range(self.num_heads):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start_index, end_index - start_index)
            if return_attention_details:
                out, mixture = self.attention_heads[i](chunk, return_attention_details=True)
                outputs.append(out)
                attention_details_.append(mixture)
            else:
                outputs.append(self.attention_heads[i](chunk))

        if return_attention_details:
            return torch.cat(outputs, dim=self.norm_axis), torch.cat(attention_details_, dim=self.norm_axis)
        else:
            return torch.cat(outputs, dim=self.norm_axis)
            
            

class GaussianBlock(nn.Module):
    def __init__(self, norm_axes, num_heads, num_gaussians, num_layers, padding_value=None, eps=1e-8):
        super().__init__()
        if len(norm_axes) != num_layers or len(num_heads) != num_layers or len(num_gaussians) != num_layers:
            raise ValueError("Lengths of norm_axes, num_heads, and num_gaussians must match num_layers.")

        self.layers = nn.ModuleList([
            MultiHeadGaussianAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], padding_value, eps)
            for i in range(num_layers)
        ])

    def forward(self, x, return_attention_details=False):
        attention_details_ = {}
        for idx, layer in enumerate(self.layers):
            if return_attention_details:
                x_, attention_details = layer(x, return_attention_details=True)
                attention_details_['layer_'+str(idx)] = attention_details
                x = x_ + x
            else:
                x = layer(x) + x

        if return_attention_details:
            return x, attention_details_
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


class ResnetBlocWithAttn(nn.Module):
    # def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, norm_axis=1, num_heads=1, num_gaussians=1, num_layers=1, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        norm_axes = [norm_axis for _ in range(num_layers)]
        num_heads = [num_heads for _ in range(num_layers)]
        num_gaussians = [num_gaussians for _ in range(num_layers)]
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            # self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
            self.attn = GaussianBlock(norm_axes=norm_axes, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        norm_axis=1,
        num_gaussians=1,
        num_heads=1,
        num_layers=1,
        res_blocks=3,
        dropout=0,
        image_size=128,
        predict_noise:bool=True,
        eps:float=0.002,
        sigma_data:float=1.0,
    ):
        super().__init__()

        self.predict_noise = predict_noise
        self.eps = eps
        self.sigma_data = sigma_data
        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        # self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_first = StyleLayer(out_channel, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        self.condition_scale1 = nn.ModuleList()
        self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()                           
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # iss_last = (ind >= num_mults)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            # if not iss_last:
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            self.condition_scale1.append(
                EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))

            self.condition_scale2.append(
                EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))
        
            self.condition_shift.append(
                StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):

                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.conv_body_down
        self.downs = nn.ModuleList(downs)
        self.final_down1 = StyleLayer(512, 512, 3, downsample=False)
        self.final_down2 = StyleLayer(512, 256, 3, downsample=True)
        self.num_latent, self.num_style_feat = 4, 512
        self.final_linear = EqualLinear(2 *2 * 256, self.num_style_feat * self.num_latent, bias=True, activation='fused_lrelu')
        self.final_styleconv = StyleLayer(512, 512, 3)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            # is_first = (ind == 3)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(idm(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    # def forward(self, x, lr, scaler, time):
    def forward_unet(self, x, lr, scaler, time):
        t = self.noise_level_mlp(time) 
        feat = self.conv_body_first(lr)
        scales1, scales2, shifts = [], [], []
        scale1 = self.condition_scale1[0](scaler)
        scales1.append(scale1.clone())
        scale2 = self.condition_scale2[0](scaler)
        scales2.append(scale2.clone())
        shift = self.condition_shift[0](feat)
        shifts.append(shift.clone())
        j = 1
        for i in range(len(self.conv_body_down)):
            feat = self.conv_body_down[i](feat)
            if j < len(self.condition_scale1) :
                scale1 = self.condition_scale1[j](scaler)
                scales1.append(scale1.clone())
                scale2 = self.condition_scale2[j](scaler)
                scales2.append(scale2.clone())
                shift = self.condition_shift[j](feat)
                shifts.append(shift.clone())
                j += 1


        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x, feats[-1].shape[2:], scales1.pop(), scales2.pop(), shifts.pop())
                x = rearrange(x, 'b (h w) c -> b c h w', h=feats[-1].shape[-2], w=feats[-1].shape[-1])

        return self.final_conv(x)
        # return x

    def forward(self, x_noisy, x_lr, scaler, t):
        # x = torch.cat([x_lr, x_noisy], dim=1)
        #########
        c_skip_t = self.sigma_data**2 / (
            (t - self.eps) ** 2 + self.sigma_data**2
        )
        c_out_t = (
            (t - self.eps)
            * self.sigma_data
            / (t**2 + self.sigma_data**2) ** 0.5
        )
        c_in_t = 1 / (t**2 + self.sigma_data**2) ** 0.5
        #########
        resclaed_t = 1000 * 0.25 * torch.log(t + 1e-44)
        x = self.forward_unet(torch.cat([x_lr, c_in_t[:,None,None,None]*x_noisy], dim=1), x_lr, scaler, resclaed_t)

        return c_skip_t[:,None,None,None]*x_noisy+c_out_t[:,None,None,None]*x
        
        
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))


####################
# UNET without scale
####################
        
class UNet_wout_Scale(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        norm_axis=1,
        num_gaussians=1,
        num_heads=1,
        num_layers=1,
        res_blocks=3,
        dropout=0,
        image_size=128,
        predict_noise:bool=True,
        eps:float=0.002,
        sigma_data:float=1.0,
    ):
        super().__init__()

        self.predict_noise = predict_noise
        self.eps = eps
        self.sigma_data = sigma_data
        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        # self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_first = StyleLayer(out_channel, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        # self.condition_scale1 = nn.ModuleList()
        # self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()                           
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # iss_last = (ind >= num_mults)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            # if not iss_last:
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            # self.condition_scale1.append(
            #     EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))

            # self.condition_scale2.append(
            #     EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))
        
            self.condition_shift.append(
                StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):

                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, 
                    norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.conv_body_down
        self.downs = nn.ModuleList(downs)
        self.final_down1 = StyleLayer(512, 512, 3, downsample=False)
        self.final_down2 = StyleLayer(512, 256, 3, downsample=True)
        self.num_latent, self.num_style_feat = 4, 512
        self.final_linear = EqualLinear(2 *2 * 256, self.num_style_feat * self.num_latent, bias=True, activation='fused_lrelu')
        self.final_styleconv = StyleLayer(512, 512, 3)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            # is_first = (ind == 3)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    norm_axis=norm_axis, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(idm(pre_channel, include_scale=False))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)        

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    # def forward(self, x, lr, scaler, time):
    def forward_unet(self, x, lr, time):
        t = self.noise_level_mlp(time)
        feat = self.conv_body_first(lr)
        shifts = []
        shift = self.condition_shift[0](feat)
        shifts.append(shift.clone())
        j = 1
        for i in range(len(self.conv_body_down)):
            feat = self.conv_body_down[i](feat)
            if j < len(self.condition_shift) :
                shift = self.condition_shift[j](feat)
                shifts.append(shift.clone())
                j += 1


        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                # print("ResnetBlockWithAttn", x.shape)
            else:
                # x = layer(x, feats[-1].shape[2:], scales1.pop(), scales2.pop(), shifts.pop())
                x = layer(x, feats[-1].shape[2:], None, None, shifts.pop())
                x = rearrange(x, 'b (h w) c -> b c h w', h=feats[-1].shape[-2], w=feats[-1].shape[-1])
                # print("IDM", x.shape)

        return self.final_conv(x)
        # return x

    def forward(self, x_noisy, x_lr, scaler, t):
        # x = torch.cat([x_lr, x_noisy], dim=1)
        #########
        c_skip_t = self.sigma_data**2 / (
            (t - self.eps) ** 2 + self.sigma_data**2
        )
        c_out_t = (
            (t - self.eps)
            * self.sigma_data
            / (t**2 + self.sigma_data**2) ** 0.5
        )
        c_in_t = 1 / (t**2 + self.sigma_data**2) ** 0.5
        #########
        resclaed_t = 1000 * 0.25 * torch.log(t + 1e-44)
        x = self.forward_unet(torch.cat([x_lr, c_in_t[:,None,None,None]*x_noisy], dim=1), x_lr, resclaed_t)

        return c_skip_t[:,None,None,None]*x_noisy+c_out_t[:,None,None,None]*x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))