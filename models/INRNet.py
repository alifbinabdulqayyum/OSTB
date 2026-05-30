'''
import torch

from torch import nn
from functools import partial

import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


non_act = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}


class ResidualINR(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        act: str = 'relu',
        ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim + coord_dim, hidden_dim))
        layers.append(non_act[act]())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, coords):
        return x + self.layers(torch.cat([x, coords], dim=-1))
    
class ResidualBilinearINR(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 2, 
        hidden_dim: int = 512,
        act: str = 'relu',
        ):
        super().__init__()

        self.bilin_layer = nn.Bilinear(
            in1_features=input_dim, 
            in2_features=coord_dim, 
            out_features=hidden_dim
        )
        self.bilin_non_lin = non_act[act]()

        lin_layers = []

        lin_layers.append(
            nn.Linear(
                in_features=hidden_dim, 
                out_features=input_dim
            )
        )

        self.lin_layers = nn.Sequential(*lin_layers)

    def forward(self, z, coord):
        x = self.bilin_non_lin(self.bilin_layer(z, coord))
        return x + self.lin_layers(x)
'''
import torch

from torch import nn
from functools import partial

import torch.nn.functional as F

from einops import rearrange

from functools import partial, reduce
import operator

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


non_act = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}
       
class SpatialPreservedConv(nn.Conv2d):
    """
    To keep spatial size of input same as output, this code only work for stride step = 1, and kernel size is odd number.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        if kernel_size % 2 == 0:
            NotImplementedError("When stride is 1, this only works for odd kernel size.")

        super(SpatialPreservedConv, self).__init__(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class INR(nn.Module):
    def __init__(self, 
        in_dim: int = 64, 
        out_dim: int = 1, 
        hidden_dim: list = [256, 256, 256, 256], 
        act: str = None,
        ):
        super().__init__()

        layers = []
        lastv = in_dim
        for hidden in hidden_dim:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(non_act[act]())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class INRwithGaussianAttention(nn.Module):
    def __init__(self, 
        in_dim: int = 64, 
        out_dim: int = 1, 
        hidden_dim: int = 256, 
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 4,
        act: str = None,
        ):
        super().__init__()

        layers = []

        norm_axes = [1 for _ in range(num_layers)]
        num_heads = [num_heads for _ in range(num_layers)]
        num_gaussians = [num_gaussians for _ in range(num_layers)]

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(GaussianBlock(norm_axes=norm_axes, num_heads=num_heads, num_gaussians=num_gaussians, num_layers=num_layers))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class ResidualINR(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        act: str = 'relu',
        ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim + coord_dim, hidden_dim))
        layers.append(non_act[act]())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, coords):
        return x + self.layers(torch.cat([x, coords], dim=-1))
    
class ResidualBilinearINR(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 2, 
        hidden_dim: int = 512,
        act: str = 'relu',
        ):
        super().__init__()

        self.bilin_layer = nn.Bilinear(
            in1_features=input_dim, 
            in2_features=coord_dim, 
            out_features=hidden_dim
        )
        self.bilin_non_lin = non_act[act]()

        lin_layers = []

        lin_layers.append(
            nn.Linear(
                in_features=hidden_dim, 
                out_features=input_dim
            )
        )

        self.lin_layers = nn.Sequential(*lin_layers)

    def forward(self, z, coord):
        x = self.bilin_non_lin(self.bilin_layer(z, coord))
        return x + self.lin_layers(x)
        
class INN(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        in_mod_depth: int = 5,
        out_mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        ):
        super().__init__()

        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(in_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.in_mod_layers = nn.Sequential(*layers)
        
        layers = []
        for _ in range(out_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.out_mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, in_mod_coords, out_mod_coords):
        x = self.head(x)
        
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        
        in_mod_coords = rearrange(in_mod_coords,
                  'b h w c -> (b h w) c')
        in_mod_coords = in_mod_coords.contiguous()
        for layer in self.in_mod_layers:
            x = layer(x, in_mod_coords)
        
        out_mod_coords = rearrange(out_mod_coords,
                  'b h w c -> (b h w) c')
        out_mod_coords = out_mod_coords.contiguous()
        for layer in self.out_mod_layers:
            x = layer(x, out_mod_coords)
            
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class EncoderINN(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        out_h: int = 15,
        out_w: int = 20,
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)

        # x = F.interpolate(x, size=[self.out_h, self.out_w])
        x = F.interpolate(x, size=[self.out_h, self.out_w], mode='area')
        
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        
        mod_coords = rearrange(mod_coords,
                  'b h w c -> (b h w) c')
        mod_coords = mod_coords.contiguous()
        for layer in self.mod_layers:
            x = layer(x, mod_coords)
            
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class DecoderINN(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        out_h: int = 120,
        out_w: int = 160,
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINR(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)
        
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        
        mod_coords = rearrange(mod_coords,
                  'b h w c -> (b h w) c')
        mod_coords = mod_coords.contiguous()
        for layer in self.mod_layers:
            x = layer(x, mod_coords)
            
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)

        # x = F.interpolate(x, size=[self.out_h, self.out_w])
        x = F.interpolate(x, size=[self.out_h, self.out_w], mode='bicubic')
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

### INR with Attention
        
import math
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
    
class ResidualINRwithAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        act: str = 'relu',
        ):
        super().__init__()

        self.attn = SelfAttention(in_channel=input_dim, norm_groups=input_dim//4)
        self.resid_inr = ResidualINR(input_dim=input_dim, coord_dim=coord_dim, hidden_dim=hidden_dim, act=act)

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = self.attn(x)
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        coords = rearrange(coords,
                  'b h w c -> (b h w) c')
        coords = coords.contiguous()
        x = self.resid_inr(x, coords)
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        return x

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class ResidualBilinearINRwithAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        act: str = 'relu',
        ):
        super().__init__()

        self.attn = SelfAttention(in_channel=input_dim, norm_groups=input_dim//4)
        self.resid_inr = ResidualBilinearINR(input_dim=input_dim, coord_dim=coord_dim, hidden_dim=hidden_dim, act=act)

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = self.attn(x)
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        coords = rearrange(coords,
                  'b h w c -> (b h w) c')
        coords = coords.contiguous()
        x = self.resid_inr(x, coords)
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        return x

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class INNwithAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        in_mod_depth: int = 5,
        out_mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        ):
        super().__init__()

        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(in_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.in_mod_layers = nn.Sequential(*layers)
        
        layers = []
        for _ in range(out_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.out_mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, in_mod_coords, out_mod_coords):
        x = self.head(x)
        
        for layer in self.in_mod_layers:
            x = layer(x, in_mod_coords)
        
        for layer in self.out_mod_layers:
            x = layer(x, out_mod_coords)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class EncoderINNwithAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        out_h: int = 15,
        out_w: int = 20,
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)

        # x = F.interpolate(x, size=[self.out_h, self.out_w])
        x = F.interpolate(x, size=[self.out_h, self.out_w], mode='area')
        
        for layer in self.mod_layers:
            x = layer(x, mod_coords)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class DecoderINNwithAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        out_h: int = 15,
        out_w: int = 20,
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)
        
        for layer in self.mod_layers:
            x = layer(x, mod_coords)
        
        # x = F.interpolate(x, size=[self.out_h, self.out_w])
        x = F.interpolate(x, size=[self.out_h, self.out_w], mode='bicubic')

        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

### INR with Gaussian Attention
        
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
    
class ResidualINRwithGaussianAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 1,
        act: str = 'relu',
        ):
        super().__init__()
        norm_axes = [1 for _ in range(num_layers)]
        num_heads = [num_heads for _ in range(num_layers)]
        num_gaussians = [num_gaussians for _ in range(num_layers)]
        self.attn = GaussianBlock(norm_axes, num_heads, num_gaussians, num_layers)
        self.resid_inr = ResidualINR(input_dim=input_dim, coord_dim=coord_dim, hidden_dim=hidden_dim, act=act)

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = self.attn(x)
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        coords = rearrange(coords,
                  'b h w c -> (b h w) c')
        coords = coords.contiguous()
        x = self.resid_inr(x, coords)
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        return x

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class ResidualBilinearINRwithGaussianAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 64, 
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 1,
        act: str = 'relu',
        ):
        super().__init__()
        norm_axes = [1 for _ in range(num_layers)]
        num_heads = [num_heads for _ in range(num_layers)]
        num_gaussians = [num_gaussians for _ in range(num_layers)]
        self.attn = GaussianBlock(norm_axes, num_heads, num_gaussians, num_layers)
        self.resid_inr = ResidualBilinearINR(input_dim=input_dim, coord_dim=coord_dim, hidden_dim=hidden_dim, act=act)

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = self.attn(x)
        x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w, c=c)
        coords = rearrange(coords,
                  'b h w c -> (b h w) c')
        coords = coords.contiguous()
        x = self.resid_inr(x, coords)
        x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)
        return x

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

class INNwithGaussianAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        in_mod_depth: int = 5,
        out_mod_depth: int = 5,
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 1,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        ):
        super().__init__()

        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(in_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            else:
                raise NotImplementedError
        self.in_mod_layers = nn.Sequential(*layers)
        
        layers = []
        for _ in range(out_mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            else:
                raise NotImplementedError
        self.out_mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, in_mod_coords, out_mod_coords):
        x = self.head(x)
        
        for layer in self.in_mod_layers:
            x = layer(x, in_mod_coords)
        
        for layer in self.out_mod_layers:
            x = layer(x, out_mod_coords)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

# def get_coords(shape, ranges=None, flatten=True):
#     """ 
#     Make coordinates at grid centers.
#     Args:
#         shape   (list): image size [H, W]
#         ranges  (list): grid boundaries [[left, right], [down, up]] 
#         flatten (bool): True
#     Returns:
#         coords  (torch.tensor): H * W, 2
#     """
#     # determine the center of each grid
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         if ranges is None:
#             v0, v1 = -0.99995, 1 #-1, 1
#         else:
#             v0, v1 = ranges[i]
#         r = (v1 - v0) / (2 * n)
#         seq = v0 + r + (2 * r) * torch.arange(n).float()
#         coord_seqs.append(seq)
    
#     # make mesh
#     coords = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
#     if flatten:
#         coords = coords.view(-1, coords.shape[-1])
#     return coords

# def look_up_feature(
#     coordinate: torch.Tensor, 
#     feature: torch.Tensor, 
#     feat_coord: torch.Tensor
# ):
#     '''
#     Args:
#         coordinate (torch.tensor) - (b, n_query_pts, 2)
#         feature    (torch.tensor) - (b, n_feature, H, W)
#     Returns:
#         feature      (torch.tensor) - (b, n_query_pts, n_feature)
#         f_coordinate (torch.tensor) - (b, n_query_pts, 2)
#     '''
#     feature = F.grid_sample(
#                 feature, 
#                 coordinate.flip(-1).unsqueeze(1), # (b, 1, n_feature, 2)
#                 mode='nearest', 
#                 align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, n_feature)

#     f_coordinate = F.grid_sample(
#                     feat_coord, 
#                     coordinate.flip(-1).unsqueeze(1),
#                     mode='nearest', 
#                     align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, 2)

#     return feature, f_coordinate

class DecoderINNwithGaussianAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        out_h: int = 15,
        out_w: int = 20,
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 1,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)
        
        for layer in self.mod_layers:
            x = layer(x, mod_coords)
        
        # # for layer in self.out_mod_layers:
        # #     x = layer(x, out_mod_coords)
        # b, c, feat_h, feat_w = x.shape

        # feat_coord = get_coords([feat_h, feat_w], flatten=False).type_as(x) \
        #     .permute(2, 0, 1) \
        #     .unsqueeze(0).expand(b, 2, *[feat_h, feat_w]) # (b, 2, H, W)
        
        # reduced_feat_coord = get_coords([self.out_h, self.out_w], flatten=True).type_as(x)

        # x, _ = look_up_feature(
        #     coordinate=reduced_feat_coord.unsqueeze(0).repeat(x.shape[0],1,1),
        #     feature=x,
        #     feat_coord=feat_coord
        # )

        # x = rearrange(x, 'b (h w) c -> b c h w', b=b, c=c, h=self.out_h, w=self.out_w)

        # x = F.interpolate(x, size=[self.out_h, self.out_w], mode='bicubic')
        x = F.interpolate(x, size=[self.out_h, self.out_w])
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))


class EncoderINNwithGaussianAttention(nn.Module):
    def __init__(self, 
        input_dim: int = 1, 
        out_dim: int = 1,
        feature_dim: int = 64,
        coord_dim: int = 3,
        hidden_dim: int = 256, 
        mod_depth: int = 5,
        out_h: int = 120,
        out_w: int = 160,
        num_heads: int = 1,
        num_gaussians: int = 16,
        num_layers: int = 1,
        encoder_net: str = 'residual linear',
        act: str = 'relu',
        ):
        super().__init__()

        self.out_h, self.out_w = out_h, out_w
        self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
        layers = []
        for _ in range(mod_depth):
            if encoder_net == 'residual linear':
                layers.append(ResidualINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            elif encoder_net == 'residual bilinear':
                layers.append(ResidualBilinearINRwithGaussianAttention(input_dim = feature_dim, 
                    coord_dim = coord_dim,
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    num_gaussians = num_gaussians,
                    num_layers = num_layers))
            else:
                raise NotImplementedError
        self.mod_layers = nn.Sequential(*layers)
        
        self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
    def forward(self, x, mod_coords):
        x = self.head(x)

        x = F.interpolate(x, size=[self.out_h, self.out_w])
        # x = F.interpolate(x, size=[self.out_h, self.out_w], mode='area')
        
        # b, c, feat_h, feat_w = x.shape

        # feat_coord = get_coords([feat_h, feat_w], flatten=False).type_as(x) \
        #     .permute(2, 0, 1) \
        #     .unsqueeze(0).expand(b, 2, *[feat_h, feat_w]) # (b, 2, H, W)
        
        # reduced_feat_coord = get_coords([self.out_h, self.out_w], flatten=True).type_as(x)

        # x, _ = look_up_feature(
        #     coordinate=reduced_feat_coord.unsqueeze(0).repeat(x.shape[0],1,1),
        #     feature=x,
        #     feat_coord=feat_coord
        # )

        # x = rearrange(x, 'b (h w) c -> b c h w', b=b, c=c, h=self.out_h, w=self.out_w)

        for layer in self.mod_layers:
            x = layer(x, mod_coords)
        
        x = self.tail(x)
        return x
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

# from neuralop.models import FNO, SFNO, TFNO

# class EncoderNeuralOpINN(nn.Module):
#     def __init__(self, 
#         n_modes: tuple = (64,64),
#         input_dim: int = 1, 
#         out_dim: int = 1,
#         feature_dim: int = 64,
#         mod_depth: int = 5,
#         out_h: int = 120,
#         out_w: int = 160,
#         m: int = 30,
#         neural_op: 'str' = 'TFNO', 
#         ):
#         super().__init__()

#         self.out_h, self.out_w = out_h, out_w
#         self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
#         if neural_op == 'FNO':
#             neural_op_func = FNO
#         elif neural_op == 'SFNO':
#             neural_op_func = SFNO 
#         elif neural_op == 'TFNO':
#             neural_op_func = TFNO
#         else:
#             raise NotImplementedError("Neural Operator not implemented")
#         layers = []
#         for _ in range(mod_depth):
#             layers.append(
#                 neural_op_func(
#                     n_modes=n_modes,
#                     in_channels=feature_dim+6*m,
#                     out_channels=feature_dim,
#                     hidden_channels=64,#feature_dim*8,
#                     # projection_channel_ratio=1
#                     factorization='tucker',
#                     implementation='factorized',
#                     rank=0.05,
#                     n_layers=1,
#                     positional_embedding=None,
#                 )
#             )
#         self.mod_layers = nn.Sequential(*layers)
        
#         self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
#     def forward(self, x, h):
#         x = self.head(x)
#         h = rearrange(h, 'b h w c -> b c h w')

#         x = F.interpolate(x, size=[self.out_h, self.out_w])

#         for layer in self.mod_layers:
#             x = layer(torch.cat((x, h), dim=1))
        
#         x = self.tail(x)
#         return x
    
#     def _count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
#         print(f"Total params: %.2fM" % (c/1000000.0))
#         print(f"Total params: %.2fk" % (c/1000.0))


# class DecoderNeuralOpINN(nn.Module):
#     def __init__(self, 
#         n_modes: tuple = (64,64),
#         input_dim: int = 1, 
#         out_dim: int = 1,
#         feature_dim: int = 64,
#         mod_depth: int = 5,
#         out_h: int = 120,
#         out_w: int = 160,
#         m: int = 30,
#         neural_op: 'str' = 'TFNO', 
#         ):
#         super().__init__()

#         self.out_h, self.out_w = out_h, out_w
#         self.head = SpatialPreservedConv(in_channels=input_dim, out_channels=feature_dim, kernel_size=3)
#         if neural_op == 'FNO':
#             neural_op_func = FNO
#         elif neural_op == 'SFNO':
#             neural_op_func = SFNO 
#         elif neural_op == 'TFNO':
#             neural_op_func = TFNO
#         else:
#             raise NotImplementedError("Neural Operator not implemented")
#         layers = []
#         for _ in range(mod_depth):
#             layers.append(
#                 neural_op_func(
#                     n_modes=n_modes,
#                     in_channels=feature_dim+6*m,
#                     out_channels=feature_dim,
#                     hidden_channels=64,#feature_dim*8,
#                     # projection_channel_ratio=1
#                     factorization='tucker',
#                     implementation='factorized',
#                     rank=0.05,
#                     n_layers=1,
#                 )
#             )
#         self.mod_layers = nn.Sequential(*layers)
        
#         self.tail = SpatialPreservedConv(in_channels=feature_dim, out_channels=out_dim, kernel_size=3)
        
#     def forward(self, x, h):
#         x = self.head(x)
#         h = rearrange(h, 'b h w c -> b c h w')

#         # b, _, out_h, out_w = x.shape

#         for layer in self.mod_layers:
#             x = layer(torch.cat((x, h), dim=1))

#         x = F.interpolate(x, size=[self.out_h, self.out_w])
        
#         x = self.tail(x)
#         return x
    
#     def _count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
#         print(f"Total params: %.2fM" % (c/1000000.0))
#         print(f"Total params: %.2fk" % (c/1000.0))