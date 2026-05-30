import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


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
        
class Upsampler(nn.Sequential):
    def __init__(
        self, 
        scale:int=4, 
        n_feats:int=64, 
        bn:bool=False, 
        act:str='relu', 
        bias:bool=True
    ):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(SpatialPreservedConv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act:
                    m.append(non_act[act]())

        elif scale == 3:
            m.append(SpatialPreservedConv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act:
                m.append(non_act[act]())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        
class ResBlock(nn.Module):
    def __init__(self, 
        n_feats: int = 64, 
        kernel_size: int = 3,
        bias: bool = True, 
        bn: bool = False, 
        act: str = 'relu', 
        res_scale: int = 1
        ):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(SpatialPreservedConv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(non_act[act]())

        self.net = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.net(x).mul(self.res_scale)
        res += x
        return res


class EDSR(nn.Module):
    def __init__(self,
        n_inputs: int = 1,
        n_feats: int = 64,
        kernel_size: int = 3,
        n_resblocks: int = 16,
        bias: bool = True,
        bn: bool = False,
        act: str = None,
        res_scale: int = 1,
        scale: int = 4,
        upsampling: bool = True,
        weighted_upsampling: bool = False,
        ):
        super().__init__()

        # define head module
        m_head = [SpatialPreservedConv(n_inputs, n_feats, kernel_size, bias)]
        self.head = nn.Sequential(*m_head)

        # define body module
        m_body = [ResBlock(n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        m_body.append(SpatialPreservedConv(n_feats, n_feats, kernel_size, bias))
        self.body = nn.Sequential(*m_body)

        # define tail module
        self.upsampling = upsampling
        self.weighted_upsampling = weighted_upsampling
        self.scale = scale
        
        if self.upsampling:
            m_tail = []
            if self.weighted_upsampling:
                m_tail.append(
                    Upsampler(scale=scale, n_feats=n_feats, bn=bn, bias=bias, act=act)
                )
            m_tail.append(SpatialPreservedConv(n_feats, n_feats, kernel_size, bias))

            self.tail = nn.Sequential(*m_tail)
        
    def forward(self, 
        x
        ):
        h, w = x.shape[2:]
        x = self.head(x)
        res = self.body(x)
        res += x
        
        if not self.upsampling:
            x = res
        else:
            if not self.weighted_upsampling:
                res = F.interpolate(res, [self.scale*h, self.scale*w])
            x = self.tail(res)
            
        return x

class CascadedEDSR(nn.Module):
    def __init__(self,
        n_blocks: int = 3,
        n_inputs: int = 1,
        n_feats: int = 64,
        kernel_size: int = 3,
        n_resblocks: int = 16,
        bias: bool = True,
        bn: bool = False,
        act: str = None,
        res_scale: int = 1,
        scale: int = 4,
        upsampling: bool = True,
        weighted_upsampling: bool = False,
        ):
        super().__init__()

        module = []

        for idx in range(n_blocks):
            module.append(EDSR(n_inputs=n_inputs if idx==0 else n_feats,
                                    n_feats=n_feats,
                                    kernel_size=kernel_size,
                                    n_resblocks=n_resblocks,
                                    bias=bias,
                                    bn=bn,
                                    act=act,
                                    res_scale=res_scale,
                                    scale=scale,
                                    upsampling=upsampling,
                                    weighted_upsampling=weighted_upsampling))
            
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

if __name__ == "__main__":
    model = CascadedEDSR(n_blocks=3, n_inputs=1, n_feats=16, kernel_size=3, n_resblocks=16, bias=True, bn=False, act="relu", res_scale=1, scale=4, upsampling=True, weighted_upsampling=False)
    x = torch.Tensor(16, 1, 60, 80)
    y = model(x)
    print(y.shape)
