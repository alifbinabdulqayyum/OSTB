import math

import torch.nn.functional as F
import torch.nn as nn
import torch

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


class MaskPredictor(nn.Module):
    def __init__(self,in_channels):
        super(MaskPredictor,self).__init__()
        self.spatial_mask=nn.Conv2d(in_channels=in_channels,out_channels=3,kernel_size=1,bias=False)
    def forward(self,x):
        spa_mask=self.spatial_mask(x)
        spa_mask=F.gumbel_softmax(spa_mask,tau=1,hard=True,dim=1)
        return spa_mask
    
class DyResBlock(nn.Module):
    def __init__(self,
        kernel_size:int=3,
        in_chn:int=1,
        out_chn:int=1,
        act:str='relu'
    ):
        super(DyResBlock,self).__init__()
        self.in_chn=in_chn
        self.shape_l_0=(in_chn,1,kernel_size,kernel_size)
        self.shape_mh_0=(out_chn,in_chn,kernel_size,kernel_size)
        self.shape_h_1=(out_chn,in_chn,kernel_size,kernel_size)
        self.MaskPredictor=MaskPredictor(self.in_chn)
        self.kernel_size=kernel_size
        self.unfold=nn.Unfold(kernel_size=kernel_size,dilation=1,padding=(kernel_size-1) // 2,stride=1)
        self.low_weights_0=nn.Parameter(torch.rand(self.shape_l_0) * 0.001, requires_grad=True)
        self.mid_weights_0=nn.Parameter(torch.rand(self.shape_mh_0) * 0.001,requires_grad=True)
        self.hig_weights_0=nn.Parameter(torch.rand(self.shape_mh_0)*0.001,requires_grad=True)
        self.hig_weights_1=nn.Parameter(torch.rand(self.shape_h_1)*0.001,requires_grad=True)
        self.non_act=non_act[act]()
        self.conv_1_low=nn.Conv2d(in_channels=in_chn,out_channels=in_chn,kernel_size=1,stride=1,bias=False,padding=0)
        self.conv_1_mid=nn.Conv2d(in_channels=in_chn,out_channels=in_chn,kernel_size=1,stride=1,bias=False,padding=0)

    def forward(self,x):
        n,c,h,w=x.size()
        low_weight_0=self.low_weights_0.view(c,self.kernel_size,self.kernel_size)
        mid_weight_0=self.mid_weights_0.view(c,-1)
        hig_weight_0=self.hig_weights_0.view(c,-1)
        hig_weight_1=self.hig_weights_1.view(c,-1)
        MaskPredictor=self.MaskPredictor(x)
        unfold=self.unfold(x).view(n,c*self.kernel_size*self.kernel_size,h,w)

        low_fre_num=[]
        mid_fre_num=[]
        hig_fre_num=[]
        sparsity=[]
        
        for i in range(n):
            low_fre_num.append(len(torch.nonzero(MaskPredictor[i,0,...])))
            mid_fre_num.append(len(torch.nonzero(MaskPredictor[i,1,...])))
            hig_fre_num.append(len(torch.nonzero(MaskPredictor[i,2,...])))
            sparsity.append((0.0633*low_fre_num[i]+0.5555*mid_fre_num[i]+hig_fre_num[i]) / (h*w))
            
        low_fre_mask=(MaskPredictor[:,0,...]).unsqueeze(1)
        mid_fre_mask=(MaskPredictor[:,1,...]).unsqueeze(1)
        hig_fre_mask=(MaskPredictor[:,2,...]).unsqueeze(1)
        
        unfold_low=unfold * (low_fre_mask.expand_as(unfold))
        unfold_low = unfold_low.view(n,c,self.kernel_size*self.kernel_size,h*w)
        unfold_low = unfold_low.permute(0,1,3,2).view(n,c,h,w,self.kernel_size,self.kernel_size)
        low=torch.einsum('nchwkj,ckj->nchw',unfold_low,low_weight_0)
        low=self.conv_1_low(self.non_act(low))

        unfold_mid=unfold * (mid_fre_mask.expand_as(unfold))
        unfold_mid=unfold_mid.view(n,c*self.kernel_size*self.kernel_size,h*w)
        mid_0=unfold_mid.transpose(1,2).matmul(mid_weight_0.t()).transpose(1,2)
        mid_0=F.fold(mid_0,(h,w),(1,1))
        mid=self.conv_1_mid(self.non_act(mid_0))
        
        unfold_hig=unfold * (hig_fre_mask.expand_as(unfold))
        unfold_hig=unfold_hig.view(n,c*self.kernel_size*self.kernel_size,h*w)
        hig_0=unfold_hig.transpose(1,2).matmul(hig_weight_0.t()).transpose(1,2)
        hig_0=F.fold(hig_0,(h,w),(1,1))
        hig_0=self.non_act(hig_0)
        unfold=self.unfold(hig_0).view(n,c*self.kernel_size*self.kernel_size,h,w)
        unfold_hig=unfold * (hig_fre_mask.expand_as(unfold))
        unfold_hig=unfold_hig.view(n,c*self.kernel_size*self.kernel_size,h*w)
        hig=unfold_hig.transpose(1,2).matmul(hig_weight_1.t()).transpose(1,2)
        hig=F.fold(hig,(h,w),(1,1))
        
        return (x+low+mid+hig), sparsity

class Upsampler(nn.Sequential):
    def __init__(self, 
        scale: int = 2, 
        n_feats: int = 64, 
        bn: bool = False, 
        act: str = 'relu', 
        bias: bool = True):

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

class FADN(nn.Module):
    def __init__(self,
        n_inputs: int = 1,
        n_feats: int = 64,
        kernel_size: int = 3,
        n_resblocks: int = 16,
        bias: bool = True,
        bn: bool = False,
        act: str = 'relu',
        scale: int = 4,
        alpha: float = 0.4,
        upsampling: bool = True
        ):
        super(FADN,self).__init__()
        self.n_resblocks = n_resblocks
        self.alpha = alpha
        # define head module
        m_head = [SpatialPreservedConv(n_inputs, n_feats, kernel_size, bias)]
        self.head = nn.Sequential(*m_head)

        # define body module
        m_body = [DyResBlock(kernel_size, n_feats, n_feats, act) for _ in range(n_resblocks)]
        m_body.append(SpatialPreservedConv(n_feats, n_feats, kernel_size, bias))
        self.body = nn.Sequential(*m_body)

        # define tail module
        self.upsampling = upsampling
        
        if self.upsampling:
            m_tail = [
                Upsampler(scale=scale, n_feats=n_feats, bn=bn, bias=bias, act=act),
                SpatialPreservedConv(n_feats, n_feats, kernel_size, bias)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        n,c,h,w=x.size()
        sparsity_sum=[0]*n
        x = self.head(x)
        res = x
        for i in range(self.n_resblocks):
            res,sparsity=self.body[i](res)
            sparsity_sum = [sparsity_sum[i]+sparsity[i] for i in range(min(len(sparsity),len(sparsity_sum)))]
        sparsity_avg = [(k / (self.n_resblocks)) for k in sparsity_sum]
        res=self.body[self.n_resblocks](res)
        res += x

        if not self.upsampling:
            return res, torch.FloatTensor(sparsity_avg).to(x.device) - self.alpha
        x = self.tail(res)
        return x, torch.FloatTensor(sparsity_avg).to(x.device) - self.alpha
    
if __name__ == "__main__":
    model = FADN(n_inputs=1, 
        n_feats=64, 
        kernel_size=3, 
        n_resblocks=16, 
        bias=True, 
        bn=False, 
        act="relu", 
        scale=2, 
        alpha=0.4,
        upsampling=True
    ).to(torch.device('cuda'))

    x = torch.Tensor(4, 1, 30, 40).to(torch.device('cuda'))
    y, sparsity_avg = model(x)
    print(y.shape)
    # print(sparsity_avg.shape)
    # print((sparsity_avg**2).mean())