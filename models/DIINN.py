import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import math
import pdb

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


def make_rdn(n_colors=1, G0=64, RDNkSize=3, RDNconfig='B',
             scale=2, no_upsampling=True):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = n_colors #1 #3
    return RDN(args)

class DIINN(nn.Module):
    def __init__(self,
                 in_feat, mode, init_q):
        super().__init__()
        
        self.encoder = make_rdn(n_colors=in_feat)
        self.decoder = ImplicitDecoder(mode=mode, init_q=init_q)

    def forward(self, x, size, bsize=None):
        x = self.encoder(x)
        x = self.decoder(x, size, bsize)
        return x 

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

def patch_norm_2d(x, kernel_size=3):
    #B, C, H, W = x.shape
    #var, mean = torch.var_mean(F.unfold(x, kernel_size=kernel_size, padding=padding).view(B, C,kernel_size**2, H, W), dim=2, keepdim=False)
    #return (x - mean) / torch.sqrt(var + 1e-6)
    mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
    mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
    var = mean_sq - mean**2
    return (x-mean)/(var + 1e-6)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1, hidden_dims=[256, 256, 256, 256], mode=1, init_q=False):
        super().__init__()

        self.mode = mode
        self.init_q = init_q

        last_dim_K = in_channels * 9
        
        if self.init_q:
            self.first_layer = nn.Sequential(nn.Conv2d(3, in_channels * 9, 1),
                                            SineAct())
            last_dim_Q = in_channels * 9
        else:
            last_dim_Q = 3

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        if self.mode == 1:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim
                last_dim_Q = hidden_dim
        elif self.mode == 2:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim
        elif self.mode == 3:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim
        elif self.mode == 4:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim 
        if self.mode == 4:
            self.last_layer = nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1, padding_mode='reflect')
        else:
            self.last_layer = nn.Conv2d(hidden_dims[-1], out_channels, 1)

    def _make_pos_encoding(self, x, size): 
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up, indexing='ij'), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact')) #important! mode='nearest' gives inconsistent results
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, syn_inp):
        if self.init_q:
            syn_inp = self.first_layer(syn_inp)
            x = syn_inp * x
        if self.mode == 1:
            k = self.K[0](x)
            q = k*self.Q[0](syn_inp)        
            for i in range(1, len(self.K)):
                k = self.K[i](k)
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 2:
            k = self.K[0](x)
            q = k*self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([k,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 3:
            k = self.K[0](x)
            q = k*self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 4:
            k = self.K[0](x)
            q = k*self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q

    # def batched_step(self, x, syn_inp, bsize):
    #     with torch.no_grad():
    #         h, w = syn_inp.shape[-2:]
    #         ql = 0
    #         preds = []
    #         while ql < w:
    #             qr = min(ql + bsize//h, w)
    #             pred = self.step(x[:, :, :, ql: qr], syn_inp[:, :, :, ql: qr])
    #             preds.append(pred)
    #             ql = qr
    #         pred = torch.cat(preds, dim=-1)
    #     return pred


    def forward(self, x, size, bsize=None):
        B, C, H_in, W_in = x.shape
        rel_coord = self._make_pos_encoding(x, size).expand(B, -1, *size) #2
        ratio = x.new_tensor([(H_in*W_in)/(size[0]*size[1])]).view(1, -1, 1, 1).expand(B, -1, *size) #2
        syn_inp = torch.cat([rel_coord, ratio], dim=1)          
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='nearest-exact')
        # if bsize is None:
        #     pred = self.step(x, syn_inp)
        # else:
        #     pred = self.batched_step(x, syn_inp, bsize)
        return self.step(x, syn_inp) #pred
