import torch
import torch.nn as nn
from functools import (partial, reduce)
import operator
import numpy as np

class StandardBlock(nn.Module):
    def __init__(self,
                 num_in_channels:int,
                 num_out_channels:int,
                 depth:int=2,
                 zero_init:bool=False,
                 normalization:str="instance"):
        super(StandardBlock, self).__init__()

        # conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

        seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        for i in range(depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)

            if i == 0:
                current_in_channels = num_in_channels
            if i == depth-1:
                current_out_channels = num_out_channels

            seq.append(
                nn.Conv2d(
                    current_in_channels,
                    current_out_channels,
                    3,
                    padding=1,
                    bias=True))
            torch.nn.init.kaiming_uniform_(seq[-1].weight,
                                           a=0.01,
                                           mode='fan_out',
                                           nonlinearity='leaky_relu')

            if normalization == "instance":
                seq.append(nn.InstanceNorm2d(current_out_channels, affine=True))

            elif normalization == "group":
                seq.append(
                    nn.GroupNorm(
                        np.min(1, current_out_channels // 8),
                        current_out_channels,
                        affine=True)
                )

            elif normalization == "batch":
                seq.append(nn.BatchNorm2d(current_out_channels, eps=1e-3))

            else:
                # print("No normalization specified.")
                pass

            seq.append(nn.LeakyReLU(inplace=True))


        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)

        self.F = nn.Sequential(*seq)

    def forward(self, x):
        x = self.F(x)
        return x

class LatentEncoder(nn.Module):
    def __init__(self,
                in_dim:int=1,
                latent_dim:int=1,
                architecture:list=[2,2,2,2],
                down_filters:list=[32,64,128],
                zero_init=False):
        super(LatentEncoder, self).__init__()
        self.input_channels = 1
        self.architecture = architecture
        self.n_levels = len(self.architecture)
        
        down_filters.append(latent_dim)

        self.module_L = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()        
        
        # Left side of the U-Net
        for i in range(self.n_levels):
            module = nn.ModuleList()
            if i < self.n_levels-1:
                self.downsampling_layers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
            
            depth = architecture[i]    
            for j in range(depth):
                if i == 0 and j == 0:
                    in_channels = self.input_channels
                else:
                    in_channels = down_filters[i-1]
                    
                if j == depth-1:
                    out_channels = down_filters[i]
                else:
                    out_channels = down_filters[i-1]

                module.append(
                    StandardBlock(in_channels, out_channels, zero_init=zero_init)
                )
            self.module_L.append(nn.Sequential(*module))
                
    def forward(self, x):        
        # Left side
        for i in range(self.n_levels):
            x = self.module_L[i](x)
            
            # Downsampling L
            if i < self.n_levels-1:
                x = self.downsampling_layers[i](x)

        return x
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

if __name__ == "__main__":
    device = torch.device('cuda')
    model = LatentEncoder(in_dim=1, latent_dim=1).to(device)
    x = torch.Tensor(1, 1, 120, 160).to(device)
    y = model(x)
    print(y.shape)
    # print(y.device)
    model._count_params()
