# import torch
# from torch import nn

# class Latent_Modality_Classifier(nn.Module):
#     def __init__(
#         self,
#         latent_dim:int,
#         hidden_dim:int
#     ):
#         super().__init__()
#         self.hidden = nn.Linear(latent_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         x = self.relu(self.hidden(x))
#         x = self.sigmoid(self.output(x))
#         return x
    

### YOUR CODE HERE
# import tensorflow as tf
"""This script defines the network.
"""
import torch.nn as nn
import torch

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

class Latent_Modality_Classifier(nn.Module):
    def __init__(
        self, 
        latent_dim:int, 
        n_channels:int,
        depth:int, 
        kernel_size:tuple, 
        patch_size:tuple, 
        dilation:int, 
        n_classes:int
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.dilation = dilation
        self.n_classes = n_classes
        
        self.patch_block = PatchConv(input_dim=self.latent_dim, channel_depth=self.n_channels, patch_size=self.patch_size)
        self.conv_mixer_block = ConvMixer(dim=self.n_channels, depth=self.depth, kernel_size=self.kernel_size, dilation=self.dilation)
        
        self.classification_layer = nn.Linear(self.n_channels, self.n_classes)
        
    def forward(self, x):
        x = self.patch_block(x)
        x = self.conv_mixer_block(x)
        
        x = torch.mean(x, axis = (-2, -1))
        
        x = self.classification_layer(x)
        
        return torch.sigmoid(x)

class PatchConv(nn.Module):
    def __init__(self, channel_depth, patch_size, input_dim):
        super().__init__()
        self.block = self.Patch_block(input_dim, channel_depth, patch_size)
    
    def forward(self, x):
        return self.block(x)
    
    def Patch_block(self, input_dim, channel_depth, patch_size):
        return nn.Sequential(nn.Conv2d(input_dim, channel_depth, kernel_size=patch_size, stride=patch_size),
                              nn.GELU(),
                              nn.BatchNorm2d(channel_depth))
    
class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size, dilation):
        super().__init__()
        self.block = self.Mixer_block(dim, depth, kernel_size, dilation)
    
    def forward(self, x):
        return self.block(x)
    
    def Mixer_block(self, dim, depth, kernel_size, dilation):
        return nn.Sequential(*[nn.Sequential(
                              Residual(nn.Sequential(
                              nn.Conv2d(dim, dim, kernel_size, groups=dim, dilation=dilation, padding=(dilation * (kernel_size - 1))//2),
                              nn.GELU(),
                              nn.BatchNorm2d(dim)
                              )),
                              nn.Conv2d(dim, dim, kernel_size=1),
                              nn.GELU(),
                              nn.BatchNorm2d(dim)
                              ) for i in range(depth)])
        
        
            
    
### END CODE HERE

if __name__ == "__main__":
    model = Latent_Modality_Classifier(latent_dim=1, n_channels=64, depth=8, kernel_size=3, patch_size=2, dilation=1, n_classes=4)
    x = torch.Tensor(16, 1, 30, 40)
    y = model(x)
    print(y.shape)
