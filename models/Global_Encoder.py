import torch
import torch.nn as nn
from functools import (partial, reduce)
import operator

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

# New weights with accuracy 80.858%
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# model = resnet50(pretrained = True)
# model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = list(model.children())[:-1]
model = nn.Sequential(*model)

class SpatialPreservedConv(nn.Conv2d):
    """
    To keep spatial size of input same as output, this code only work for stride step = 1, and kernel size is odd number.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        if kernel_size % 2 == 0:
            NotImplementedError("When stride is 1, this only works for odd kernel size.")

        super(SpatialPreservedConv, self).__init__(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class Global_Encoder(nn.Module):
    def __init__(self, 
        n_inps: int = 2,
        n_feats: int = 3, 
        kernel_size: int = 3,
        out_feats: int = 64,
        bias: bool = True, 
        ):
        super(Global_Encoder, self).__init__()

        m = []
        
        m.append(SpatialPreservedConv(n_inps, n_feats, kernel_size, bias=bias))
        # m.append(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))
        m.extend(list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[4:-1])
        # m.append(resnet50(pretrained = True))
        m.append(nn.Flatten())
        m.append(nn.Linear(in_features=512,out_features=out_feats))
        
        self.net = nn.Sequential(*m)

    def forward(self, x):
        # b = x.shape[0]
        # return self.net(x).view(b, -1)
        return self.net(x)
    
    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        print(f"Total params: %.2fM" % (c/1000000.0))
        print(f"Total params: %.2fk" % (c/1000.0))

if __name__ == "__main__":
    device = torch.device('cuda')
    model = Global_Encoder(n_inps=1, n_feats=64, kernel_size=3, bias=True).to(device)
    x = torch.Tensor(1, 1, 15, 20).to(device)
    y = model(x)
    print(y.shape)
    # print(y.device)
    model._count_params()
