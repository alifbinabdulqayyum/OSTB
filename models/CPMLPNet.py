import torch
import torch.nn.functional as F
import operator

from torch import nn
from functools import (partial, reduce)
from models.FourierEncoding import (BasicEncoding, PositionalEncoding, GaussianEncoding)
from hydra.utils import log


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


class MultiLayerBlock(nn.Module):
    def __init__(self, in_size, out_size, act):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, out_size),
            non_act[act](),
        )

    def forward(self, x):
        return self.layers(x)


class CPMLP(nn.Module):
    def __init__(self, 
        encode: str = None,
        gauss_sigma: float = 0.,
        pos_freq_const: float = 0.,
        pos_freq_num: int = 0,
        input_size: int = 0,
        encoded_size: int = 0,
        hidden_size: list = [],
        output_size: int = 0,
        act: str = None,
    ):
        super().__init__()

        self.net = []
        log.info("Model: CPMLP")
        
        # encode layer
        if encode: 
            if encode == 'Gaussian':
                encode_layer = GaussianEncoding(gauss_sigma, input_size, encoded_size)
            elif encode == 'Basic':
                encode_layer = BasicEncoding()
            elif encode == 'Position':
                encode_layer = PositionalEncoding(pos_freq_const, pos_freq_num)
            log.info(f"Encoding method: {encode}")
        else:
            encode_layer = nn.Linear(input_size, encoded_size*2)
            log.info("Encoding method: None")
        self.net.append(encode_layer)
        
        # hidden layer
        if encode == 'Gaussian':
            for in_s, out_s in zip([encoded_size*2]+hidden_size[:-1], hidden_size):
                self.net.append(MultiLayerBlock(in_s, out_s, act))
        elif encode == 'Basic':
            for in_s, out_s in zip([4]+hidden_size[:-1], hidden_size):
                self.net.append(MultiLayerBlock(in_s, out_s, act))            
        elif encode == 'Position':
            for in_s, out_s in zip([4*pos_freq_num]+hidden_size[:-1], hidden_size):
                self.net.append(MultiLayerBlock(in_s, out_s, act))   
        else:
            for in_s, out_s in zip([encoded_size*2]+hidden_size[:-1], hidden_size):
                self.net.append(MultiLayerBlock(in_s, out_s, act))

        # last layer
        self.net.append(nn.Linear(hidden_size[-1], output_size))










        self.net = nn.Sequential(*self.net)

        self._count_params()

    def forward(self, x):
        out = self.net(x)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        log.info(f"Total params: %.2fM" % (c/1000000.0))
        log.info(f"Total params: %.2fk" % (c/1000.0))