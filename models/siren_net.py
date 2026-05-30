import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from einops import rearrange
from functools import (partial, reduce)
from hydra.utils import log


# helpers
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# non-linear activation
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


non_act = {"relu": partial(nn.ReLU),
       "sigmoid": partial(nn.Sigmoid),
       "tanh": partial(nn.Tanh),
       "selu": partial(nn.SELU),
       "softplus": partial(nn.Softplus),
       "gelu": partial(nn.GELU),
       "swish": partial(Swish),
       "elu": partial(nn.ELU)}


# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_weights(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_weights(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "b d -> b () d")

        return self.last_layer(x)


# modulatory feed forward
class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)


class SirenModulatorNet(nn.Module):
    def __init__(self, 
        dim_in: int = 2,
        dim_hidden: int = 256,
        dim_out: int = 1,
        num_layers: int = 4,
        act: str = "relu",
        w0_initial: int = 30,
        latent_dim: int = 116 
    ):
        super().__init__()

        log.info("Model: Siren Modulator Net with 1D conv encoding contidional INR")

        encode_lst = [
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 1874),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 467),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 116)
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            # nn.BatchNorm1d(1, 28)
        ]
        self.encoder = nn.Sequential(*encode_lst)

        self.net = SirenNet(
            dim_in = dim_in,                          # input dimension, ex. 2d coor
            dim_hidden = dim_hidden,                  # hidden dimension
            dim_out = dim_out,                        # output dimension, ex. rgb value
            num_layers = num_layers,                  # number of layers
            final_activation = non_act[act](),        # activation of final layer
            w0_initial = w0_initial                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = dim_hidden,
                num_layers = num_layers
            )

        self._count_params()

    def forward(self, x1, x2):
        """
        Args:
            x1: low resolution (b, 1, h_LR, w_LR)
            x2: coordinates    (b, n_query_pts, 2)
        Returns:
            out: prediction    (b, n_query_pts, 1)
        """
        b = x1.shape[0]

        # b, 1, h_LR, w_LR -> b, latent (116)
        latent = torch.squeeze(self.encoder(x1.view(b, 1, -1)))

        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"
        mods = self.modulator(latent) if modulate else None

        out = self.net(x2, mods)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        log.info(f"Total params: %.2fM" % (c/1000000.0))
        log.info(f"Total params: %.2fk" % (c/1000.0))


if __name__ == '__main__':
    x1 = torch.randn(16, 1, 75, 100)
    x2 = torch.randn(16, 7500, 2)

    model = SirenModulatorNet()
    out = model(x1, x2)

    print(out.shape)