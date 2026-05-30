import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from functools import (partial, reduce)
from models.FourierEncoding import (BasicEncoding, PositionalEncoding, GaussianEncoding)
from models.siren_net import (SirenNet, Modulator)
from hydra.utils import log


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


class Resblock(nn.Module):
    def __init__(self, in_size, hidden_size, act, dropout_rate, res = True):
        """ 
        Init method.
        """
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_size),
            non_act[act](),
            nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        ) 
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(hidden_size),
            non_act[act](),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        ) 
        self.res = res
        
    def forward(self, x):
        """
        Forward pass of the function.
        """
        out = self.layer1(x)
        if self.res:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out


class ZtoThetaEncoder(nn.Module):
    def __init__(self,
        act: str = None, 
        dropout_rate: float = 0.,
    ):
        """
        Init method.
        """
        super(ZtoThetaEncoder, self).__init__()

        #----------------------------------------------------------------------
        # experiment 1
        #----------------------------------------------------------------------
        self.input_layer  = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.downsampling = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.block = Resblock(64, 64, act, dropout_rate)
        self.output_layer = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3, bias=False)

        # #----------------------------------------------------------------------
        # # experiment 2
        # #----------------------------------------------------------------------
        # self.input_layer   = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=False)
        # self.downsampling1 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2, bias=False)
        # self.downsampling2 = nn.Conv2d(16, 8, kernel_size=7, stride=2, padding=(2, 3), bias=False)
        # self.output_layer  = nn.Conv2d(25, 33, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        Forward pass of the function.
        """
        # ----------------------------------------------------------------------
        # experiment 1
        # ----------------------------------------------------------------------
        # b, 1, 75, 100 -> b, 32, 75, 100
        out = self.input_layer(x)

        # b, 32, 75, 100 -> b, 64, 38, 50
        out = self.downsampling(out)
        
        # b, 64, 38, 50 -> b, 64, 38, 50
        out = self.block(out)

        # b, 64, 38, 50 -> b, 32, 38, 50
        out = self.output_layer(out)
        return out

        # #----------------------------------------------------------------------
        # # experiment 2
        # #----------------------------------------------------------------------
        # # b, 1, 75, 100 -> b, 16, 75, 100
        # out = self.input_layer(x)

        # # b, 16, 75, 100 -> b, 8, 18, 25
        # out = self.downsampling2(self.downsampling1(out))

        # # b, 8, 18, 25 -> b, 25, 8, 18
        # out = torch.permute(out, (0, 3, 1, 2))

        # # b, 25, 8, 18 -> b, 33, 8, 18
        # out = self.output_layer(out)
        # return out


class ZtoThetaNet(nn.Module):
    def __init__(self, 
        encode: str = None,
        gauss_sigma: float = 15.,
        pos_freq_const: float = 50.,
        pos_freq_num: int = 80,
        input_size: int = 2,
        encoded_size: int = 160,
        output_size: int = 1,
        act: str = None,
        dropout_rate: float = 0.,
    ):
        super().__init__()

        log.info("Model: z to theta mapping contidional INR")
        
        # encode layer
        if encode: 
            if encode == "Gaussian":
                self.fourier_layer = GaussianEncoding(gauss_sigma, input_size, encoded_size)
            elif encode == "Basic":
                self.fourier_layer = BasicEncoding()
            elif encode == "Position":
                self.fourier_layer = PositionalEncoding(pos_freq_const, pos_freq_num)
            log.info(f"Encoding method: {encode}")
        else:
            self.fourier_layer = nn.Linear(input_size, encoded_size*2)
            log.info("Encoding method: None")
        
        self.encoder = ZtoThetaEncoder(act, dropout_rate)

        self.act1 = non_act[act]()
        # self.act2 = non_act[act]()
        # self.act3 = non_act[act]()
        # self.act4 = non_act[act]()

        self.out_layer = nn.Linear(190, output_size)

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
        
        #----------------------------------------------------------------------
        # experiment 1
        #----------------------------------------------------------------------
        # b, 1, h_LR, w_LR -> b, 320, 190
        W = torch.reshape(self.encoder(x1), (b, 320, 190))
        
        # b, n_query_pts, 2 -> b, n_query_pts, 320
        x2 = self.fourier_layer(x2)

        # b, n_query_pts, 190
        out = torch.einsum("bij, bjk -> bik", x2, W)
        out = self.act1(out)

        # # add more layers and reuse the encoded weights decrease the performance
        # # b, n_query_pts, 320
        # out = torch.einsum("bij, bjk -> bik", out, torch.permute(W, (0, 2, 1)))
        # out = self.act2(out)

        # # b, n_query_pts, 190
        # out = torch.einsum("bij, bjk -> bik", out, W)
        # out = self.act3(out)

        return self.out_layer(out)

        # #----------------------------------------------------------------------
        # # experiment 2
        # #----------------------------------------------------------------------
        # # b, 1, h_LR, w_LR -> b, 33, 8, 18
        # idx1 = 0
        # idx2 = 2*48
        # idx3 = 2*48+48*48
        # idx4 = 2*48+48*48+48*48

        # W  = torch.reshape(self.encoder(x1), (b, 33*8*18))
        # W1 = torch.reshape(W[:,idx1:idx2], (b, 2, 48))  # b, 2, 48
        # W2 = torch.reshape(W[:,idx2:idx3], (b, 48, 48)) # b, 48, 48
        # W3 = torch.reshape(W[:,idx3:idx4], (b, 48, 48)) # b, 48, 48
        # W4 = torch.reshape(W[:,idx4:], (b, 48, 1))      # b, 48, 1
        
        # # b, n_query_pts, 2
        # out = x2
        # out = torch.einsum("bij, bjk -> bik", out, W1)
        # out = self.act1(out)
        # out = torch.einsum("bij, bjk -> bik", out, W2)
        # out = self.act2(out)
        # out = torch.einsum("bij, bjk -> bik", out, W3)
        # out = self.act3(out)
        # out = torch.einsum("bij, bjk -> bik", out, W4)
        # out = self.act4(out)
        # return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        log.info(f"Total params: %.2fM" % (c/1000000.0))
        log.info(f"Total params: %.2fk" % (c/1000.0))


class MLPNet(nn.Module):
    def __init__(self, 
        in_dim: int = 64, 
        out_dim: int = 1, 
        hidden_size: list = [256, 256, 256, 256], 
        act: str = None,
        ):
        super().__init__()

        layers = []
        lastv = in_dim
        for hidden in hidden_size:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(non_act[act]())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class ZcatXNet(nn.Module):
    def __init__(self, 
        input_size: int = 2,
        hidden_size: list = [256, 256, 256, 256],
        output_size: int = 1,
        act: str = None,
    ):
        super().__init__()

        log.info("Model: x, z concatenation contidional INR")

        #----------------------------------------------------------------------
        # experiment 1
        #----------------------------------------------------------------------
        encode_lst = [
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 1874),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 467),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 116),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=4),
            nn.BatchNorm1d(1, 28)
        ]
        self.encoder = nn.Sequential(*encode_lst)

        # self.mlp = MLPNet(in_dim=28+input_size, out_dim=output_size, hidden_size=hidden_size, act=act)

        #----------------------------------------------------------------------
        # experiment 2
        #----------------------------------------------------------------------

        self.siren = SirenNet(
            dim_in = 2,                        # input dimension, ex. 2d coor
            dim_hidden = 256,                  # hidden dimension
            dim_out = 3,                       # output dimension, ex. rgb value
            num_layers = 5,                    # number of layers
            # final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        self.modulator = Modulator(
            dim_in = latent_dim,
            dim_hidden = net.dim_hidden,
            num_layers = net.num_layers
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
        n_query_pts = x2.shape[1]
        
        #----------------------------------------------------------------------
        # experiment 1
        #----------------------------------------------------------------------
        # b, 1, h_LR, w_LR -> b, 1, latent (28) -> b, n_query_pts, latent (28)
        x1 = self.encoder(x1.view(b, 1, -1)).repeat(1, n_query_pts, 1)
        
        # b, n_query_pts, 2 -> b, n_query_pts, 2 + latent (28)
        out = torch.cat((x1, x2), dim=2)

        # b, n_query_pts, 2 + latent (28) -> b, n_query_pts, 1
        return self.mlp(out)



    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        log.info(f"Total params: %.2fM" % (c/1000000.0))
        log.info(f"Total params: %.2fk" % (c/1000.0))