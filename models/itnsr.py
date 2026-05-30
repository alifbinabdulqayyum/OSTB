import torch
import torch.nn as nn
import torch.nn.functional as F
# import math
# import models
# from models import register
# from utils import make_coord
from einops import repeat
import math

################

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=1,
        hidden_layer=[64],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        act=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        layers_hidden = [in_dim] + hidden_layer + [out_dim]

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=act,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

################

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
    def __init__(self):
        super(RDN, self).__init__()
        G0 = 64
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }['B']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(1, G0, kSize, padding=(kSize-1)//2, stride=1)
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
        
        self.out_dim = G0
        

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return x

################
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='sine'):
        super().__init__()
        # pdb.set_trace()
        self.act = nn.GELU()
        
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)
################

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class ITNSR(nn.Module):

    def __init__(self, 
                 local_ensemble=True, 
                 feat_unfold=True, 
                 scale_token=True,
                 wKAN:bool=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.scale_token = scale_token

        self.encoder = RDN()
        if wKAN:
            self.imnet = KAN(in_dim=4, out_dim=self.encoder.out_dim*9)
        else:
            self.imnet = MLP(in_dim=4, out_dim=self.encoder.out_dim*9, hidden_list=[256,256,256,256])
        
        
        # if embedding_coord is not None:
        #     self.embedding_q = models.make(embedding_coord)
        #     self.embedding_s = models.make(embedding_scale)
        # else:
        #     self.embedding_q = None
        #     self.embedding_s = None

        self.embedding_q = None
        self.embedding_s = None

        if local_ensemble:
            # w = {
            #     'name': 'mlp',
            #     'args': {
            #         'in_dim': 4,
            #         'out_dim': 1,
            #         'hidden_list': [256],
            #         'act': 'gelu'
            #     }
            # }
            # self.Weight = models.make(w)
            if wKAN:
                self.Weight = KAN(in_dim=4, out_dim=1, hidden_layer=[32])
            else:
                self.Weight = MLP(in_dim=4, out_dim=1, hidden_list=[256])

            # score = {
            #     'name': 'mlp',
            #     'args': {
            #         'in_dim': 2,
            #         'out_dim': 1,
            #         'hidden_list': [256],
            #         'act': 'gelu'
            #     }
            # }
            # self.Score = models.make(score)
            if wKAN:
                self.Score = KAN(in_dim=2, out_dim=1, hidden_layer=[32])
            else:
                self.Score = MLP(in_dim=2, out_dim=1, hidden_list=[256])
 

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, scale=None):

        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        # K
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device)  \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # enhance local features
        if self.local_ensemble:
            # v_lst = [(-1,-1),(-1,0),(-1,1),(0, -1), (0, 0), (0, 1),(1, -1),(1, 0),(1,1)]#
            v_lst = [(i,j) for i in range(-1, 2, 2) for j in range(-1, 2, 2)]
            # v_lst = [(-1,0), (1,0), (0,1), (0, -1), (-2,-2), (-2, 2), (2, -2), (2, 2)]
            eps_shift = 1e-6
            preds = []
            for v in v_lst:
                vx = v[0]
                vy = v[1]
                # project to LR field 
                tx = ((feat.shape[-2] - 1) / (1 - scale[:,0,0])).view(feat.shape[0],  1)
                ty = ((feat.shape[-1] - 1) / (1 - scale[:,0,1])).view(feat.shape[0],  1)
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0
                bs, q = coord.shape[:2]
                coord_ = coord.clone()

                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                #Interpolate K to HR resolution  
                value = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #Interpolate K to HR resolution 
                coord_k = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #calculate relation of Q-K
                if self.embedding_q:
                    Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                    K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                    rel = Q - K
                    
                    rel[:, 0] *= feat.shape[-2]
                    rel[:, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        # scale = scale.view(bs*q,-1)
                        scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                        inp = torch.cat([inp, scale_], dim=-1)

                else:
                    Q, K = coord, coord_k
                    rel = Q - K
                    rel[:, :, 0] *= feat.shape[-2]
                    rel[:, :, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        inp = torch.cat([inp, scale_], dim=-1)

                score = self.Score(rel.view(bs * q, -1)).view(bs, q, -1)
                
                weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], -1)
                pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
                
                pred +=score
                preds.append(pred)

            preds = torch.stack(preds,dim=-1)

            ret = self.Weight(preds.view(bs*q, -1)).view(bs, q, -1)
        else:
            #V
            bs, q = coord.shape[:2]
            value = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            #K
            coord_k = F.grid_sample(
                feat_coord, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            if self.embedding_q:
                Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                rel = Q - K
                
                rel[:, 0] *= feat.shape[-2]
                rel[:, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    # scale = scale.view(bs*q,-1)
                    scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                    inp = torch.cat([inp, scale_], dim=-1)

            else:
                Q, K = coord, coord_k
                rel = Q - K
                rel[:, :, 0] *= feat.shape[-2]
                rel[:, :, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, scale_], dim=-1)
            
            
            weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
            pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
            ret = pred
        
        return ret

    def forward(self, inp, coord, scale):

        self.gen_feat(inp)
        return self.query_rgb(coord, scale)




