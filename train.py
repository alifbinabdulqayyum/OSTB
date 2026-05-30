import sys
# from datamodules.data_set import load_3D_data, get_coords, ZscoreStandardizer_3D, MinMaxStandardizer, LIIFDataset, \
#     LIIFVizDataset, IDMDataset, IDMDatasetv2, LIIFDatasetv2
from datamodules.data_set import get_coords, LIIFDatasetv2
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
# from models.LatentEncoder import LatentEncoder
# from torchvision.transforms.functional import InterpolationMode
# import random
import torch.nn as nn
from einops import rearrange
# from models.sr3_modules import diffusion, unet, edsr, mlp
from models.sr3_modules import edsr
from models.INRNet import EncoderINNwithGaussianAttention, DecoderINNwithGaussianAttention, EncoderINNwithAttention, DecoderINNwithAttention, EncoderINN, DecoderINN
from models.FourierEncoding import PositionalEncoding
# from models.sr3_modules.MMAttentionEncoder import MMAttentionEncoderv2, MMAttentionEncoderv1
from models.INRNet import INR, INRwithGaussianAttention
from models.KAN import KAN
from inspect import isfunction
import numpy as np
import torch
import torch.nn.functional as F

import functools
from torch.nn import init
# import math
from tqdm.contrib import tzip

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
    
def exists(x):
    return x is not None
    
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

####################
# 3D Coordinates
####################
EPSILON = 1e-6
def get_3D_coords(xy_shape, 
                  z:float=0.9999,
                  ranges=None, 
                  flatten=True):
    """ 
    Make coordinates at grid centers.
    Args:
        shape   (list): image size [H, W]
        ranges  (list): grid boundaries [[left, right], [down, up]] 
        flatten (bool): True
    Returns:
        coords  (torch.tensor): H * W, 2
    """
    # determine the center of each grid
    coord_seqs = []
    for i, n in enumerate(xy_shape):
        if ranges is None:
            v0, v1 = -1 + EPSILON, 1 - EPSILON
        else:
            v0, v1 = ranges[i]
        # r = (v1 - v0) / (2 * n)
        r = (v1 - v0) / (2 * (n - 1))
        # seq = v0 + r + (2 * r) * torch.arange(n).float()
        seq = v0 + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    # make mesh
    coords = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    z = z*torch.ones(*coords.shape[:-1],1)
    coords = torch.cat([coords,z],dim=-1)
    if flatten:
        coords = rearrange(coords,
                          'h w c -> (h w) c')
    return coords

####################
# evaluation
####################

def look_up_feature(
    coordinate: torch.Tensor, 
    feature: torch.Tensor, 
    feat_coord: torch.Tensor
):
    '''
    Args:
        coordinate (torch.tensor) - (b, n_query_pts, 2)
        feature    (torch.tensor) - (b, n_feature, H, W)
    Returns:
        feature      (torch.tensor) - (b, n_query_pts, n_feature)
        f_coordinate (torch.tensor) - (b, n_query_pts, 2)
    '''
    feature = F.grid_sample(
                feature, 
                coordinate.flip(-1).unsqueeze(1), # (b, 1, n_feature, 2)
                mode='nearest', 
                align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, n_feature)

    f_coordinate = F.grid_sample(
                    feat_coord, 
                    coordinate.flip(-1).unsqueeze(1),
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1) # (b, n_feature, 2)

    return feature, f_coordinate

####################
# LIIF decoder
####################

def pred_liif(liif_decoder,
              feature,
              coord,
              cell):

    b, q = coord.shape[:2]
    
    feature_data = F.unfold(feature, 3, padding=1).view(
        feature.shape[0], 
        feature.shape[1] * 9, 
        feature.shape[2], 
        feature.shape[3]
    )
    
    # field radius (global: [-1, 1])
    rx = 2 / feature_data.shape[-2] / 2
    ry = 2 / feature_data.shape[-1] / 2

    feat_coord = get_coords(feature_data.shape[-2:], flatten=False).type_as(feature_data) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feature_data.shape[0], 2, *feature_data.shape[-2:]) # (b, 2, H, W)
    
    vx_lst = [-1, 1]
    vy_lst = [-1, 1]
    eps_shift = 1e-6
    
    self_preds = []
    areas = []

    for vx in vx_lst:
        for vy in vy_lst:
            #########
            coord_ = coord.clone() # (b, n_query_pts, 2)
            # left-top, left-down, right-top, right-down move one radius.
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_feat, q_coord = look_up_feature(coord_, feature_data, feat_coord)

            relative_offset = coord - q_coord
            relative_offset[:, :, 0] *= feature_data.shape[-2]
            relative_offset[:, :, 1] *= feature_data.shape[-1]

            area = torch.abs(relative_offset[:, :, 0] * relative_offset[:, :, 1])

            inp = torch.cat([q_feat, relative_offset], dim=-1)

            decoded_cell = 2 / (cell.unsqueeze(1).repeat(1, coord.shape[1], 1))

            inp = torch.cat([inp, decoded_cell], dim=-1)

            self_pred = liif_decoder(inp.view(b * q, -1)).view(b, q, -1)

            self_preds.append(self_pred)

            areas.append(area + 1e-9)
            
            #########
            
    tot_area = torch.stack(areas).sum(dim=0)
    
    t = areas[0]; areas[0] = areas[3]; areas[3] = t
    t = areas[1]; areas[1] = areas[2]; areas[2] = t
    
    self_ret = 0
    for pred, area in zip(self_preds, areas):
        self_ret = self_ret + pred * (area / tot_area).unsqueeze(-1)

    return self_ret

####################
# evaluation
####################

@torch.no_grad()
def eval_model(
        inn_encoder,
        inn_decoder,
        feature_decoder,
        liif_decoder,
        val_dataloader,
        device,
        recon_loss,
        h_in_tensor:torch.Tensor,
        h_out_tensor:torch.Tensor,
        single_sample_eval:bool=True,
    ):

    inn_encoder.eval()
    inn_decoder.eval()
    feature_decoder.eval()
    liif_decoder.eval()

    print(' '*90, end='\r', file=sys.stderr)

    for batch in val_dataloader:
        coord, cell = batch["grid_HR"].to(device), batch["cell"].to(device)
        d_img_LR = batch["d_img_LR"]

        [b, m, _, _] = d_img_LR.shape

        d_img_LR = rearrange(d_img_LR, 
                        'b m h w -> m b h w', 
                        w=d_img_LR.shape[-1],
                        h=d_img_LR.shape[-2],
                        m=d_img_LR.shape[-3],
                        b=d_img_LR.shape[0])

        c_img_HR = batch['c_img_HR']
        c_img_HR = rearrange(c_img_HR,
                             'b m q c -> m b q c',
                             b=c_img_HR.shape[0],
                             m=c_img_HR.shape[1])
        
        num_evals = 0

        for mod_in_idx, h_in in enumerate(h_in_tensor):
            d_img_HR_in = d_img_LR[mod_in_idx][:,None,:,:].to(device)
            for mod_out_idx, h_out in enumerate(h_out_tensor):      
                d_img_LR_hat = inn_encoder(
                    d_img_HR_in, 
                    h_in.unsqueeze(0).repeat(b,1,1,1).to(device)
                )

                d_img_LR_hat = inn_decoder(
                    d_img_LR_hat, 
                    h_out.unsqueeze(0).repeat(b,1,1,1).to(device)
                )

                d_img_HR_hat = feature_decoder(d_img_LR_hat, shape=None)

                c_img_HR_target = c_img_HR[mod_out_idx].to(device)

                c_img_HR_hat = pred_liif(liif_decoder,
                    d_img_HR_hat,
                    coord,
                    cell)

                num_evals += 1

                loss = recon_loss(c_img_HR_hat, c_img_HR_target)

                template = '# [{}/{}]: Loss={:.5f}'
                line = template.format(num_evals, 16, loss)
                print(line, end='\n', file=sys.stderr)   

            print(' '*90, end='\r', file=sys.stderr)

        if single_sample_eval:break
    
    inn_encoder.train()
    inn_decoder.train()
    feature_decoder.train()
    liif_decoder.train()

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
#     logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

import argparse
parser = argparse.ArgumentParser('Train Multi-Modality Super Resolution with LIIF')
parser.add_argument('--attention', type=str, choices=['no-attention', 'regular-attention', 'gaussian-attention'], default='gaussian-attention')
parser.add_argument('--file-prefix', type=str, choices=['ua', 'va'], default='ua')
parser.add_argument('--num-gaussians', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-h', type=int, default=600)
parser.add_argument('--max-w', type=int, default=800)
parser.add_argument('--dh-HR', type=int, default=120)
parser.add_argument('--dw-HR', type=int, default=160)
parser.add_argument('--n-query-points', type=int, default=16384)
parser.add_argument('--max-scale', type=float, default=5.0)
parser.add_argument('--max-val', type=float, default=50.0)
parser.add_argument('--num-opts', type=int, default=0)
parser.add_argument('--max-opts', type=int, default=7500)
parser.add_argument('--crop-h', type=int, default=750)
parser.add_argument('--crop-w', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=25.0)
parser.add_argument('--m', type=int, default=30)
parser.add_argument('--min-lr', type=float, default=1e-5)
parser.add_argument('--max-lr', type=float, default=1e-4)
parser.add_argument('--save-interval', type=int, default=1500)
parser.add_argument('--eval-interval', type=int, default=1500)
parser.add_argument('--mod-depth', type=int, default=8)
parser.add_argument('--region', type=str, default=None, choices=['A', 'B', 'C', 'D'])
parser.add_argument('--skip-height', type=int, default=None, choices=[10, 60, 160, 200])

args = parser.parse_args()
num_gaussians = args.num_gaussians 

data_dir = './data'

import os
if args.attention == 'gaussian-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-gaussian{}-attention-models-3dinn-only-mod-depth-{}".format(num_gaussians, args.mod_depth), args.file_prefix)
elif args.attention == 'regular-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-attention-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
elif args.attention == 'no-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
else:
    raise NotImplementedError("Not Available INN method")

# if args.attention == 'gaussian-attention':
#     model_save_dir = os.path.join("saved-liif-wINN-gaussian{}-attention-models-3dinn-only-mod-depth-{}".format(num_gaussians, args.mod_depth), args.file_prefix)
# elif args.attention == 'regular-attention':
#     model_save_dir = os.path.join("saved-liif-wINN-attention-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
# elif args.attention == 'no-attention':
#     model_save_dir = os.path.join("saved-liif-wINN-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
# else:
#     raise NotImplementedError("Not Available INN method")

if args.region is not None:
    model_save_dir = os.path.join(model_save_dir, args.region)

if args.skip_height is not None:
    model_save_dir = os.path.join(model_save_dir, "skip_height_{}".format(args.skip_height))

os.makedirs(model_save_dir, exist_ok=True)

batch_size=args.batch_size

max_h, max_w = args.max_h, args.max_w #600, 800
dh_HR, dw_HR = args.dh_HR, args.dw_HR #120, 160

n_query_points = args.n_query_points #16384
max_scale = args.max_scale #5.0

max_val = args.max_val #50.0

num_opts = args.num_opts #0
max_opts = args.max_opts #60000

crop_h = args.crop_h
crop_w = args.crop_w

import json
with open('idx-list.json', 'r') as f:
    idx_dict = json.load(f)

train_dataset = LIIFDatasetv2(
    data_dir=[os.path.join(data_dir, "wind-{}m/{}".format(10, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(60, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(160, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(200, args.file_prefix))],
    file_prefix=[args.file_prefix, args.file_prefix, args.file_prefix, args.file_prefix],
    # train_frac = 0.8,
    # train = True,
    idx_list=idx_dict['train'],
    max_h=max_h,
    max_w=max_w,
    dh_HR=dh_HR,
    dw_HR=dw_HR,
    crop_h=crop_h,
    crop_w=crop_w,
    dim_reduction_step=3,
    dim_reduction_method='NN', #'bicubic',
    crop=True,
    full=False,
    max_val=max_val,
    n_query_points=n_query_points,
    max_scale=max_scale,
    region=args.region,
)

train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              num_workers=8, 
                              pin_memory=True, 
                              shuffle=True)

val_dataset = LIIFDatasetv2(
    data_dir=[os.path.join(data_dir, "wind-{}m/{}".format(10, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(60, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(160, args.file_prefix)),
              os.path.join(data_dir, "wind-{}m/{}".format(200, args.file_prefix))],
    file_prefix=[args.file_prefix, args.file_prefix, args.file_prefix, args.file_prefix],
    # train_frac = 0.8,
    # train = False,
    idx_list=idx_dict['val'],
    max_h=max_h,
    max_w=max_w,
    dh_HR=dh_HR,
    dw_HR=dw_HR,
    crop_h=crop_h,
    crop_w=crop_w,
    dim_reduction_step=3,
    dim_reduction_method='NN', #'bicubic',
    crop=False,
    full=True,
    max_val=max_val,
    n_query_points=n_query_points,
    max_scale=max_scale,
    region=args.region, 
)

val_dataloader = DataLoader(dataset=val_dataset, 
                            batch_size=1, 
                            num_workers=8, 
                            pin_memory=True, 
                            shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = args.sigma #25.0
m = args.m #30

pos_encoder = PositionalEncoding(sigma=sigma, m=m)

if args.attention == 'gaussian-attention':
    inn_encoder = EncoderINNwithGaussianAttention(input_dim=1, 
            out_dim=1, 
            feature_dim=64, 
            coord_dim=6*m, 
            hidden_dim=256, 
            mod_depth=args.mod_depth, 
            num_heads=1,
            num_gaussians=num_gaussians,
            num_layers=1,
            out_h=15,
            out_w=20).to(device)
    if num_opts>0:
        inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_encoder, init_type="orthogonal") 

    inn_decoder = DecoderINNwithGaussianAttention(input_dim=1, 
            out_dim=1, 
            feature_dim=64, 
            coord_dim=6*m, 
            hidden_dim=256, 
            mod_depth=args.mod_depth, 
            num_heads=1,
            num_gaussians=num_gaussians,
            num_layers=1,
            out_h=120, #dh_HR,
            out_w=160, #dw_HR
    ).to(device)
    if num_opts>0:
        inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_decoder, init_type="orthogonal") 
elif args.attention == 'regular-attention':
    inn_encoder = EncoderINNwithAttention(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=15,
            out_w=20,
    ).to(device)
    if num_opts>0:
        inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_encoder, init_type="orthogonal")

    inn_decoder = DecoderINNwithAttention(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=120,
            out_w=160
    ).to(device)
    if num_opts>0:
        inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_decoder, init_type="orthogonal") 
elif args.attention == 'no-attention':
    inn_encoder = EncoderINN(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=15,
            out_w=20
    ).to(device)
    if num_opts>0:
        inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_encoder, init_type="orthogonal")

    inn_decoder = DecoderINN(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=120,
            out_w=160
    ).to(device)
    if num_opts>0:
        inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    else:
        init_weights(inn_decoder, init_type="orthogonal") 
else:
    raise NotImplementedError("Not Available INN method")

feature_decoder = edsr.EDSR(n_inputs=1, 
    n_outputs=64, 
    n_resblocks=16, 
    n_feats=64, 
    res_scale=1, 
    scale=8,
    no_upsampling=True).to(device)
if num_opts>0:
    feature_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'feature_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
else:
    init_weights(feature_decoder, init_type="orthogonal")

# liif_decoder = INR(
#     in_dim=64*9 + 4, #+ 4*args.m + 1000, #1000 is for resnet50
#     act='relu',
# ).to(device)
liif_decoder = KAN( 
    in_dim=64*9 + 4, #args.n_encoder_features*9 + 4 + 4*args.m + 1000, #1000 is for resnet50
).to(device)
if num_opts>0:
    liif_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'liif_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
else:
    # init_weights(liif_decoder, init_type="orthogonal")
    pass

# train #

recon_loss = nn.L1Loss(reduction='mean') 

param_list = [
    {'params': inn_encoder.parameters()}, #'lr': encoder_lr, 'weight_decay': encoder_wd},
    {'params': inn_decoder.parameters()}, #'lr': encoder_lr, 'weight_decay': encoder_wd},  
    {'params': feature_decoder.parameters()}, #'lr': decoder_lr, 'weight_decay': encoder_wd},
    {'params': liif_decoder.parameters()}, #'lr': decoder_lr, 'weight_decay': encoder_wd},
]

# num_epochs = 1000

optimizer = torch.optim.Adam(param_list)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                              base_lr=args.min_lr, 
                                              max_lr=args.max_lr, 
                                              step_size_up=3000, 
                                              step_size_down=None, 
                                              mode='triangular', 
                                              gamma=1.0,
                                              cycle_momentum=False)

save_interval = args.save_interval #15000
eval_interval = args.eval_interval #15000

h_in_tensor = torch.stack([get_3D_coords(xy_shape=[dh_HR//8, dw_HR//8], z=h, flatten=False) for h in [0.04, 0.24, 0.64, 0.8]], dim=0)
h_out_tensor = torch.stack([get_3D_coords(xy_shape=[dh_HR//8, dw_HR//8], z=h, flatten=False) for h in [0.04, 0.24, 0.64, 0.8]], dim=0)

h_in_tensor = pos_encoder(h_in_tensor)
h_out_tensor = pos_encoder(h_out_tensor)

height_list = [10, 60, 160, 200]
height_idx = [0, 1, 2, 3]

if args.skip_height is not None:
    skip_height_idx = height_list.index(args.skip_height)
    height_idx.remove(skip_height_idx)

# ======================================================= #
inn_encoder.train()
inn_decoder.train()
feature_decoder.train()
liif_decoder.train()
while num_opts < max_opts:
    for batch in train_dataloader:
        coord, cell = batch["grid_HR"].to(device), batch["cell"].to(device)
        d_img_LR = batch["d_img_LR"]

        [b, m, _, _] = d_img_LR.shape

        d_img_LR = rearrange(d_img_LR, 
                        'b m h w -> m b h w', 
                        w=d_img_LR.shape[-1],
                        h=d_img_LR.shape[-2],
                        m=d_img_LR.shape[-3],
                        b=d_img_LR.shape[0])

        c_img_HR = batch["c_img_HR"]
        c_img_HR = rearrange(c_img_HR,
                             'b m q c -> m b q c',
                             b=c_img_HR.shape[0],
                             m=c_img_HR.shape[1])

        # mod_in_idx = np.random.randint(low=0, high=4)
        # mod_out_idx = np.random.randint(low=0, high=4)

        mod_in_idx, mod_out_idx = np.random.choice(height_idx, size=2, replace=True)

        h_in = h_in_tensor[mod_in_idx].unsqueeze(0).repeat(b, 1, 1, 1).to(device)
        h_out = h_out_tensor[mod_out_idx].unsqueeze(0).repeat(b, 1, 1, 1).to(device)

        # for mod_in_idx, h_in in enumerate(h_in_tensor):
        d_img_HR_in = d_img_LR[mod_in_idx][:,None,:,:].to(device)
            # for mod_out_idx, h_out in enumerate(h_out_tensor):
        
        # print("Input Data Shape: ", d_img_HR_in.shape)
        d_img_LR_hat = inn_encoder(d_img_HR_in, h_in)
        # print("Latent Data Shape: ", d_img_LR_hat.shape)

        d_img_LR_hat = inn_decoder(d_img_LR_hat, h_out)
        # print("Latent Data in Target Modality Shape: ", d_img_LR_hat.shape)
        d_img_HR_hat = feature_decoder(d_img_LR_hat, shape=None)
        # print("Feature Data Shape: ", d_img_HR_hat.shape)

        c_img_HR_target = c_img_HR[mod_out_idx].to(device)

        c_img_HR_hat = pred_liif(liif_decoder,
              d_img_HR_hat,
              coord,
              cell)
            
        loss = recon_loss(c_img_HR_hat, c_img_HR_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        num_opts += 1     

        template = '# [{}/{}]: Loss={:.5f}'
        line = template.format(num_opts, max_opts, loss)
        print(line, end='\r', file=sys.stderr)   

        if num_opts%eval_interval==0:
            eval_model(
                inn_encoder,
                inn_decoder,
                feature_decoder,
                liif_decoder,
                val_dataloader,
                device,
                recon_loss,
                h_in_tensor,
                h_out_tensor
            )

        if num_opts%save_interval==0:
            path = os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts))
            torch.save(inn_encoder.state_dict(), path)

            path = os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts))
            torch.save(inn_decoder.state_dict(), path)

            path = os.path.join(model_save_dir, 'feature_decoder_opt_{}.pt'.format(num_opts))
            torch.save(feature_decoder.state_dict(), path)

            path = os.path.join(model_save_dir, 'liif_decoder_opt_{}.pt'.format(num_opts))
            torch.save(liif_decoder.state_dict(), path)

        # if num_opts%lr_schedule_interval==0:
        scheduler.step()

# ======================================================= #

# eval_idm(
#     latent_encoder,
#     edsr_encoder,
#     unet_decoder,
#     val_dataloader,
#     sqrt_alphas_cumprod_prev,
#     betas,
#     num_timesteps,
#     device,
#     recon_loss,
#     dh_LR, #:int, #= 30
#     dw_LR, #:int, #= 40
#     max_h, #:int, #= 480, 
#     max_w, #:int, #= 640
#     dh_HR, #:int, #= 240
#     dw_HR, #:int, #= 320
#     single_sample_eval=False,
#     save_sample=True
# )

# ======================================================= #

# # ======================================================= #

# from functools import (partial, reduce)
# import operator

# for mod_latent_encoder in latent_encoder:
#     mod_latent_encoder._count_params()

# c = 0
# for p in latent_encoder.parameters():
#     c += reduce(operator.mul, list(p.size()))
# print(f"Total params: %.2fM" % (c/1000000.0))
# print(f"Total params: %.2fk" % (c/1000.0))

# for mod_edsr_encoder in edsr_encoder:
#     mod_edsr_encoder._count_params()
    
# c = 0
# for p in edsr_encoder.parameters():
#     c += reduce(operator.mul, list(p.size()))
# print(f"Total params: %.2fM" % (c/1000000.0))
# print(f"Total params: %.2fk" % (c/1000.0))
    
# for mod_unet_decoder in unet_decoder:
#     mod_unet_decoder._count_params()
    
# c = 0
# for p in unet_decoder.parameters():
#     c += reduce(operator.mul, list(p.size()))
# print(f"Total params: %.2fM" % (c/1000000.0))
# print(f"Total params: %.2fk" % (c/1000.0))

# import subprocess as sp
# import os

# def get_gpu_memory():
#     command = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     return memory_free_values

# print(get_gpu_memory())

# print(total_loss)
