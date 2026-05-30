import sys
from datamodules.data_set import load_3D_data, get_coords, ZscoreStandardizer_3D, MinMaxStandardizer, LIIFDataset, LIIFVizDataset, IDMDataset, IDMDatasetv2, LIIFDatasetv2, LIIFDatasetVizv2
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
from models.LatentEncoder import LatentEncoder
# from torchvision.transforms.functional import InterpolationMode
import random
import torch.nn as nn
from einops import rearrange 
# from models.sr3_modules import diffusion, unet, edsr, mlp
from models.sr3_modules import edsr#, mlp
# from models.sr3_modules.MMAttentionEncoder import MMAttentionEncoderv2, MMAttentionEncoderv1
from models.INRNet import INR, INRwithGaussianAttention
from models.KAN import KAN
from models.INRNet import EncoderINNwithGaussianAttention, DecoderINNwithGaussianAttention, EncoderINNwithAttention, DecoderINNwithAttention, EncoderINN, DecoderINN
from models.FourierEncoding import PositionalEncoding
from inspect import isfunction
import numpy as np
import torch
import torch.nn.functional as F  

import functools
from torch.nn import init
import math
from tqdm.contrib import tzip

from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

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
        lpips,
        h_in_tensor:torch.Tensor,
        h_out_tensor:torch.Tensor,
        dh_HR:int, #= 240
        dw_HR:int, #= 320
        scale:float,
        sample_size:int=1,
        split_size:int=16384,
        plot_image:bool=False,
    ):

    print(' '*90, end='\r', file=sys.stderr)

    data = {}
    if plot_image:
        data["target"] = np.empty(shape=(min(sample_size, val_dataloader.__len__()),4,int(scale*dh_HR),int(scale*dw_HR)))
        data["prediction"] = np.empty(shape=(min(sample_size, val_dataloader.__len__()),4,4,int(scale*dh_HR),int(scale*dw_HR)))

    data["PSNR"] = np.empty(shape=(min(sample_size, val_dataloader.__len__()),4,4))
    data["SSIM"] = np.empty(shape=(min(sample_size, val_dataloader.__len__()),4,4))
    data["LPIPS"] = np.empty(shape=(min(sample_size, val_dataloader.__len__()),4,4))

    for sample_idx, batch in enumerate(val_dataloader):
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
            if plot_image:
                data["target"][sample_idx][mod_in_idx] = c_img_HR[mod_in_idx].view(-1, int(dh_HR*scale), int(dw_HR*scale), 1).permute(0, 3, 1, 2).cpu().numpy().squeeze()
            for mod_out_idx, h_out in enumerate(h_out_tensor):     
                coord_splits = torch.split(coord, split_size_or_sections=split_size, dim=1)
                c_img_HR_hat = []
                for coord_split in coord_splits:
                    d_img_LR_hat = inn_encoder(
                        d_img_HR_in, 
                        h_in.unsqueeze(0).repeat(b,1,1,1).to(device)
                    )

                    d_img_LR_hat = inn_decoder(
                        d_img_LR_hat, 
                        h_out.unsqueeze(0).repeat(b,1,1,1).to(device)
                    )

                    d_img_HR_hat = feature_decoder(d_img_LR_hat, shape=None)

                    # c_img_HR_target = c_img_HR[mod_out_idx].view(-1, int(dh_HR*scale), int(dw_HR*scale), 1).permute(0, 3, 1, 2).to(device)

                    c_img_HR_hat.append(pred_liif(liif_decoder,
                        d_img_HR_hat,
                        coord_split,
                        cell))
                    
                c_img_HR_hat = torch.cat(c_img_HR_hat, dim=1)
                
                c_img_HR_target = c_img_HR[mod_out_idx].view(-1, int(dh_HR*scale), int(dw_HR*scale), 1).permute(0, 3, 1, 2).to(device)
                c_img_HR_hat = c_img_HR_hat.view(-1, int(dh_HR*scale), int(dw_HR*scale), 1).permute(0, 3, 1, 2)

                if plot_image:
                    data["prediction"][sample_idx][mod_in_idx][mod_out_idx] = c_img_HR_hat.cpu().numpy().squeeze()

                num_evals += 1

                noise = c_img_HR_target.cpu().numpy().squeeze() - c_img_HR_hat.cpu().numpy().squeeze()
                val_range = 2.0
                psnr_value = 10*np.log10((val_range**2)/((noise ** 2).mean()))

                data_range = 2.0
                ssim_value = ssim(c_img_HR_target.cpu().numpy().squeeze(), c_img_HR_hat.cpu().numpy().squeeze(), data_range=data_range)

                data["PSNR"][sample_idx][mod_in_idx][mod_out_idx] = psnr_value
                data["SSIM"][sample_idx][mod_in_idx][mod_out_idx] = ssim_value

                # lpips_value = lpips(c_img_HR_hat.clamp(min=-1.0,max=1.0).tile(1,3,1,1).cpu(), c_img_HR_target.tile(1,3,1,1).cpu())
                lpips_value = lpips(c_img_HR_hat.tile(1,3,1,1).cpu(), c_img_HR_target.tile(1,3,1,1).cpu())

                data["LPIPS"][sample_idx][mod_in_idx][mod_out_idx] = lpips_value

                template = '# [{}/{}]: LPIPS={:.5f}, PSNR={:.5f}, SSIM={:.5f}'
                line = template.format(num_evals, 16, lpips_value, psnr_value, ssim_value)
                print(line, end='\n', file=sys.stderr) 

            print(' '*90, end='\r', file=sys.stderr)

        if (sample_idx+1)%sample_size==0:
            break

    return data
    
data_dir = './data'

import argparse
parser = argparse.ArgumentParser('Train Multi-Modality Super Resolution')
parser.add_argument('--attention', type=str, choices=['no-attention', 'regular-attention', 'gaussian-attention'], default='gaussian-attention')
parser.add_argument('--file-prefix', type=str, choices=['ua', 'va'], default='ua')
parser.add_argument('--num-gaussians', type=int, default=2)
# parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-h', type=int, default=600)
parser.add_argument('--max-w', type=int, default=800)
parser.add_argument('--dh-HR', type=int, default=120)
parser.add_argument('--dw-HR', type=int, default=160)
# parser.add_argument('--n-query-points', type=int, default=16384)
parser.add_argument('--max-scale', type=float, default=5.0)
parser.add_argument('--max-val', type=float, default=50.0)
parser.add_argument('--num-opts', type=int, default=15000)
# parser.add_argument('--max-opts', type=int, default=7500)
parser.add_argument('--crop-h', type=int, default=750)
parser.add_argument('--crop-w', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=25.0)
parser.add_argument('--m', type=int, default=30)
parser.add_argument('--mod-depth', type=int, default=8)
parser.add_argument('--split-size', type=int, default=16384)
parser.add_argument('--train-region', type=str, default=None, choices=['A', 'B', 'C', 'D'])
parser.add_argument('--test-region', type=str, default=None, choices=['A', 'B', 'C', 'D'])
parser.add_argument('--skip-height', type=int, default=None, choices=[10, 60, 160, 200])
parser.add_argument('--scale', type=float, default=None)
args = parser.parse_args()
num_gaussians = args.num_gaussians 

# import os
# if args.attention == 'gaussian-attention':
#     # model_save_dir = "saved-liif-wINN-gaussian{}-attention-models-v3".format(num_gaussians)
#     # model_save_dir = os.path.join("saved-liif-wKAN-wINN-gaussian{}-attention-models-3dinn-only-v3-modified".format(num_gaussians), args.file_prefix)
#     model_save_dir = os.path.join("saved-liif-wKAN-wINN-gaussian{}-attention-models-3dinn-only-v3".format(num_gaussians), args.file_prefix)
#     # model_save_dir = os.path.join("saved-liif-wINN-gaussian{}-attention-models-3dinn-only-v3-modified".format(num_gaussians), args.file_prefix)
#     # model_save_dir = os.path.join("saved-liif-wINN-gaussian{}-attention-models-3dinn-only-v3".format(num_gaussians), args.file_prefix)
# elif args.attention == 'regular-attention':
#     # model_save_dir = "saved-liif-wINN-attention-models-v3"
#     model_save_dir = os.path.join("saved-liif-wKAN-wINN-attention-models-3dinn-only-v3-modified", args.file_prefix)
# elif args.attention == 'no-attention':
#     # model_save_dir = "saved-liif-wINN-models-v3"
#     model_save_dir = os.path.join("saved-liif-wKAN-wINN-models-3dinn-only-v3-modified", args.file_prefix)
# else:
#     raise NotImplementedError("Not Available INN method")

import os
if args.attention == 'gaussian-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-gaussian{}-attention-models-3dinn-only-mod-depth-{}".format(num_gaussians, args.mod_depth), args.file_prefix)
elif args.attention == 'regular-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-attention-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
elif args.attention == 'no-attention':
    model_save_dir = os.path.join("saved-liif-wKAN-wINN-models-3dinn-only-mod-depth-{}".format(args.mod_depth), args.file_prefix)
else:
    raise NotImplementedError("Not Available INN method")

if args.train_region is not None:
    model_save_dir = os.path.join(model_save_dir, args.train_region)

if args.skip_height is not None:
    model_save_dir = os.path.join(model_save_dir, "skip_height_{}".format(args.skip_height))

# result_save_dir = os.path.join("Results-liif-wKAN-new-{}".format(args.file_prefix))#, "train-region-{}".format(args.train_region), "test-region-{}".format(args.test_region))
# result_save_dir = os.path.join("Results-liif-wKAN-region-ablation-{}".format(args.file_prefix))
result_save_dir = os.path.join("Results-liif-wKAN-height-ablation-{}".format(args.file_prefix))

# skip_height
if args.skip_height is not None:
    result_save_dir = os.path.join(result_save_dir, "skip_height_{}".format(args.skip_height))

# train_region
if args.train_region is not None:
    result_save_dir = os.path.join(result_save_dir, "train-region-{}".format(args.train_region))

# test_region
if args.test_region is not None:
    result_save_dir = os.path.join(result_save_dir, "test-region-{}".format(args.test_region))

# create directory for saving the results
os.makedirs(result_save_dir, exist_ok=True)

num_opts = args.num_opts #15000 #1200000
print("Using Model Opt: ", num_opts)
# recon_loss = nn.L1Loss(reduction='mean')

dh_HR, dw_HR = args.dh_HR, args.dw_HR
max_h, max_w = args.max_h, args.max_w 
crop_h, crop_w = args.crop_h, args.crop_w 

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
    inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_encoder.eval()

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
    inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_decoder.eval()
    
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
    inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_encoder.eval()

    inn_decoder = DecoderINNwithAttention(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=120,
            out_w=160
    ).to(device)
    inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_decoder.eval()

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
    inn_encoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_encoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_encoder.eval()

    inn_decoder = DecoderINN(input_dim=1,
            out_dim=1,
            feature_dim=64,
            coord_dim=6*m,
            hidden_dim=256,
            mod_depth=args.mod_depth,
            out_h=120,
            out_w=160
    ).to(device)
    inn_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'inn_decoder_opt_{}.pt'.format(num_opts)), map_location=device))
    inn_decoder.eval()
else:
    raise NotImplementedError("Not Available INN method")

feature_decoder = edsr.EDSR(n_inputs=1, 
    n_outputs=64, 
    n_resblocks=16, 
    n_feats=64, 
    res_scale=1, 
    scale=8, 
    no_upsampling=True).to(device)
# init_weights(feature_decoder, init_type="orthogonal")
feature_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'feature_decoder_opt_{}.pt'.format(num_opts)), map_location='cpu'))
feature_decoder.eval()

# liif_decoder = INR(
#     in_dim=64*9 + 4, #+ 4*args.m + 1000, #1000 is for resnet50
#     act='relu',
# ).to(device)
liif_decoder = KAN(
    in_dim=64*9 + 4, #args.n_encoder_features*9 + 4 + 4*args.m + 1000, #1000 is for resnet50
).to(device)
# liif_decoder = INRwithGaussianAttention(
#     in_dim=64*9 + 4, #+ 4*args.m + 1000, #1000 is for resnet50
#     hidden_dim=256,
#     num_heads=1,
#     num_gaussians=16,
#     num_layers=1,
#     act='relu',
# ).to(device)
# init_weights(liif_decoder, init_type="orthogonal")
liif_decoder.load_state_dict(torch.load(os.path.join(model_save_dir, 'liif_decoder_opt_{}.pt'.format(num_opts)), map_location='cpu'))
liif_decoder.eval()

# train #

lpips = LPIPS(net_type='squeeze', reduction='mean', normalize=False).to(torch.device('cpu'))

h_in_tensor = torch.stack([get_3D_coords(xy_shape=[dh_HR//8, dw_HR//8], z=h, flatten=False) for h in [0.04, 0.24, 0.64, 0.8]], dim=0)
h_out_tensor = torch.stack([get_3D_coords(xy_shape=[dh_HR//8, dw_HR//8], z=h, flatten=False) for h in [0.04, 0.24, 0.64, 0.8]], dim=0)

h_in_tensor = pos_encoder(h_in_tensor)
h_out_tensor = pos_encoder(h_out_tensor)

# ======================================================= #

import json
with open('idx-list.json', 'r') as f:
    idx_dict = json.load(f)

# for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
# for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]:
# for scale in [3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]:
# for scale in [3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
# for scale in [6.0, 5.75, 5.5, 5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]:
# for scale in [6.0, 5.75]:
# for scale in [5.5, 5.25, 5.0]:
# for scale in [4.75, 4.5, 4.25, 4.0]:
# for scale in [3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25]:
for scale in [args.scale]:
    # print(
    #     "Super Resolution Scale = {}, \
    #     Mod Depth = {}, \
    #     Test Region = {}, \
    #     Num Gaussians = {}".format(
    #         scale, args.mod_depth, args.test_region, args.num_gaussians
    #     )
    # )

    # val_dataset = LIIFDatasetVizv2(
    #     data_dir=[os.path.join(data_dir, "wind-{}m/{}".format(10, args.file_prefix)),
    #             os.path.join(data_dir, "wind-{}m/{}".format(60, args.file_prefix)),
    #             os.path.join(data_dir, "wind-{}m/{}".format(160, args.file_prefix)),
    #             os.path.join(data_dir, "wind-{}m/{}".format(200, args.file_prefix))],
    #     file_prefix=[args.file_prefix, args.file_prefix, args.file_prefix, args.file_prefix],
    #     # train_frac = 0.8,
    #     # train=False,
    #     idx_list=idx_dict['val'],
    #     max_h=max_h,
    #     max_w=max_w,
    #     dh_HR=dh_HR,
    #     dw_HR=dw_HR,
    #     crop_h=crop_h,
    #     crop_w=crop_w,
    #     dim_reduction_step=3,
    #     dim_reduction_method='NN',
    #     crop=False,
    #     full=True, #choose either crop or full
    #     max_val=50.0,
    #     res_scale=scale,
    #     region=args.test_region,
    # )

    # val_dataloader = DataLoader(dataset=val_dataset, 
    #                         batch_size=1, 
    #                         num_workers=1, 
    #                         pin_memory=True, 
    #                         shuffle=False)

    if args.attention == 'gaussian-attention':
        filepath = os.path.join(result_save_dir, "data-liif-wKAN-gaussian{}-attention-mod-depth-{}-scale-{}.npz".format(num_gaussians, args.mod_depth, scale))
    elif args.attention == 'regular-attention':
        filepath = os.path.join(result_save_dir, "data-liif-wKAN-attention-mod-depth-{}-scale-{}.npz".format(args.mod_depth, scale))
    elif args.attention == 'no-attention':
        filepath = os.path.join(result_save_dir, "data-liif-wKAN-mod-depth-{}-scale-{}.npz".format(args.mod_depth, scale))
    else:
        raise NotImplementedError("Not Available INN method")
    
    print_command = "Super Resolution Scale = {}, \
        Mod Depth = {}, \
        Test Region = {}, \
        Num Gaussians = {}, \
        File Name = {}".format(
            scale, args.mod_depth, args.test_region, args.num_gaussians, filepath
        )

    if os.path.exists(filepath):
        print("File Already Exists For ", print_command)
        pass
    else:
        print("Generating File For ", print_command)
        val_dataset = LIIFDatasetVizv2(
            data_dir=[os.path.join(data_dir, "wind-{}m/{}".format(10, args.file_prefix)),
                    os.path.join(data_dir, "wind-{}m/{}".format(60, args.file_prefix)),
                    os.path.join(data_dir, "wind-{}m/{}".format(160, args.file_prefix)),
                    os.path.join(data_dir, "wind-{}m/{}".format(200, args.file_prefix))],
            file_prefix=[args.file_prefix, args.file_prefix, args.file_prefix, args.file_prefix],
            # train_frac = 0.8,
            # train=False,
            idx_list=idx_dict['test'],
            max_h=max_h,
            max_w=max_w,
            dh_HR=dh_HR,
            dw_HR=dw_HR,
            crop_h=crop_h,
            crop_w=crop_w,
            dim_reduction_step=3,
            dim_reduction_method='NN',
            crop=False,
            full=True, #choose either crop or full
            max_val=50.0,
            res_scale=scale,
            region=args.test_region,
        )

        val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=1, 
                                num_workers=1, 
                                pin_memory=True, 
                                shuffle=False)

        data = eval_model(
            inn_encoder,
            inn_decoder,
            feature_decoder,
            liif_decoder,
            val_dataloader,
            device,
            lpips,
            h_in_tensor,
            h_out_tensor,
            dh_HR, #= 240
            dw_HR, #= 320
            scale,
            sample_size=800,
            plot_image=False,
            split_size=args.split_size,
        )

        # if args.attention == 'gaussian-attention':
        #     np.savez(filepath, **data)
        # elif args.attention == 'regular-attention':
        #     np.savez(filepath, **data)
        # elif args.attention == 'no-attention':
        #     np.savez(filepath, **data)
        # else:
        #     raise NotImplementedError("Not Available INN method")

        np.savez(filepath, **data)

        print("Generated File For ", print_command)

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
