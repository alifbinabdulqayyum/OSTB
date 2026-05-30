import torch
import math
import random
import numpy as np
# import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
# from pytorch_lightning.trainer.supporters import CombinedLoader
from typing import Optional
# from hydra.utils import log
import glob
import os
import pywt
from tqdm import tqdm

def load_3D_data(
    m0_data_dir:str, 
    m1_data_dir:str,
    m2_data_dir:str, 
    m3_data_dir:str,
    m0_file_prefix:str,
    m1_file_prefix:str,
    m2_file_prefix:str,
    m3_file_prefix:str,
    total_size:int=None,
    visualization:bool=False,
    val_size:int=250
):
    m0_idx_list = [
        idx.replace(m0_data_dir+ '/'+m0_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m0_data_dir+'/*.npy')
    ]
    m0_idx_list = list(map(int, m0_idx_list))
    m0_idx_list.sort()

    m1_idx_list = [
        idx.replace(m1_data_dir+ '/'+m1_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m1_data_dir+'/*.npy')
    ]
    m1_idx_list = list(map(int, m1_idx_list))
    m1_idx_list.sort()

    m2_idx_list = [
        idx.replace(m2_data_dir+ '/'+m2_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m2_data_dir+'/*.npy')
    ]
    m2_idx_list = list(map(int, m2_idx_list))
    m2_idx_list.sort()

    m3_idx_list = [
        idx.replace(m3_data_dir+ '/'+m3_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m3_data_dir+'/*.npy')
    ]
    m3_idx_list = list(map(int, m3_idx_list))
    m3_idx_list.sort()

    if m0_idx_list == m1_idx_list == m2_idx_list == m3_idx_list:
        if total_size is not None:
            m0_idx_list = m0_idx_list[:total_size]
            m1_idx_list = m1_idx_list[:total_size]
            m2_idx_list = m2_idx_list[:total_size]
            m3_idx_list = m3_idx_list[:total_size]

        # To reduce the time to load the data at visualization
        if visualization:
            total_datapoints = len(m0_idx_list)
            m0_idx_list = m0_idx_list[total_datapoints-val_size:]
            m1_idx_list = m1_idx_list[total_datapoints-val_size:]
            m2_idx_list = m2_idx_list[total_datapoints-val_size:]
            m3_idx_list = m3_idx_list[total_datapoints-val_size:]
        # =================== #
        data = []
        with tqdm(total=len(m0_idx_list)) as pbar:
            for idx, timestep in enumerate(m0_idx_list):
                m0_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))
                m1_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))
                m2_tmp = np.load(os.path.join(m2_data_dir, m2_file_prefix+'_{}.npy'.format(timestep)))
                m3_tmp = np.load(os.path.join(m3_data_dir, m3_file_prefix+'_{}.npy'.format(timestep)))

                data.append(
                    np.stack(
                        [
                            m0_tmp,
                            m1_tmp,
                            m2_tmp,
                            m3_tmp
                        ],
                        axis=0
                    )
                )
                
                pbar.update(1)
        data = np.stack(data, axis=0)
    else:
        data = None

    return torch.from_numpy(data), m0_idx_list
    

def load_data(
    m0_data_dir:str, 
    m1_data_dir:str,
    m0_file_prefix:str,
    m1_file_prefix:str,
    total_size:int=None,
    visualization:bool=False,
    val_size:int=250
):
    """
    Load images and return the image stack.
    Args:
        m0_data_dir (string): first modality dataset file path
        m1_data_dir (string): second modality dataset file path
        m0_file_prefix (string): first modality dataset filename prefix
        m1_file_prefix (string): second modality dataset filename prefix
    Returns:
        (mm_m0_dat, mm_m1_dat) (tensor,tensor): (num * H * W, num * H * W)
        m0_only_dat (tensor): num * H * W
        m1_only_dat (tensor): num * H * W
    """  

    m0_idx_list = [
        idx.replace(m0_data_dir+ '/'+m0_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m0_data_dir+'/*.npy')
    ]
    m0_idx_list = list(map(int, m0_idx_list))
    m0_idx_list.sort()

    m1_idx_list = [
        idx.replace(m1_data_dir+ '/'+m1_file_prefix+'_', "").replace('.npy', "") 
        for idx in glob.glob(m1_data_dir+'/*.npy')
    ]
    m1_idx_list = list(map(int, m1_idx_list))
    m1_idx_list.sort()

    if m0_idx_list == m1_idx_list:
        if total_size is not None:
            m0_idx_list = m0_idx_list[:total_size]
            m1_idx_list = m1_idx_list[:total_size]
        
        # To reduce the time to load the data at visualization
        if visualization:
            total_datapoints = len(m0_idx_list)
            m0_idx_list = m0_idx_list[total_datapoints-val_size:]
            m1_idx_list = m1_idx_list[total_datapoints-val_size:]
        # =================== #
        m0_dat, m1_dat = [], []
        with tqdm(total=len(m0_idx_list)) as pbar:
            for idx, timestep in enumerate(m0_idx_list):
                m0_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))
                m1_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))

                # m0_tmp = np.expand_dims(m0_tmp, axis=0)
                # m1_tmp = np.expand_dims(m1_tmp, axis=0)

                # if idx == 0:
                #     m0_dat = m0_tmp
                #     m1_dat = m1_tmp
                # else:
                #     m0_dat = np.vstack((m0_dat, m0_tmp))
                #     m1_dat = np.vstack((m1_dat, m1_tmp))

                m0_dat.append(m0_tmp)
                m1_dat.append(m1_tmp)
                pbar.update(1)
        m0_dat = np.stack(m0_dat, axis=0)
        m1_dat = np.stack(m1_dat, axis=0)
    else:
        m0_dat, m1_dat = None, None

    return ((torch.from_numpy(m0_dat), torch.from_numpy(m1_dat)), (m0_idx_list, m1_idx_list))


# def load_data(
#     m0_data_dir:str, 
#     m1_data_dir:str,
#     m0_file_prefix:str,
#     m1_file_prefix:str
# ):
#     """
#     Load images and return the image stack.
#     Args:
#         m0_data_dir (string): first modality dataset file path
#         m1_data_dir (string): second modality dataset file path
#         m0_file_prefix (string): first modality dataset filename prefix
#         m1_file_prefix (string): second modality dataset filename prefix
#     Returns:
#         (mm_m0_dat, mm_m1_dat) (tensor,tensor): (num * H * W, num * H * W)
#         m0_only_dat (tensor): num * H * W
#         m1_only_dat (tensor): num * H * W
#     """  

#     m0_idx_list = [
#         idx.replace(m0_data_dir+ '/'+m0_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m0_data_dir+'/*.npy')
#     ]
#     m0_idx_list = set(map(int, m0_idx_list))

#     m1_idx_list = [
#         idx.replace(m1_data_dir+ '/'+m1_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m1_data_dir+'/*.npy')
#     ]
#     m1_idx_list = set(map(int, m1_idx_list))

#     all_idx_list = m0_idx_list | m1_idx_list
#     mm_idx_list = m0_idx_list & m1_idx_list
#     m0_only_idx_list = m0_idx_list - mm_idx_list
#     m1_only_idx_list = m1_idx_list - mm_idx_list

#     if mm_idx_list is not None:
#         for idx, timestep in enumerate(mm_idx_list):
#             m0_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))
#             m1_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))

#             m0_tmp = np.expand_dims(m0_tmp, axis=0)
#             m1_tmp = np.expand_dims(m1_tmp, axis=0)

#             if idx == 0:
#                 m0_dat = m0_tmp
#                 m1_dat = m1_tmp
#             else:
#                 m0_dat = np.vstack((m0_dat, m0_tmp))
#                 m1_dat = np.vstack((m1_dat, m1_tmp))
#             # if idx==64-1:break
#     else:
#         m0_dat, m1_dat = None, None

#     if len(m0_only_idx_list):
#         for idx, timestep in enumerate(m0_only_idx_list):
#             m0_only_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))

#             m0_only_tmp = np.expand_dims(m0_only_tmp, axis=0)

#             if idx == 0:
#                 m0_only_dat = m0_only_tmp
#             else:
#                 m0_only_dat = np.vstack((m0_only_dat, m0_only_tmp))

#     else:
#         m0_only_dat = None

#     if len(m1_only_idx_list):
#         for idx, timestep in enumerate(m1_only_idx_list):
#             m1_only_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))

#             m1_only_tmp = np.expand_dims(m1_only_tmp, axis=0)

#             if idx == 0:
#                 m1_only_dat = m1_only_tmp
#             else:
#                 m1_only_dat = np.vstack((m1_only_dat, m1_only_tmp))

#     else:
#         m1_only_dat = None

#     return ((torch.from_numpy(m0_dat), torch.from_numpy(m1_dat)), 
#             torch.from_numpy(m0_only_dat) if isinstance(m0_only_dat, np.ndarray) else m0_only_dat, 
#             torch.from_numpy(m1_only_dat) if isinstance(m1_only_dat, np.ndarray) else m1_only_dat)

EPSILON = 1e-6

def get_coords(shape, ranges=None, flatten=True):
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
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1 + EPSILON, 1 - EPSILON #-1, 1
        else:
            v0, v1 = ranges[i]
        # r = (v1 - v0) / (2 * n)
        r = (v1 - v0) / (2 * (n - 1))
        # seq = v0 + r + (2 * r) * torch.arange(n).float()
        seq = v0 + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    # make mesh
    coords = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        coords = coords.view(-1, coords.shape[-1])
    return coords


class ZscoreStandardizer(object):
    '''
    Normalization transformation
    '''
    def __init__(self, x):
        # self.mean = torch.mean(x, 0)
        self.mean = torch.mean(x, (-2,-1), keepdim=True)
        # self.std = torch.std(x, 0)
        self.std = torch.std(x, (-2,-1), keepdim=True)
        self.epsilon = 1e-10
        assert self.mean.shape == self.std.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.epsilon) + self.mean

class ZscoreStandardizer_3D(object):
    '''
    Normalization transformation
    '''
    def __init__(self, x):
        # self.mean = torch.mean(x, 0)
        self.mean = torch.mean(x, (-3,-2,-1), keepdim=True)
        # self.std = torch.std(x, 0)
        self.std = torch.std(x, (-3,-2,-1), keepdim=True)
        self.epsilon = 1e-10
        assert self.mean.shape == self.std.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.epsilon) + self.mean

class MinMaxStandardizer(object):
    '''
    Min-Max transformation
    '''
    def __init__(self, x):
        self.minVal = torch.min(x, 0)[0]
        self.maxVal = torch.max(x, 0)[0]
        self.epsilon = 1e-10

        assert self.minVal.shape == self.maxVal.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minVal) / (self.maxVal - self.minVal) + self.epsilon

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.epsilon) * (self.maxVal - self.minVal) + self.minVal


# class WindSpeedDataset(Dataset):
#     def __init__(self, 
#         dataset: Dataset, 
#         up_scale: int = 1,
#         low_resol: list = [75, 100],
#         sampling_rate: float = 1.0,
#     ):
      
#         self.dataset = dataset
#         self.up_scale = up_scale
#         self.low_resol = low_resol
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # high resolution sample: H, W -> 1, H, W
#         img_HR = self.dataset[idx]
#         img_HR = torch.unsqueeze(img_HR, 0)

#         # paired sample
#         h_LR, w_LR = self.low_resol
#         h_HR, w_HR = round(h_LR * self.up_scale), round(w_LR * self.up_scale)

#         # random crop
#         # img_HR: 1, h_HR, w_HR
#         # img_LR: 1, h_LR, w_LR
#         img_HR = transforms.RandomCrop((h_HR, w_HR))(img_HR)
#         img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC)(img_HR)
        
#         # get HR coordinates: h_HR * w_HR, 2
#         grid_HR = get_coords([h_HR, w_HR])
        
#         # subset of grid_HR and img_HR 
#         # grid_HR: n_query_pts, 2 
#         # img_HR:  n_query_pts, 1
#         n_query_pts = round(h_HR * w_HR * self.sampling_rate)
#         query_pts = np.random.choice(len(grid_HR), n_query_pts, replace=False)
#         grid_HR = grid_HR[query_pts]

#         if not img_HR.is_contiguous():
#             img_HR = img_HR.contiguous()
#         img_HR = img_HR.view(1, -1).permute(1, 0)
#         img_HR = img_HR[query_pts]

#         return {
#                 "img_LR": img_LR,
#                 "grid_HR": grid_HR,
#                 "img_HR": img_HR
#                 }


class LIIFDataset(Dataset):
    def __init__(self, 
            m0_data_dir: str,
            m1_data_dir: str,
            file_prefix: str,
            liif_scales: list = [1, 5],
            low_resol: list = [75, 100],
            sampling_rate: float = 1.0,
            query_points: int = 256,
            max_val: float = 50,
            train_frac: float = 0.8,
            total_size: int = None,
            train: bool = True
    ):
        self.liif_scales = liif_scales
        self.low_resol = low_resol
        self.sampling_rate = sampling_rate
        self.query_points = query_points
        self.max_val = max_val

        self.m0_data_dir = m0_data_dir
        self.m1_data_dir = m1_data_dir

        self.m0_file_prefix, self.m1_file_prefix = file_prefix, file_prefix

        m0_idx_list = [
            idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m0_data_dir+'/*.npy')
        ]
        m0_idx_list = list(map(int, m0_idx_list))
        # self.m0_idx_list.sort()

        m1_idx_list = [
            idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m1_data_dir+'/*.npy')
        ]
        m1_idx_list = list(map(int, m1_idx_list))
        # self.m1_idx_list.sort()

        self.idx_list = list(set(m0_idx_list) & set(m1_idx_list))
        self.idx_list.sort()

        if total_size is not None:
            total_size = min(len(self.idx_list, total_size))
        else:
            total_size = len(self.idx_list)

        train_size = int(total_size * train_frac)
        val_size = total_size - train_size

        if train:
            self.idx_list = self.idx_list[:train_size]
        else:
            self.idx_list = self.idx_list[train_size:train_size+val_size]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # high resolution sample: H, W -> 1, H, W

        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))

        # high resolution sample: H, W -> 1, H, W
        m0_img_HR = torch.unsqueeze(torch.FloatTensor(m0_tmp), 0) / self.max_val
        
        # high resolution sample: H, W -> 1, H, W
        m1_img_HR = torch.unsqueeze(torch.FloatTensor(m1_tmp), 0) / self.max_val

        # paired sample 
        up_scale = random.uniform(self.liif_scales[0], self.liif_scales[1])
        h_LR, w_LR = self.low_resol
        h_HR, w_HR = round(h_LR * up_scale), round(w_LR * up_scale)
        
        # random crop
        # img_HR: 1, h_HR, w_HR
        # img_LR: 1, h_LR, w_LR
        m0_img_HR, m1_img_HR = transforms.RandomCrop((h_HR, w_HR))(torch.vstack((m0_img_HR,m1_img_HR)))
        
        m0_img_HR = torch.unsqueeze(m0_img_HR, 0)
        m1_img_HR = torch.unsqueeze(m1_img_HR, 0)
        
        m0_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img_HR)
        m1_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img_HR)
        
        # get HR coordinates: h_HR * w_HR, 2
        grid_HR = get_coords([h_HR, w_HR])
        
        # subset of grid_HR and img_HR 
        # grid_HR: n_query_pts, 2 
        # img_HR:  n_query_pts, 1
        n_query_pts = self.query_points
        query_pts = np.random.choice(len(grid_HR), n_query_pts, replace=False)
        grid_HR = grid_HR[query_pts]

        if not m0_img_HR.is_contiguous():
            m0_img_HR = m0_img_HR.contiguous()
        m0_img_HR = m0_img_HR.view(1, -1).permute(1, 0)
        m0_img_HR = m0_img_HR[query_pts]
        
        if not m1_img_HR.is_contiguous():
            m1_img_HR = m1_img_HR.contiguous()
        m1_img_HR = m1_img_HR.view(1, -1).permute(1, 0)
        m1_img_HR = m1_img_HR[query_pts]

        # get cell
        cell = torch.ones(2).int()
        cell[0] = h_HR
        cell[1] = w_HR

        return {
                "m0_img_LR": m0_img_LR,
                "m1_img_LR": m1_img_LR,
                "grid_HR": grid_HR,
                "m0_img_HR": m0_img_HR,
                "m1_img_HR": m1_img_HR,
                "cell": cell
                }


# class WindSpeedVizDataset(Dataset):
#     def __init__(self, 
#         dataset: Dataset, 
#         up_scale: int = 1,
#         low_resol: list = [75, 100],
#     ):
      
#         self.dataset = dataset
#         self.up_scale = up_scale
#         self.low_resol = low_resol

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # high resolution sample: H, W -> 1, H, W
#         img_HR = self.dataset[idx]
#         img_HR = torch.unsqueeze(img_HR, 0)

#         # paired sample
#         h_LR, w_LR = self.low_resol
#         h_HR, w_HR = round(h_LR * self.up_scale), round(w_LR * self.up_scale)

#         # random crop
#         # img_HR: 1, h_HR, w_HR
#         # img_LR: 1, h_LR, w_LR
#         img_HR = transforms.RandomCrop((h_HR, w_HR))(img_HR)
#         img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC)(img_HR)
        
#         if not img_HR.is_contiguous():
#             img_HR = img_HR.contiguous()
#         img_HR = img_HR.view(1, -1).permute(1, 0)        

#         # get HR coordinates: h_HR * w_HR, 2
#         grid_HR = get_coords([h_HR, w_HR])

#         return {
#                 "img_LR": img_LR,
#                 "grid_HR": grid_HR,
#                 "img_HR": img_HR
#                 }


class LIIFVizDataset(Dataset):
    def __init__(self, 
        m0_data_dir: str, 
        m1_data_dir: str,
        file_prefix: str,
        up_scale: float = 1,
        low_resol: list = [75, 100],
        max_val: float = 50.0,
        total_size: int = None,
        train_frac: float = 0.8,
        train: bool = False,
    ):
      
        self.m0_data_dir = m0_data_dir
        self.m1_data_dir = m1_data_dir

        self.m0_file_prefix, self.m1_file_prefix = file_prefix, file_prefix

        m0_idx_list = [
            idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m0_data_dir+'/*.npy')
        ]
        m0_idx_list = list(map(int, m0_idx_list))
        # self.m0_idx_list.sort()

        m1_idx_list = [
            idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
            for idx in glob.glob(self.m1_data_dir+'/*.npy')
        ]
        m1_idx_list = list(map(int, m1_idx_list))
        # self.m1_idx_list.sort()

        self.idx_list = list(set(m0_idx_list) & set(m1_idx_list))
        self.idx_list.sort()

        if total_size is not None:
            total_size = min(len(self.idx_list, total_size))
        else:
            total_size = len(self.idx_list)

        train_size = int(total_size * train_frac)
        val_size = total_size - train_size

        if train:
            self.idx_list = self.idx_list[:train_size]
        else:
            self.idx_list = self.idx_list[train_size:train_size+val_size]
        self.up_scale = up_scale
        self.low_resol = low_resol
        self.max_val = max_val

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # # high resolution sample: H, W -> 1, H, W
        # m0_img = self.m0_dataset[idx] / self.max_val
        # m0_img = torch.unsqueeze(m0_img, 0)
        
        # # high resolution sample: H, W -> 1, H, W
        # m1_img = self.m1_dataset[idx] / self.max_val
        # m1_img = torch.unsqueeze(m1_img, 0)

        # paired sample
        h_LR, w_LR = self.low_resol
        h_HR, w_HR = round(h_LR * self.up_scale), round(w_LR * self.up_scale)

        # paired sample
        # img_HR: 1, h_HR, w_HR
        # img_LR: 1, h_LR, w_LR
        # m0_img_HR, m1_img_HR = transforms.RandomCrop((h_HR, w_HR))(torch.vstack((m0_img_HR,m1_img_HR)))
        
        # m0_img_HR = torch.unsqueeze(m0_img_HR, 0)
        # m1_img_HR = torch.unsqueeze(m1_img_HR, 0)

        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))

        # high resolution sample: H, W -> 1, H, W
        m0_img = torch.unsqueeze(torch.FloatTensor(m0_tmp), 0) / self.max_val
        
        # high resolution sample: H, W -> 1, H, W
        m1_img = torch.unsqueeze(torch.FloatTensor(m1_tmp), 0) / self.max_val
        
        m0_img_HR = transforms.Resize((h_HR, w_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img)
        m1_img_HR = transforms.Resize((h_HR, w_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img)
        
        m0_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m0_img)
        m1_img_LR = transforms.Resize((h_LR, w_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(m1_img)
        
        # grid_HR: h_HR * w_HR, 2 
        # img_HR:  h_HR * w_HR, 1
        grid_HR = get_coords([h_HR, w_HR])

        # if not m0_img_HR.is_contiguous():
        #     m0_img_HR = m0_img_HR.contiguous()
        # m0_img_HR = m0_img_HR.view(1, -1).permute(1, 0)
        
        # if not m1_img_HR.is_contiguous():
        #     m1_img_HR = m1_img_HR.contiguous()
        # m1_img_HR = m1_img_HR.view(1, -1).permute(1, 0)

        # get cell
        cell = torch.ones(2).int()
        cell[0] = h_HR
        cell[1] = w_HR

        return {
                "m0_img_LR": m0_img_LR,
                "m1_img_LR": m1_img_LR,
                "grid_HR": grid_HR,
                "m0_img_HR": m0_img_HR,
                "m1_img_HR": m1_img_HR,
                "cell": cell
                }


# class DataSetModule(pl.LightningDataModule):
#     def __init__(self, 
#         data_dir: str = "data/wind/ua/",
#         pre_method: str = "zscore",
#         n_train_val_test: list = [800, 100, 100],
#         b_train_val_test: list = [20, 10, 10],
#         up_scales: list = [3, 3, 5],
#         liif: bool = False,
#         liif_scales: list = [1, 5],
#         low_resol: list = [75, 100],
#         sampling_rate: float = 1.0,
#         num_workers: int = 4,
#     ):
#         super().__init__()

#         self.data_dir = data_dir
#         self.pre_method = pre_method
#         self.n_train, self.n_val, self.n_test = n_train_val_test
#         self.b_train, self.b_val, self.b_test = b_train_val_test
#         self.up_scale_train, self.up_scale_val, self.up_scale_test = up_scales
#         self.liif = liif
#         self.liif_scales = liif_scales
#         self.low_resol = low_resol
#         self.sampling_rate = sampling_rate
#         self.num_workers = num_workers

#     def prepare_data(self):
#         pass

#     def setup(self, stage: Optional[str] = None):
#         # load data: num_sample * H * W
#         train_data = load_data(self.data_dir, 0, self.n_train)
#         val_data   = load_data(self.data_dir, self.n_train, self.n_train+self.n_val)
#         test_data  = load_data(self.data_dir, self.n_train+self.n_val, self.n_train+self.n_val+self.n_test)

#         # data normalization
#         normalizer = self._get_normalizer(torch.vstack((train_data, val_data)))
#         train_data = normalizer.do(train_data)
#         val_data   = normalizer.do(val_data)
#         test_data  = normalizer.do(test_data)

#         # prepare visualization data
#         idx_viz_val  = np.random.permutation(val_data.shape[0])[:self.b_val]
#         idx_viz_test = np.random.permutation(test_data.shape[0])[:self.b_test]
#         val_data_viz  = val_data[idx_viz_val,:,:]
#         test_data_viz = test_data[idx_viz_test,:,:]
        
#         # dataset
#         if self.liif:
#             self.train_data = LIIFDataset(train_data, liif_scales=self.liif_scales, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.val_data1  = LIIFDataset(val_data, liif_scales=self.liif_scales, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.val_data2  = LIIFVizDataset(val_data_viz, up_scale=self.up_scale_val, low_resol=self.low_resol)
#             self.test_data1 = LIIFDataset(test_data, liif_scales=self.liif_scales, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.test_data2 = LIIFVizDataset(test_data_viz, up_scale=self.up_scale_test, low_resol=self.low_resol)
#         else:
#             self.train_data = WindSpeedDataset(train_data, up_scale=self.up_scale_train, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.val_data1  = WindSpeedDataset(val_data, up_scale=self.up_scale_val, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.val_data2  = WindSpeedVizDataset(val_data_viz, up_scale=self.up_scale_val, low_resol=self.low_resol)
#             self.test_data1 = WindSpeedDataset(test_data, up_scale=self.up_scale_test, low_resol=self.low_resol, sampling_rate=self.sampling_rate)
#             self.test_data2 = WindSpeedVizDataset(test_data_viz, up_scale=self.up_scale_test, low_resol=self.low_resol)

#         # free memory
#         del train_data, val_data, val_data_viz, test_data, test_data_viz
#         return normalizer

#     def train_dataloader(self):
#         return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, pin_memory=True, shuffle=True)

#     def val_dataloader(self):
#         loader_eval = DataLoader(dataset=self.val_data1, batch_size=self.b_val, num_workers=self.num_workers, pin_memory=True, shuffle=False)
#         loader_viz  = DataLoader(dataset=self.val_data2, batch_size=self.b_val, num_workers=self.num_workers, pin_memory=True, shuffle=False)
#         loaders = {"loader_eval": loader_eval, "loader_viz": loader_viz}
#         combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
#         return combined_loaders
            
#     def test_dataloader(self):
#         loader_eval = DataLoader(dataset=self.test_data1, batch_size=self.b_test, num_workers=self.num_workers, pin_memory=True, shuffle=False)
#         loader_viz  = DataLoader(dataset=self.test_data2, batch_size=self.b_test, num_workers=self.num_workers, pin_memory=True, shuffle=False)
#         loaders = {"loader_eval": loader_eval, "loader_viz": loader_viz}
#         combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
#         return combined_loaders

#     def _get_normalizer(self, dat):
#         '''
#         Create normalizer
#         '''
#         if self.pre_method == "zscore":
#             log.info(f"Preprocessing method: {self.pre_method}")
#             return ZscoreStandardizer(dat)
#         elif self.pre_method == "minmax":
#             log.info(f"Preprocessing method: {self.pre_method}")
#             return MinMaxStandardizer(dat)
#         else:
#             log.info("Preprocessing method: None")
#             return None

import random

class IDMDataset(Dataset):
    def __init__(self, 
            dataset: Dataset,
            d_HR_resol: list = [480, 640],
            max_resol: list = [1500, 2000],
            d_LR_resol: list = [60, 80],
            sampling_rate: float = 1.0,
    ):
      
        self.dataset = dataset
        self.sampling_rate = sampling_rate
        self.dh_HR, self.dw_HR = d_HR_resol
        self.dh_LR, self.dw_LR = d_LR_resol
        self.max_h, self.max_w = max_resol

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_HR = self.dataset[idx]

        return {
                "img_HR": img_HR,
                }


# def load_3D_data(
#     m0_data_dir:str, 
#     m1_data_dir:str,
#     m2_data_dir:str, 
#     m3_data_dir:str,
#     m0_file_prefix:str,
#     m1_file_prefix:str,
#     m2_file_prefix:str,
#     m3_file_prefix:str,
#     total_size:int=None,
#     visualization:bool=False,
#     val_size:int=250
# ):
#     m0_idx_list = [
#         idx.replace(m0_data_dir+ '/'+m0_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m0_data_dir+'/*.npy')
#     ]
#     m0_idx_list = list(map(int, m0_idx_list))
#     m0_idx_list.sort()

#     m1_idx_list = [
#         idx.replace(m1_data_dir+ '/'+m1_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m1_data_dir+'/*.npy')
#     ]
#     m1_idx_list = list(map(int, m1_idx_list))
#     m1_idx_list.sort()

#     m2_idx_list = [
#         idx.replace(m2_data_dir+ '/'+m2_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m2_data_dir+'/*.npy')
#     ]
#     m2_idx_list = list(map(int, m2_idx_list))
#     m2_idx_list.sort()

#     m3_idx_list = [
#         idx.replace(m3_data_dir+ '/'+m3_file_prefix+'_', "").replace('.npy', "") 
#         for idx in glob.glob(m3_data_dir+'/*.npy')
#     ]
#     m3_idx_list = list(map(int, m3_idx_list))
#     m3_idx_list.sort()

#     if m0_idx_list == m1_idx_list == m2_idx_list == m3_idx_list:
#         if total_size is not None:
#             m0_idx_list = m0_idx_list[:total_size]
#             m1_idx_list = m1_idx_list[:total_size]
#             m2_idx_list = m2_idx_list[:total_size]
#             m3_idx_list = m3_idx_list[:total_size]

#         # To reduce the time to load the data at visualization
#         if visualization:
#             total_datapoints = len(m0_idx_list)
#             m0_idx_list = m0_idx_list[total_datapoints-val_size:]
#             m1_idx_list = m1_idx_list[total_datapoints-val_size:]
#             m2_idx_list = m2_idx_list[total_datapoints-val_size:]
#             m3_idx_list = m3_idx_list[total_datapoints-val_size:]
#         # =================== #
#         data = []
#         with tqdm(total=len(m0_idx_list)) as pbar:
#             for idx, timestep in enumerate(m0_idx_list):
#                 m0_tmp = np.load(os.path.join(m0_data_dir, m0_file_prefix+'_{}.npy'.format(timestep)))
#                 m1_tmp = np.load(os.path.join(m1_data_dir, m1_file_prefix+'_{}.npy'.format(timestep)))
#                 m2_tmp = np.load(os.path.join(m2_data_dir, m2_file_prefix+'_{}.npy'.format(timestep)))
#                 m3_tmp = np.load(os.path.join(m3_data_dir, m3_file_prefix+'_{}.npy'.format(timestep)))

#                 data.append(
#                     np.stack(
#                         [
#                             m0_tmp,
#                             m1_tmp,
#                             m2_tmp,
#                             m3_tmp
#                         ],
#                         axis=0
#                     )
#                 )
                
#                 pbar.update(1)
#         data = np.stack(data, axis=0)
#     else:
#         data = None

#     return torch.from_numpy(data), m0_idx_list   

class IDMDatasetv2(Dataset):
    def __init__(self, 
                data_dir:list, 
                file_prefix: list,
                # total_size: int=None,
                # train_frac:float=0.8,
                # train:bool=True,
                idx_list: list,
                max_h:int=360,
                max_w:int=480,
                dh_HR:int=120,
                dw_HR:int=160,
                crop_h:int=750,
                crop_w:int=1000,
                dim_reduction_step:int=3,
                dim_reduction_method:str='bicubic',
                crop:bool=True,
                full:bool=False, #choose either crop or full
                max_val:float=48.0,
                # n_query_points:int=4096,
                max_scale:float=3.0,
                region:str=None,
    ):
      
        self.m0_data_dir, self.m1_data_dir, self.m2_data_dir, self.m3_data_dir = data_dir

        self.m0_file_prefix, self.m1_file_prefix, self.m2_file_prefix, self.m3_file_prefix = file_prefix

        # m0_idx_list = [
        #     idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m0_data_dir+'/*.npy')
        # ]
        # m0_idx_list = list(map(int, m0_idx_list))
        # # self.m0_idx_list.sort()

        # m1_idx_list = [
        #     idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m1_data_dir+'/*.npy')
        # ]
        # m1_idx_list = list(map(int, m1_idx_list))
        # # self.m1_idx_list.sort()

        # m2_idx_list = [
        #     idx.replace(self.m2_data_dir+ '/'+self.m2_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m2_data_dir+'/*.npy')
        # ]
        # m2_idx_list = list(map(int, m2_idx_list))
        # # self.m2_idx_list.sort()

        # m3_idx_list = [
        #     idx.replace(self.m3_data_dir+ '/'+self.m3_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m3_data_dir+'/*.npy')
        # ]
        # m3_idx_list = list(map(int, m3_idx_list))
        # # self.m3_idx_list.sort()

        # self.idx_list = list(set(m0_idx_list) & set(m1_idx_list) & set(m2_idx_list) & set(m3_idx_list))
        # self.idx_list.sort()

        # if total_size is not None:
        #     total_size = min(len(self.idx_list, total_size))
        # else:
        #     total_size = len(self.idx_list)

        # train_size = int(total_size * train_frac)
        # val_size = total_size - train_size

        # if train:
        #     self.idx_list = self.idx_list[:train_size]
        # else:
        #     self.idx_list = self.idx_list[train_size:train_size+val_size]

        self.idx_list = idx_list

        self.crop_h = crop_h
        self.crop_w = crop_w

        self.dim_reduction_method=dim_reduction_method
        
        self.max_h = max_h
        self.max_w = max_w

        self.dh_HR = dh_HR
        self.dw_HR = dw_HR

        self.dh_LR = dh_HR // (2**dim_reduction_step)
        self.dw_LR = dw_HR // (2**dim_reduction_step)

        # self.n_query_points = n_query_points
        self.max_scale = max_scale

        self.crop = crop
        self.full = full

        self.max_val = max_val 

        self.region = region

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))
        m2_tmp = np.load(os.path.join(self.m2_data_dir, self.m2_file_prefix+'_{}.npy'.format(timestep)))
        m3_tmp = np.load(os.path.join(self.m3_data_dir, self.m3_file_prefix+'_{}.npy'.format(timestep)))

        img_HR = np.stack(
                    [
                        m0_tmp,
                        m1_tmp,
                        m2_tmp,
                        m3_tmp
                    ],
                    axis=0
                )
        
        img_HR = torch.FloatTensor(img_HR) #/ self.max_val
        # img_HR -= img_HR.mean(axis=(-2,-1), keepdim=True)
        img_HR /= self.max_val

        # img_HR_max = img_HR.max()
        # img_HR_min = img_HR.min()

        if self.region is not None:
            if self.region == 'A':
                img_HR = img_HR[:,:750,:1000]
            elif self.region == 'B': 
                img_HR = img_HR[:,:750,1000:]
            elif self.region == 'C':
                img_HR = img_HR[:,750:,:1000]
            elif self.region == 'D':
                img_HR = img_HR[:,750:,1000:]
            else:
                pass

        if self.crop:
            img_HR = transforms.RandomCrop((self.crop_h, self.crop_w))(img_HR)
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0),
                                                     size=(self.max_h, self.max_w),
                                                     align_corners=True,
                                                     mode='bicubic').squeeze()
        elif self.full:
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0),
                                                     size=(self.max_h, self.max_w),
                                                     align_corners=True,
                                                     mode='bicubic').squeeze()
        
        # scale = np.random.uniform(low=1.0, high=self.max_scale)

        # ch_HR = round(self.dh_HR * scale)
        # cw_HR = round(self.dw_HR * scale)

        # c_img_HR = transforms.Resize((ch_HR, cw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #transforms.RandomCrop((ch_HR, cw_HR))(img_HR)
        # d_img_HR = transforms.Resize((self.dh_HR, self.dw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #(c_img_HR)

        # if self.dim_reduction_method == 'bicubic':
        #     d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(d_img_HR)
        # elif self.dim_reduction_method == 'bilinear':
        #     d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BILINEAR, antialias=None)(d_img_HR)
        # elif self.dim_reduction_method == 'wavelet-transform':
        #     d_img_LR, *_ = pywt.wavedec2(d_img_HR, 'haar', 'antisymmetric', level=self.dim_reduction_step)
        #     d_img_LR = torch.FloatTensor(d_img_LR)
        # elif self.dim_reduction_method == 'NN':
        #     d_img_LR = d_img_HR
        # else:
        #     raise NotImplementedError

        c_img_HR = img_HR

        # c_img_HR = rearrange(c_img_HR, 
        #                 'm h w -> m (h w)', 
        #                 w=c_img_HR.shape[-1],
        #                 h=c_img_HR.shape[-2],
        #                 m=c_img_HR.shape[0]).permute(1,0)

        # get HR coordinates: h_HR * w_HR, 2
        # grid_HR = get_coords([ch_HR, cw_HR], flatten=False)
        
        # subset of grid_HR and img_HR 
        # grid_HR: n_query_pts, 2 
        # img_HR:  n_query_pts, 1
        # n_query_pts = self.n_query_points
        # query_pts = np.random.choice(len(grid_HR), n_query_pts, replace=False)
        # grid_HR = grid_HR[query_pts]

        if not c_img_HR.is_contiguous():
            c_img_HR = c_img_HR.contiguous()
        # c_img_HR = c_img_HR[query_pts].permute(1,0).unsqueeze(axis=-1)

        # # get cell
        # cell = torch.ones(2).int()
        # cell[0] = ch_HR
        # cell[1] = cw_HR

        return {
                # "d_img_LR": torch.FloatTensor(d_img_LR),
                # "d_img_HR": torch.FloatTensor(d_img_HR),
                "c_img_HR": torch.FloatTensor(c_img_HR),
                # "grid_HR": torch.FloatTensor(grid_HR),
                # "cell": cell
                }
    

class LIIFDatasetv2(Dataset):
    def __init__(self, 
                data_dir:list, 
                file_prefix: list,
                # total_size: int=None,
                # train_frac:float=0.8,
                # train:bool=True,
                idx_list: list,
                max_h:int=360,
                max_w:int=480,
                dh_HR:int=120,
                dw_HR:int=160,
                crop_h:int=750,
                crop_w:int=1000,
                dim_reduction_step:int=3,
                dim_reduction_method:str='bicubic',
                crop:bool=True,
                full:bool=False, #choose either crop or full
                max_val:float=50.0,
                n_query_points:int=4096,
                max_scale:float=5.0,
                region:str=None,
    ):
      
        self.m0_data_dir, self.m1_data_dir, self.m2_data_dir, self.m3_data_dir = data_dir

        self.m0_file_prefix, self.m1_file_prefix, self.m2_file_prefix, self.m3_file_prefix = file_prefix

        # m0_idx_list = [
        #     idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m0_data_dir+'/*.npy')
        # ]
        # m0_idx_list = list(map(int, m0_idx_list))
        # # self.m0_idx_list.sort()

        # m1_idx_list = [
        #     idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m1_data_dir+'/*.npy')
        # ]
        # m1_idx_list = list(map(int, m1_idx_list))
        # # self.m1_idx_list.sort()

        # m2_idx_list = [
        #     idx.replace(self.m2_data_dir+ '/'+self.m2_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m2_data_dir+'/*.npy')
        # ]
        # m2_idx_list = list(map(int, m2_idx_list))
        # # self.m2_idx_list.sort()

        # m3_idx_list = [
        #     idx.replace(self.m3_data_dir+ '/'+self.m3_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m3_data_dir+'/*.npy')
        # ]
        # m3_idx_list = list(map(int, m3_idx_list))
        # # self.m3_idx_list.sort()

        # self.idx_list = list(set(m0_idx_list) & set(m1_idx_list) & set(m2_idx_list) & set(m3_idx_list))
        # self.idx_list.sort()

        # if total_size is not None:
        #     total_size = min(len(self.idx_list, total_size))
        # else:
        #     total_size = len(self.idx_list)

        # train_size = int(total_size * train_frac)
        # val_size = total_size - train_size

        # if train:
        #     self.idx_list = self.idx_list[:train_size]
        # else:
        #     self.idx_list = self.idx_list[train_size:train_size+val_size]

        self.idx_list = idx_list

        self.crop_h = crop_h
        self.crop_w = crop_w

        self.dim_reduction_method=dim_reduction_method
        
        self.max_h = max_h
        self.max_w = max_w

        self.dh_HR = dh_HR
        self.dw_HR = dw_HR

        self.dh_LR = dh_HR // (2**dim_reduction_step)
        self.dw_LR = dw_HR // (2**dim_reduction_step)

        self.n_query_points = n_query_points
        self.max_scale = max_scale

        self.crop = crop
        self.full = full

        self.max_val = max_val 

        self.region = region

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))
        m2_tmp = np.load(os.path.join(self.m2_data_dir, self.m2_file_prefix+'_{}.npy'.format(timestep)))
        m3_tmp = np.load(os.path.join(self.m3_data_dir, self.m3_file_prefix+'_{}.npy'.format(timestep)))

        img_HR = np.stack(
                    [
                        m0_tmp,
                        m1_tmp,
                        m2_tmp,
                        m3_tmp
                    ],
                    axis=0
                )
        
        img_HR = torch.FloatTensor(img_HR) #/ self.max_val
        # img_HR -= img_HR.mean(axis=(-2,-1), keepdim=True)
        img_HR /= self.max_val

        # img_HR_max = img_HR.max()
        # img_HR_min = img_HR.min()

        if self.region is not None:
            if self.region == 'A':
                img_HR = img_HR[:,:750,:1000]
            elif self.region == 'B': 
                img_HR = img_HR[:,:750,1000:]
            elif self.region == 'C':
                img_HR = img_HR[:,750:,:1000]
            elif self.region == 'D':
                img_HR = img_HR[:,750:,1000:]
            else:
                pass

        if self.crop:
            img_HR = transforms.RandomCrop((self.crop_h, self.crop_w))(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0),
                                                     size=(self.max_h, self.max_w), 
                                                     align_corners=True,
                                                     mode='bicubic').squeeze()
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
        elif self.full:
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0),
                                                     size=(self.max_h, self.max_w), 
                                                     align_corners=True,
                                                     mode='bicubic').squeeze()
        
        scale = np.random.uniform(low=1.0, high=self.max_scale)

        ch_HR = round(self.dh_HR * scale)
        cw_HR = round(self.dw_HR * scale)

        # c_img_HR = transforms.Resize((ch_HR, cw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #transforms.RandomCrop((ch_HR, cw_HR))(img_HR)
        # d_img_HR = transforms.Resize((self.dh_HR, self.dw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #(c_img_HR)

        c_img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0), 
                                                   size=(ch_HR, cw_HR),
                                                   align_corners=True,
                                                   mode='bicubic').squeeze()
        d_img_HR = torch.nn.functional.interpolate(c_img_HR.unsqueeze(0),
                                                   size=(self.dh_HR, self.dw_HR),
                                                   align_corners=True,
                                                   mode='bicubic').squeeze()

        if self.dim_reduction_method == 'bicubic':
            # d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(d_img_HR)
            d_img_LR = torch.nn.functional.interpolate(d_img_HR.unsqueeze(0),
                                                       size=(self.dh_LR, self.dw_LR),
                                                       align_corners=True,
                                                       mode='bicubic').squeeze()
        elif self.dim_reduction_method == 'bilinear':
            # d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BILINEAR, antialias=None)(d_img_HR)
            d_img_LR = torch.nn.functional.interpolate(d_img_HR.unsqueeze(0),
                                                       size=(self.dh_LR, self.dw_LR),
                                                       align_corners=True,
                                                       mode='bilinear').squeeze()
        elif self.dim_reduction_method == 'wavelet-transform':
            d_img_LR, *_ = pywt.wavedec2(d_img_HR, 'haar', 'antisymmetric', level=self.dim_reduction_step)
            d_img_LR = torch.FloatTensor(d_img_LR)
        elif self.dim_reduction_method == 'NN':
            d_img_LR = d_img_HR
        else:
            raise NotImplementedError

        c_img_HR = rearrange(c_img_HR, 
                        'm h w -> m (h w)', 
                        w=c_img_HR.shape[-1],
                        h=c_img_HR.shape[-2],
                        m=c_img_HR.shape[0]).permute(1,0)

        # get HR coordinates: h_HR * w_HR, 2
        grid_HR = get_coords([ch_HR, cw_HR])
        
        # subset of grid_HR and img_HR 
        # grid_HR: n_query_pts, 2 
        # img_HR:  n_query_pts, 1
        n_query_pts = self.n_query_points
        query_pts = np.random.choice(len(grid_HR), n_query_pts, replace=False)
        grid_HR = grid_HR[query_pts]

        if not c_img_HR.is_contiguous():
            c_img_HR = c_img_HR.contiguous()
        c_img_HR = c_img_HR[query_pts].permute(1,0).unsqueeze(axis=-1)

        # get cell
        cell = torch.ones(2).int()
        cell[0] = ch_HR
        cell[1] = cw_HR

        return {
                "d_img_LR": torch.FloatTensor(d_img_LR),
                "d_img_HR": torch.FloatTensor(d_img_HR),
                "c_img_HR": torch.FloatTensor(c_img_HR),
                "grid_HR": torch.FloatTensor(grid_HR),
                "cell": cell
                }
    

class LIIFDatasetVizv2(Dataset):
    def __init__(self, 
                data_dir:list, 
                file_prefix: list,
                idx_list: list,
                # total_size: int=None,
                # train_frac:float=0.8,
                # train:bool=True,
                max_h:int=360,
                max_w:int=480,
                dh_HR:int=120,
                dw_HR:int=160,
                crop_h:int=750,
                crop_w:int=1000,
                dim_reduction_step:int=3,
                dim_reduction_method:str='bicubic',
                crop:bool=True,
                full:bool=False, #choose either crop or full
                max_val:float=48.0,
                res_scale:float=3.0,
                region:str=None,
    ):
      
        self.m0_data_dir, self.m1_data_dir, self.m2_data_dir, self.m3_data_dir = data_dir

        self.m0_file_prefix, self.m1_file_prefix, self.m2_file_prefix, self.m3_file_prefix = file_prefix

        # m0_idx_list = [
        #     idx.replace(self.m0_data_dir+ '/'+self.m0_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m0_data_dir+'/*.npy')
        # ]
        # m0_idx_list = list(map(int, m0_idx_list))
        # # self.m0_idx_list.sort()

        # m1_idx_list = [
        #     idx.replace(self.m1_data_dir+ '/'+self.m1_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m1_data_dir+'/*.npy')
        # ]
        # m1_idx_list = list(map(int, m1_idx_list))
        # # self.m1_idx_list.sort()

        # m2_idx_list = [
        #     idx.replace(self.m2_data_dir+ '/'+self.m2_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m2_data_dir+'/*.npy')
        # ]
        # m2_idx_list = list(map(int, m2_idx_list))
        # # self.m2_idx_list.sort()

        # m3_idx_list = [
        #     idx.replace(self.m3_data_dir+ '/'+self.m3_file_prefix+'_', "").replace('.npy', "") 
        #     for idx in glob.glob(self.m3_data_dir+'/*.npy')
        # ]
        # m3_idx_list = list(map(int, m3_idx_list))
        # # self.m3_idx_list.sort()

        # self.idx_list = list(set(m0_idx_list) & set(m1_idx_list) & set(m2_idx_list) & set(m3_idx_list))
        # self.idx_list.sort()

        # if total_size is not None:
        #     total_size = min(len(self.idx_list, total_size))
        # else:
        #     total_size = len(self.idx_list)

        # train_size = int(total_size * train_frac)
        # val_size = total_size - train_size

        # if train:
        #     self.idx_list = self.idx_list[:train_size]
        # else:
        #     self.idx_list = self.idx_list[train_size:train_size+val_size]

        self.idx_list = idx_list

        self.crop_h = crop_h
        self.crop_w = crop_w

        self.dim_reduction_method=dim_reduction_method
        self.dim_reduction_step=dim_reduction_step

        self.max_h = max_h
        self.max_w = max_w

        self.ch_HR = int(res_scale*dh_HR)
        self.cw_HR = int(res_scale*dw_HR)

        self.dh_HR = dh_HR
        self.dw_HR = dw_HR

        self.dh_LR = dh_HR // (2**dim_reduction_step)
        self.dw_LR = dw_HR // (2**dim_reduction_step)

        self.crop = crop
        self.full = full

        self.max_val = max_val

        self.region = region

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        timestep = self.idx_list[idx]

        m0_tmp = np.load(os.path.join(self.m0_data_dir, self.m0_file_prefix+'_{}.npy'.format(timestep)))
        m1_tmp = np.load(os.path.join(self.m1_data_dir, self.m1_file_prefix+'_{}.npy'.format(timestep)))
        m2_tmp = np.load(os.path.join(self.m2_data_dir, self.m2_file_prefix+'_{}.npy'.format(timestep)))
        m3_tmp = np.load(os.path.join(self.m3_data_dir, self.m3_file_prefix+'_{}.npy'.format(timestep)))

        img_HR = np.stack(
                    [
                        m0_tmp,
                        m1_tmp,
                        m2_tmp,
                        m3_tmp
                    ],
                    axis=0
                )
        
        img_HR = torch.FloatTensor(img_HR) #/ self.max_val
        # img_HR -= img_HR.mean(axis=(-2,-1), keepdim=True)
        img_HR /= self.max_val

        # img_HR_max = img_HR.max()
        # img_HR_min = img_HR.min()

        if self.region is not None:
            if self.region == 'A':
                img_HR = img_HR[:,:750,:1000]
            elif self.region == 'B': 
                img_HR = img_HR[:,:750,1000:]
            elif self.region == 'C':
                img_HR = img_HR[:,750:,:1000]
            elif self.region == 'D':
                img_HR = img_HR[:,750:,1000:]
            else:
                pass

        if self.crop:
            img_HR = transforms.RandomCrop((self.crop_h, self.crop_w))(img_HR)
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqeeze(0),
                                                     size=(self.max_h, self.max_w),
                                                     align_corners=True,
                                                     mode='bicubic').squueze()
        elif self.full:
            # img_HR = transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR)
            img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0),
                                                     size=(self.max_h, self.max_w),
                                                     align_corners=True,
                                                     mode='bicubic').squeeze()

        # c_img_HR = transforms.Resize((self.ch_HR, self.cw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #transforms.RandomCrop((ch_HR, cw_HR))(img_HR)
        # d_img_HR = transforms.Resize((self.dh_HR, self.dw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #(c_img_HR)

        c_img_HR = torch.nn.functional.interpolate(img_HR.unsqueeze(0), 
                                                   size=(self.ch_HR, self.cw_HR),
                                                   align_corners=True,
                                                   mode='bicubic').squeeze()
        d_img_HR = torch.nn.functional.interpolate(c_img_HR.unsqueeze(0),
                                                   size=(self.dh_HR, self.dw_HR),
                                                   align_corners=True,
                                                   mode='bicubic').squeeze()

        if self.dim_reduction_method == 'bicubic':
            # d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(d_img_HR)
            d_img_LR = torch.nn.functional.interpolate(d_img_HR.unsqueeze(0),
                                                       size=(self.dh_LR, self.dw_LR),
                                                       align_corners=True,
                                                       mode='bicubic').squeeze()
        elif self.dim_reduction_method == 'bilinear':
            # d_img_LR = transforms.Resize((self.dh_LR, self.dw_LR), interpolation=InterpolationMode.BILINEAR, antialias=None)(d_img_HR)
            d_img_LR = torch.nn.functional.interpolate(d_img_HR.unsqueeze(0),
                                                       size=(self.dh_LR, self.dw_LR),
                                                       align_corners=True,
                                                       mode='bilinear').squeeze()
        elif self.dim_reduction_method == 'wavelet-transform':
            d_img_LR, *_ = pywt.wavedec2(d_img_HR, 'haar', 'antisymmetric', level=self.dim_reduction_step)
            d_img_LR = torch.FloatTensor(d_img_LR)
        elif self.dim_reduction_method == 'NN':
            d_img_LR = d_img_HR
        else:
            raise NotImplementedError

        c_img_HR = rearrange(c_img_HR, 
                        'm h w -> m (h w)', 
                        w=c_img_HR.shape[-1],
                        h=c_img_HR.shape[-2],
                        m=c_img_HR.shape[0]).permute(1,0)

        # get HR coordinates: h_HR * w_HR, 2
        grid_HR = get_coords([self.ch_HR, self.cw_HR])

        if not c_img_HR.is_contiguous():
            c_img_HR = c_img_HR.contiguous()
        c_img_HR = c_img_HR.permute(1,0).unsqueeze(axis=-1)

        # get cell
        cell = torch.ones(2).int()
        cell[0] = self.ch_HR
        cell[1] = self.cw_HR

        return {
                "d_img_LR": torch.FloatTensor(d_img_LR),
                "d_img_HR": torch.FloatTensor(d_img_HR),
                "c_img_HR": torch.FloatTensor(c_img_HR),
                "grid_HR": torch.FloatTensor(grid_HR),
                "cell": cell
                }
    
def batch_processing(
        img_HR:torch.tensor, 
        max_scale:float=3.0, 
        dh_HR:int=120,
        dw_HR:int=160,
        dh_LR:int=15,
        dw_LR:int=20,
        dim_reduction_step:int=3,
        dim_reduction_method:str='bicubic',
        scale:float=None
    ):
    if scale is None:
        scale = np.random.uniform(low=1.0, high=max_scale)

    ch_HR = round(dh_HR * scale)
    cw_HR = round(dw_HR * scale)

    b = img_HR.shape[0]

    c_img_HR = transforms.Resize((ch_HR, cw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #transforms.RandomCrop((ch_HR, cw_HR))(img_HR)
    d_img_HR = transforms.Resize((dh_HR, dw_HR), interpolation=InterpolationMode.BICUBIC, antialias=None)(img_HR) #(c_img_HR)

    if dim_reduction_method == 'bicubic':
        d_img_LR = transforms.Resize((dh_LR, dw_LR), interpolation=InterpolationMode.BICUBIC, antialias=None)(d_img_HR)
    elif dim_reduction_method == 'bilinear':
        d_img_LR = transforms.Resize((dh_LR, dw_LR), interpolation=InterpolationMode.BILINEAR, antialias=None)(d_img_HR)
    elif dim_reduction_method == 'wavelet-transform':
        d_img_LR, *_ = pywt.wavedec2(d_img_HR, 'haar', 'antisymmetric', level=dim_reduction_step)
        d_img_LR = torch.FloatTensor(d_img_LR)
    elif dim_reduction_method == 'NN':
        d_img_LR = d_img_HR
    else:
        raise NotImplementedError
    
    # get HR coordinates: h_HR * w_HR, 2
    grid_HR = get_coords([ch_HR, cw_HR], flatten=False).unsqueeze(0).repeat(b, 1, 1, 1)

    # get cell
    cell = torch.ones(2).int()
    cell[0] = ch_HR
    cell[1] = cw_HR

    cell = cell.unsqueeze(0).repeat(b, 1)

    return c_img_HR, d_img_HR, grid_HR, cell