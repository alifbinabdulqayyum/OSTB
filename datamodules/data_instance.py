import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
from typing import Optional
from hydra.utils import log


class ZscoreStandardizer(object):
    '''
    Normalization transformation
    '''
    def __init__(self, x):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
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
        self.minVal = torch.min(x)
        self.maxVal = torch.max(x)
        self.epsilon = 1e-10
        assert self.minVal.shape == self.maxVal.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minVal) / (self.maxVal - self.minVal) + self.epsilon

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.epsilon) * (self.maxVal - self.minVal) + self.minVal


class DataInstanceModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str = 'data/wind/ua_0.npy',
        pre_method: str = 'zscore',
        ratio_HL: int = 10,
        n_train_val_test: list = [1, 1, 1],
        b_train_val_test: list = [1, 1, 1],
    ):
        super().__init__()

        self.data_dir = data_dir
        self.pre_method = pre_method
        self.ratio_HL = ratio_HL
        self.n_train, self.n_val, self.n_test = n_train_val_test
        self.b_train, self.b_val, self.b_test = b_train_val_test        

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # define HR and LR sample
        img_HR = torch.from_numpy(np.load(self.data_dir))
        img_LR = torch.unsqueeze(img_HR, 0)
        img_LR = F.resize(img_LR, (img_HR.shape[0]//self.ratio_HL, img_HR.shape[1]//self.ratio_HL), interpolation=InterpolationMode.BICUBIC)
        img_LR = torch.squeeze(img_LR)

        # get HR and LR coordinates
        grid_HR = self._get_coordinates(img_HR)
        grid_LR = self._get_coordinates(img_LR)

        # get training, validation, and test datasets
        # x      - grid - ntrain * H * W * 2
        # target - img  - ntrain * H * W * 1
        grid_LR = torch.unsqueeze(grid_LR, 0)
        grid_HR = torch.unsqueeze(grid_HR, 0)
        img_LR = torch.unsqueeze(torch.unsqueeze(img_LR, 0), 3)
        img_HR = torch.unsqueeze(torch.unsqueeze(img_HR, 0), 3)

        self.train_data = TensorDataset(grid_LR, img_LR)
        self.val_data = TensorDataset(grid_LR, img_LR)
        self.test_data = TensorDataset(grid_HR, img_HR)

        # get normalizer for training and test data
        normalizer = {}
        normalizer_train = self._get_normalizer(img_LR)
        normalizer_test = self._get_normalizer(img_HR)
        normalizer['normalizer_train'] = normalizer_train
        normalizer['normalizer_test'] = normalizer_test
        return normalizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data, batch_size=self.b_val, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=self.b_test, shuffle=False)

    def _get_coordinates(self, img):
        '''
        Create input pixel coordinates in the unit square
        Args:
            img  (torch.tensor)  -  (H, W)
        Returns:
            grid (torch.tensor)  -  (H, W, 2)
        '''
        h_axis = torch.linspace(0, 1, steps=img.shape[0])
        w_axis = torch.linspace(0, 1, steps=img.shape[1])
        grid = torch.stack(torch.meshgrid(h_axis, w_axis), dim=-1)
        return grid

    def _get_normalizer(self, img):
        '''
        Create normalizer for the sample
        '''
        if self.pre_method == 'zscore':
            log.info(f"Preprocessing method: {self.pre_method}")
            return ZscoreStandardizer(img)
        elif self.pre_method == 'minmax':
            log.info(f"Preprocessing method: {self.pre_method}")
            return MinMaxStandardizer(img)
        else:
            log.info("Preprocessing method: None")
            return None