import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions

from utils.misc import get_rank

from utils import *

import imageio
import scipy.io
import cv2

class SLDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.use_mask = True

        if self.config.category == "captured-geoalbamb":
            cam_mask = imageio.imread(os.path.join(self.config.root_dir, "mask-undistorted.png")) / 255.0
            H, W = cam_mask.shape
            pro_mask = np.ones((H, W)) * -1
            pro_mask[720:, :] = 0

            print(f'image shape: H{H} W{W}')

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = W // self.config.img_downscale, H // self.config.img_downscale
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            self.w, self.h = w, h
            self.img_wh = (self.w, self.h)
            self.near, self.far = self.config.near_plane, self.config.far_plane

            # laod calibrated intrinsic paramters & pro-cam extrinsics
            calibration_data = scipy.io.loadmat(os.path.join(self.config.root_dir, "paramcv.mat"))
            self.fx_cam = calibration_data['mtxl'][0, 0]
            self.fy_cam = calibration_data['mtxl'][1, 1]
            self.cx_cam = calibration_data['mtxl'][0, 2]
            self.cy_cam = calibration_data['mtxl'][1, 2]

            self.fx_pro = calibration_data['mtxp'][0, 0]
            self.fy_pro = calibration_data['mtxp'][1, 1]
            self.cx_pro = calibration_data['mtxp'][0, 2]
            self.cy_pro = calibration_data['mtxp'][1, 2]
            # ray directions for all pixels in camera and projector image plane
            # (2, h, w, 3)
            self.directions = torch.stack([
                get_ray_directions(self.w, self.h, self.fx_cam, self.fy_cam, self.cx_cam, self.cy_cam,
                                   use_pixel_centers=self.config.use_pixel_centers, OPENGL=self.config.opengl),
                get_ray_directions(self.w, self.h, self.fx_pro, self.fy_pro, self.cx_pro, self.cy_pro,
                                   use_pixel_centers=self.config.use_pixel_centers, OPENGL=self.config.opengl)
            ], axis=0).to(self.rank)

            num_patterns = int(self.config.scene.split('_')[2][1:])
            print(f"num_patters: {num_patterns}")
            if num_patterns != self.config.num_patterns:
                raise Exception("mismatch between number of pattern and data directory!")

            # load images and projected patterns
            images = np.zeros((self.h, self.w, num_patterns))
            patterns = np.zeros((self.h, self.w, num_patterns))
            image_dir = os.path.join(self.config.root_dir, "images")
            pattern_dir = os.path.join(self.config.root_dir, "patterns")

            for k in range(num_patterns):
                patterns[:720, :, k] = imageio.imread(os.path.join(pattern_dir,
                                                           f"pt{k}-ldr.png"))[:720, :] / 255.0
                patterns[pro_mask == 0, k] = 0
                for kx in range(self.config.avg_itr):
                    images[..., k] = images[..., k] + imageio.imread(os.path.join(image_dir, f"scene_images/im{k}-itr{kx}-cubic-undistorted.png")) / 255.0
                images[..., k] = images[..., k] / self.config.avg_itr
                images[cam_mask == 0, k] = 0

            self.all_images = torch.from_numpy(np.stack([torch.from_numpy(images), torch.from_numpy(patterns)], axis=0)).\
                float().to(self.rank)
            self.all_fg_masks = torch.from_numpy(np.stack([torch.from_numpy(cam_mask), torch.from_numpy(pro_mask)], axis=0)).\
                float().to(self.rank)

            if self.config.noise:
                img = Image.open(os.path.join(image_dir, f"scene_images/white_noisy.png"))
                white = TF.to_tensor(img)[0, ...]
                white[cam_mask == 0] = 0

                img = Image.open(os.path.join(image_dir, f"scene_images/black_noisy.png"))
                black = TF.to_tensor(img)[0, ...]
                black[cam_mask == 0] = 0

                self.all_albedo = torch.stack([white-black, torch.ones_like(white)], dim=0).float().to(self.rank)
                self.all_ambient = torch.stack([black, torch.zeros_like(black)], dim=0).float().to(self.rank)

            else:
                temp = 0
                for kx in range(self.config.avg_itr):
                    temp = temp + imageio.imread(os.path.join(image_dir, f"scene_images/white-itr{kx}-cubic-undistorted.png")) / 255.0

                img = temp / self.config.avg_itr
                white = TF.to_tensor(img)[0, ...]
                white[cam_mask == 0] = 0

                temp = 0
                for kx in range(self.config.avg_itr):
                    if self.config.scene_object == 'david' and self.config.capture_id not in ['davidExp2', 'davidExp1']:
                        temp = np.zeros_like(cam_mask)
                    elif self.config.scene_object in ['face', 'mesh']:
                        temp = temp + imageio.imread(os.path.join(image_dir, f"scene_images/black-itr{kx}-undistorted.png")) / 255.0
                    else:
                        temp = temp + imageio.imread(os.path.join(image_dir, f"scene_images/black-itr{kx}-cubic-undistorted.png")) / 255.0

                img = temp / self.config.avg_itr
                black = TF.to_tensor(img)[0, ...]
                black[cam_mask == 0] = 0

                self.all_albedo = torch.stack([white-black, torch.ones_like(white)], dim=0).float().to(self.rank)
                self.all_ambient = torch.stack([black, torch.zeros_like(black)], dim=0).float().to(self.rank)
                self.all_albedo = torch.stack([torch.ones_like(white), torch.ones_like(white)], dim=0).float().to(self.rank)
                self.all_ambient = torch.stack([torch.zeros_like(black), torch.zeros_like(black)], dim=0).float().to(self.rank)


            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
            image_rot = np.eye(3)

            # scale and shift to fit scene in render volume
            image_tvec = np.array([0, 0.015, 0.65]).reshape((3, 1))
            scaling = np.eye(4)
            scaling[0, 0] = 0.125
            scaling[1, 1] = 0.125
            scaling[2, 2] = 0.125
            image_w2c = np.concatenate([np.concatenate([image_rot, image_tvec], 1), bottom], axis=0) @ scaling

            pattern_rot = calibration_data['Rlp']
            pattern_tvec = calibration_data['Tlp'] * 0.001
            pattern_w2c = np.concatenate([np.concatenate([pattern_rot, pattern_tvec], 1), bottom], axis=0) @ image_w2c

            w2c_mats = np.stack([image_w2c, pattern_w2c], axis=0)
            camtoworlds = np.linalg.inv(w2c_mats)

            self.all_w2c = torch.from_numpy(w2c_mats[:, :3, :4]).float().to(self.rank)
            self.all_c2w = torch.from_numpy(camtoworlds[:, :3, :4]).float().to(self.rank)

            depth_cam = np.ones((1024, 1280)) * (0)

            depth_cam[cam_mask == 0] = 0

            depth_proj = np.ones((1024, 1280)) * (-1)
            self.all_depth = torch.from_numpy(np.stack([depth_cam, depth_proj], axis=0)).float().to(self.rank)
           

class SLDataset(Dataset, SLDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class SLIterableDataset(IterableDataset, SLDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('sl')
class SLDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = SLIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = SLDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = SLDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = SLDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
