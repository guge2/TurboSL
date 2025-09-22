import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy


@systems.register('neus-sl-system-geometry-albedo-ambient-wz-camproblur')
class NeuSSLWZCamproblurSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'],
                          self.dataset.all_w2c, self.dataset.all_c2w, self.dataset.all_images,
                          self.dataset.fx_pro, self.dataset.fy_pro, self.dataset.cx_pro, self.dataset.cy_pro,
                          self.dataset.fx_cam, self.dataset.fy_cam, self.dataset.cx_cam, self.dataset.cy_cam,
                          self.global_step)

    def preprocess_data(self, batch, stage):
        if 'index' in batch:  # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,),
                                      device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]

            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)

            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])

            projims = self.dataset.all_images[1:2]
            projims_1xkxhxw = projims.permute(0, 3, 1, 2)
            projims_blur_1xkxhxw = self.model.blur.apply_blur(projims_1xkxhxw)
            projims_blur_1xhxwxk = projims_blur_1xkxhxw.permute(0, 2, 3, 1)
            all_images = torch.cat([self.dataset.all_images[0:1], projims_blur_1xhxwxk], dim=0)
            rgb = all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])


            depth = self.dataset.all_depth[index, y, x].view(-1)
            albedo = self.dataset.all_albedo[index, y, x].view(-1)
            ambient = self.dataset.all_ambient[index, y, x].view(-1)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1)
        else:
            c2w = self.dataset.all_c2w[index][0]
            directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            projims = self.dataset.all_images[1:2]
            projims_1xkxhxw = projims.permute(0, 3, 1, 2)
            with torch.no_grad():
                projims_blur_1xkxhxw = self.model.blur.apply_blur(projims_1xkxhxw)
            projims_blur_1xhxwxk = projims_blur_1xkxhxw.permute(0, 2, 3, 1)
            all_images = torch.cat([self.dataset.all_images[0:1], projims_blur_1xhxwxk], dim=0)
            rgb = all_images[index].view(-1, self.dataset.all_images.shape[-1])
            depth = self.dataset.all_depth[index].view(-1)
            albedo = self.dataset.all_albedo[index].view(-1)
            ambient = self.dataset.all_ambient[index].view(-1)
            fg_mask = self.dataset.all_fg_masks[index].view(-1)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((self.config.dataset.num_patterns,), dtype=torch.float32,
                                                         device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((self.config.dataset.num_patterns,), dtype=torch.float32,
                                                         device=self.rank)
            elif self.config.model.background_color == 'black':
                self.model.background_color = torch.zeros((self.config.dataset.num_patterns,), dtype=torch.float32,
                                                          device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.zeros((self.config.dataset.num_patterns,),
                                                      dtype=torch.float32, device=self.rank)

        rgb = rgb * torch.abs(fg_mask[..., None]) + self.model.background_color * (1 - torch.abs(fg_mask[..., None]))

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'albedo': albedo,
            'ambient': ambient,
            'fg_mask': fg_mask,
            'depth': depth,
        })

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            if out['num_samples'] != 0:
                train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                                              self.config.model.max_train_num_rays)
            else:
                print('!!!!zeros samples!!!!')
                print(f'train_num_rays: {self.train_num_rays} -----  num_samples: {out["num_samples"]}')
                self.train_num_rays = self.config.model.max_train_num_rays

        if self.global_step < 1000:
            alpha1 = self.config.system.loss.alpha1
            alpha2 = 0
        else:
            alpha1 = self.config.system.loss.alpha1
            alpha2 = self.config.system.loss.alpha2
        
        loss_rgb_mse_cam = \
            alpha1 * \
        F.mse_loss(out['comp_rgb_proj_albedo'][batch['depth'] != -1, :],
           batch['rgb'][batch['depth'] != -1, :])
       
        difference = out['comp_rgb_cam_albedo'][batch['depth'] == -1, :] - batch['rgb'][batch['depth'] == -1, :]
        if self.config.system.loss.alpha1 == 0 and self.config.system.loss.alpha2>0:
            difference_scaled = difference * 0.05
        else:
            difference_scaled = difference * out['comp_alb_cam'][batch['depth'] == -1, :]
        loss_rgb_mse_proj = \
            alpha2 * \
                F.mse_loss(difference_scaled, torch.zeros_like(difference_scaled))
        self.log('train/loss_rgb_mse_cam', loss_rgb_mse_cam)
        self.log('train/loss_rgb_mse_proj', loss_rgb_mse_proj)

        loss += loss_rgb_mse_cam * self.C(self.config.system.loss.lambda_rgb_mse)
        loss += loss_rgb_mse_proj * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1_cam = \
            alpha1 * \
        F.l1_loss(out['comp_rgb_proj_albedo'][batch['depth'] != -1, :],
           batch['rgb'][batch['depth'] != -1, :])

        loss_rgb_l1_proj = \
            alpha2 * \
                F.l1_loss(difference_scaled, torch.zeros_like(difference_scaled))

        self.log('train/loss_rgb_l1_cam', loss_rgb_l1_cam)
        self.log('train/loss_rgb_l1_proj', loss_rgb_l1_proj)

        loss += loss_rgb_l1_cam * self.C(self.config.system.loss.lambda_rgb_l1)
        loss += loss_rgb_l1_proj * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.) ** 2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        opacity = torch.clamp(out['opacity'], 1.e-3, 1. - 1.e-3)
        loss_mask_opacity = binary_cross_entropy(opacity[batch['fg_mask'] != -1], batch['fg_mask'][batch['fg_mask'] != -1].float())
        loss_mask = loss_mask_opacity
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.use_mask else 0.0)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):

        out = self(batch)

        psnr = self.criterions['psnr'](out['comp_rgb_proj'], batch['rgb'])
        W, H = self.dataset.img_wh
        depth_range = (4, 6)
        depth_range_proj = (5, 7)
        depth_error_range = (0, 0.05)

        for i in range(1):
            if batch['index'][0].item() == 0:
                self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}_{i}.png", [
                    {'type': 'grayscale', 'img': (batch['rgb']).view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': (out['comp_rgb_proj_albedo']).view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': torch.mean(
                        torch.abs((out['comp_rgb_proj_albedo']) - (batch['rgb'])).view(H, W, self.config.dataset.num_patterns), 2, False),
                     'kwargs': {'data_range': (0, 0.03)}},
                    {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'data_range': depth_range}},
                    {'type': 'grayscale',
                     'img': torch.abs(out['depth'] - batch['depth']).view(H, W),
                     'kwargs': {'data_range': depth_error_range}},
                    {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])
            else:
                self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}_{i}.png", [
                    {'type': 'grayscale',
                     'img': (batch['rgb']).view(H, W, self.config.dataset.num_patterns)[:, :,
                            i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale',
                     'img': out['comp_rgb_cam_albedo'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': torch.mean(
                        torch.abs(batch['rgb'] - out['comp_rgb_cam_albedo']).view(H, W, self.config.dataset.num_patterns), 2, False),
                     'kwargs': {'data_range': (0, 0.05)}},
                    {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3),
                     'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'data_range': depth_range_proj}},
                    {'type': 'grayscale',
                     'img': torch.abs(out['depth'] - batch['depth']).view(H, W),
                     'kwargs': {'data_range':  depth_error_range}},
                    {'type': 'grayscale', 'img': out['opacity'].view(H, W),
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])

        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}_appearance.png", [
            {'type': 'grayscale', 'img': out['comp_cosine'].view(H, W) * batch['fg_mask'].view(H, W),
             'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': out['comp_cosine_reflectance'].view(H, W) * batch['fg_mask'].view(H, W),
             'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'data_range': depth_range, 'cmap': 'magma'}},
            {'type': 'grayscale',
             'img': torch.abs(out['depth'] - batch['depth']).view(H, W),
             'kwargs': {'data_range': depth_error_range}},
        ])

        return {
            'psnr': psnr,
            'index': batch['index']
        }

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_proj'], batch['rgb'])
        W, H = self.dataset.img_wh
        depth_range = (4, 6)
        depth_range_proj = (5, 7)
        depth_error_range = (0, 0.05)

        for i in range(self.config.dataset.num_patterns):
            if batch['index'][0].item() == 0:
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}.png", [
                    {'type': 'grayscale', 'img': batch['rgb'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': out['comp_rgb_proj'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'data_range': depth_range}},
                    {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_albedo.png", [
                    {'type': 'grayscale', 'img': out['comp_cosine_reflectance'].view(H, W)*batch['fg_mask'].view(H, W),
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_cosine.png", [
                    {'type': 'grayscale', 'img': out['comp_cosine'].view(H, W)*batch['fg_mask'].view(H, W),
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_normal.png", [
                    {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3),
                     'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_image_recon.png", [
                    {'type': 'grayscale',
                     'img': out['comp_rgb_proj_albedo'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_image_recon_albinv.png", [
                    {'type': 'grayscale', 'img': out['comp_rgb_proj'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                ])

                self.save_data(f'data/normal_{self.global_step}.npy',
                               out['comp_normal'].view(H, W, 3))
                self.save_data(f'data/camera_depth_{self.global_step}.npy', out['depth'].view(H, W))
                self.save_data(f'data/surface_points_{self.global_step}.npy',
                               out['surface_points_proj_output'].view(H, W, 3))
                self.save_data(f'data/image_error_{self.global_step}.npy',
                        torch.abs((out['comp_rgb_proj_albedo']) - (batch['rgb'])).view(H, W, self.config.dataset.num_patterns))
                self.save_data(f'data/reconstructed_image_{self.global_step}.npy', out['comp_rgb_proj_albedo'].view(H, W, self.config.dataset.num_patterns))

            else:
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}.png", [
                    {'type': 'grayscale', 'img': batch['rgb'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': out['comp_rgb_cam'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'data_range': depth_range_proj}},
                    {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
                ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}_{i}_pattern_recon.png", [
                    {'type': 'grayscale',
                     'img': out['comp_rgb_cam_albedo'].view(H, W, self.config.dataset.num_patterns)[:, :, i],
                     'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                ])
                self.save_data(f'data/pattern_error_{self.global_step}.npy',
                        torch.abs((out['comp_rgb_cam_albedo']) - (batch['rgb'])).view(H, W, self.config.dataset.num_patterns))
                self.save_data(f'data/reconstructed_pattern_{self.global_step}.npy', out['comp_rgb_cam_albedo'].view(H, W, self.config.dataset.num_patterns))
        return {
            'psnr': psnr,
            'index': batch['index']
        }

    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )

            mesh = self.model.isosurface()
            self.save_mesh(
                f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
                mesh['v_pos'],
                mesh['t_pos_idx'],
            )

            points, sdf_values = self.model.output_sdf(100)
            self.save_data(f'data/sdf_values_{self.global_step}.npy', sdf_values)
            self.save_data(f'data/sdf_points_{self.global_step}.npy', points)