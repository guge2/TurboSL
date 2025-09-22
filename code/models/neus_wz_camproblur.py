import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_alpha, accumulate_along_rays
from models.neus import VarianceNetwork

@models.register('neus-geometry-albedo-ambient-wz-camproblur')
class NeuSWZCamproblurModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.albedo = models.make(self.config.albedo.name, self.config.albedo)
        self.ambient = models.make(self.config.ambient.name, self.config.ambient)
        self.variance = VarianceNetwork(self.config.variance)

        self.register_buffer('scene_aabb', torch.as_tensor(
            [-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius,
             self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        self.blur = models.make(self.config.blur.name, self.config.blur)

    def update_step(self, epoch, global_step):
        update_module_step(self.variance, epoch, global_step)
        update_module_step(self.blur, epoch, global_step)
        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[..., None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[..., None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,
                                             occ_thre=self.config.get('grid_prune_occ_thre', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    @torch.no_grad()
    def output_sdf(self, grid_resolution):
        x_range = torch.linspace(-self.config.radius, self.config.radius, grid_resolution).to(0)
        y_range = torch.linspace(-self.config.radius, self.config.radius, grid_resolution).to(0)
        z_range = torch.linspace(-self.config.radius, self.config.radius, grid_resolution).to(0)
        mesh_grid = torch.tensor(list(product(x_range, y_range, z_range)), dtype=torch.float32).to(0)
        sdf_values = self.geometry(mesh_grid, with_grad=False, with_feature=False)
        return mesh_grid, sdf_values


    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)
        true_cos = (dirs * normal).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive
        # Estimate signed distances at section points
        estimated_next_sdf = sdf[..., None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[..., None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays,
                 all_w2c=None, all_c2w= None, all_images=None,
                 fx_pro=None, fy_pro=None, cx_pro=None, cy_pro=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None,
                 step=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

        with torch.no_grad():
            cam_o = all_c2w[0, :, 3]
            proj_o = all_c2w[1, :, 3]

            if self.config.back_rendering:
                rays_o = rays_o + 10 * rays_d
                rays_d = -rays_d
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        sdf, sdf_grad = self.geometry(positions, with_grad=True, with_feature=False)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[..., None]

        ########################
        w2c_proj = all_w2c[1, ...]
        w2c_cam = all_w2c[0, ...]

        K_proj = np.array([[fx_pro, 0, cx_pro],
                           [0, fy_pro, cy_pro],
                           [0, 0, 1]])
        K_proj = torch.from_numpy(K_proj).float().to(0)

        K_cam = np.array([[fx_cam, 0, cx_cam],
                           [0, fy_cam, cy_cam],
                           [0, 0, 1]])
        K_cam = torch.from_numpy(K_cam).float().to(0)


        images_proj_original = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

        images_proj_blur = self.blur(images_proj_original)
        ########################
        # radiance estimated from projection onto projector image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_proj, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj = torch.mm(K_proj, temp)
        else:
            positions_proj = torch.mm(K_proj, torch.mm(w2c_proj, positions_hom))

        positions_proj = positions_proj / positions_proj[2, :]
        positions_proj[0, :] = (positions_proj[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj[1, :] = (positions_proj[1, :] / all_images.shape[1]) * 2 - 1
        positions_proj = positions_proj[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj = F.grid_sample(images_proj_blur, positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj = rgb_proj.squeeze(0).squeeze(1).T

        # radiance estimated from projection onto camera image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_cam = torch.mm(K_cam, temp)
        else:
            positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

        positions_cam = positions_cam / positions_cam[2, :]
        positions_cam[0, :] = (positions_cam[0, :] / all_images.shape[2]) * 2 - 1
        positions_cam[1, :] = (positions_cam[1, :] / all_images.shape[1]) * 2 - 1
        positions_cam = positions_cam[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_cam = F.grid_sample(images_cam, positions_cam, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_cam = rgb_cam.squeeze(0).squeeze(1).T
        ########################
        # compute the ray estimates including the depth
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)

        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        comp_rgb_proj = accumulate_along_rays(weights, ray_indices, values=rgb_proj, n_rays=n_rays)
        comp_rgb_proj = comp_rgb_proj + self.background_color * (1.0 - opacity)

        comp_rgb_cam = accumulate_along_rays(weights, ray_indices, values=rgb_cam, n_rays=n_rays)
        comp_rgb_cam = comp_rgb_cam + self.background_color * (1.0 - opacity)

        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

        ########################
        # compute the surface points for each ray
        surface_points = rays_o + depth[:, None] * rays_d
        surface_points_hom = torch.cat((surface_points, torch.ones_like(depth).unsqueeze(-1)), dim=1).T

        # image rgb computed from the depth estimate and projector image
        if self.config.opengl:
            temp = torch.mm(w2c_proj, surface_points_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            surface_points_proj = torch.mm(K_proj, temp)
        else:
            surface_points_proj = torch.mm(K_proj, torch.mm(w2c_proj, surface_points_hom))
        surface_points_proj = surface_points_proj / surface_points_proj[2, :]
        surface_points_proj_output = surface_points_proj.T

        ################
        # compute albedo and ambient along rays
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_cam = torch.mm(K_cam, temp)
        else:
            positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

        positions_cam = positions_cam / positions_cam[2, :]
        positions_cam[0, :] = (positions_cam[0, :] / all_images.shape[2])
        positions_cam[1, :] = (positions_cam[1, :] / all_images.shape[1])
        if positions_cam.shape[1] > 0:
            positions_alb = self.albedo(positions_cam[:2, :].T)
            positions_amb = self.ambient(positions_cam[:2, :].T)
        else:
            zeros_tensor = torch.zeros_like(positions_cam[:1, :0])
            positions_alb = zeros_tensor.T
            positions_amb = zeros_tensor.T

#########################
        # ray accumulation image formation
        if self.config.back_rendering:
            normal = -normal

        comp_alb_cam = accumulate_along_rays(weights, ray_indices,
                                                    values=torch.clamp(positions_alb * torch.cosine_similarity(normal, proj_o - positions,
                                                                                    dim=1)[:, None], min=0.1, max=1),
                                                    n_rays=n_rays)
        comp_alb_cam = comp_alb_cam + self.background_color * (1.0 - opacity.unsqueeze(-1))

##########################
        comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                                                    values=(rgb_cam-positions_amb)/torch.clamp(positions_alb *
                                                            torch.cosine_similarity(normal, proj_o - positions,
                                                                                    dim=1)[:, None], min=0.1,
                                                            max=1),
                                                    n_rays=n_rays)
        comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
                                                    values=rgb_proj*torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None] * positions_alb + positions_amb,
                                                    n_rays=n_rays)
        comp_cosine_reflectance = accumulate_along_rays(weights, ray_indices,
                                                    values=torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None] * positions_alb,
                                                    n_rays=n_rays)
        comp_cosine = accumulate_along_rays(weights, ray_indices,
                                                    values=torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None],
                                                    n_rays=n_rays)

        comp_rgb_cam_albedo = comp_rgb_cam_albedo #+ rgb_proj_samples * (1.0 - opacity.unsqueeze(-1))
        comp_rgb_proj_albedo = comp_rgb_proj_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))
        
        rv = {
            'comp_rgb_proj': comp_rgb_proj,
            'comp_rgb_proj_albedo': comp_rgb_proj_albedo,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_rgb_cam_albedo': comp_rgb_cam_albedo,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'surface_points_proj_output': surface_points_proj_output,
            'comp_alb_cam': comp_alb_cam,
            'comp_cosine_reflectance': comp_cosine_reflectance,
            'comp_cosine': comp_cosine
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_c2w, all_images,
                fx_pro, fy_pro, cx_pro, cy_pro,
                fx_cam, fy_cam, cx_cam, cy_cam,
                step):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_c2w=all_c2w,all_images=all_images,
                                fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam,
                                step=step)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_c2w=all_c2w, all_images=all_images,
                              fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam,
                              step=step)

        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        return losses
