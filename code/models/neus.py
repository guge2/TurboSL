import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_alpha, accumulate_along_rays


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
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

        sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        rgb = self.texture(feature, t_dirs, normal)
        # import pdb;pdb.set_trace()

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

        # import pdb;
        # pdb.set_trace()

        rv = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays)
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
        losses.update(self.texture.regularizations(out))
        return losses


@models.register('neus-geometry-only')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        # self.texture = models.make(self.config.texture.name, self.config.texture)
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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 all_w2c=None, all_images=None,
                 fx_pro=None, fy_pro=None, cx_pro=None, cy_pro=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
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

        images_proj = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

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
        rgb_proj = F.grid_sample(images_proj, positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj = rgb_proj.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # radiance estimated from projection onto camera image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_cam = torch.mm(K_cam, temp)
        else:
            positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

        # import pdb; pdb.set_trace()
        positions_cam = positions_cam / positions_cam[2, :]
        positions_cam[0, :] = (positions_cam[0, :] / all_images.shape[2]) * 2 - 1
        positions_cam[1, :] = (positions_cam[1, :] / all_images.shape[1]) * 2 - 1
        positions_cam = positions_cam[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_cam = F.grid_sample(images_cam, positions_cam, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_cam = rgb_cam.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

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
        surface_points_proj[0, :] = (surface_points_proj[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_proj[1, :] = (surface_points_proj[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_proj[0, :] = ((surface_points_proj[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_proj[1, :] = ((surface_points_proj[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_proj = surface_points_proj[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_proj = F.grid_sample(images_proj, surface_points_proj, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_proj = comp_rgb_depth_proj.squeeze(0).squeeze(1).T


        # image rgb computed from the depth estimate and camera image
        surface_points_hom = torch.cat((surface_points, torch.ones_like(depth).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, surface_points_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            surface_points_cam = torch.mm(K_cam, temp)
        else:
            surface_points_cam = torch.mm(K_cam, torch.mm(w2c_cam, surface_points_hom))

        surface_points_cam = surface_points_cam / surface_points_cam[2, :]

        # if rays_o[0, 0] == 0 and depth.shape[0]==4096 and surface_points_cam[1, 0] >= 600:
        #     import pdb;
        #     pdb.set_trace()
        surface_points_cam[0, :] = (surface_points_cam[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_cam[1, :] = (surface_points_cam[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_cam[0, :] = ((surface_points_cam[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_cam[1, :] = ((surface_points_cam[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_cam = surface_points_cam[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_cam = F.grid_sample(images_cam, surface_points_cam, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_cam = comp_rgb_depth_cam.squeeze(0).squeeze(1).T


        rv = {
            'comp_rgb_proj': comp_rgb_proj,
            'comp_rgb_depth_proj': comp_rgb_depth_proj,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_rgb_depth_cam': comp_rgb_depth_cam,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_images,
                fx_pro, fy_pro, cx_pro, cy_pro,
                fx_cam, fy_cam, cx_cam, cy_cam):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_images=all_images,
                                fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_images=all_images,
                              fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses


@models.register('neus-geometry-only-dualproj')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        # self.texture = models.make(self.config.texture.name, self.config.texture)
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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 all_w2c=None, all_images=None,
                 fx_pro_1=None, fy_pro_1=None, cx_pro_1=None, cy_pro_1=None,
                 fx_pro_2=None, fy_pro_2=None, cx_pro_2=None, cy_pro_2=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
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


        w2c_proj_2 = all_w2c[2, ...]
        w2c_proj_1 = all_w2c[1, ...]
        w2c_cam = all_w2c[0, ...]

        K_proj_1 = np.array([[fx_pro_1, 0, cx_pro_1],
                           [0, fy_pro_1, cy_pro_1],
                           [0, 0, 1]])
        K_proj_1 = torch.from_numpy(K_proj_1).float().to(0)

        K_proj_2 = np.array([[fx_pro_2, 0, cx_pro_2],
                           [0, fy_pro_2, cy_pro_2],
                           [0, 0, 1]])
        K_proj_2 = torch.from_numpy(K_proj_2).float().to(0)

        K_cam = np.array([[fx_cam, 0, cx_cam],
                           [0, fy_cam, cy_cam],
                           [0, 0, 1]])
        K_cam = torch.from_numpy(K_cam).float().to(0)

        images_proj_2 = torch.permute(all_images[2, ...], (2, 0, 1)).unsqueeze(0)
        images_proj_1 = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

        # radiance estimated from projection onto projector one image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_proj_1, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_1 = torch.mm(K_proj_1, temp)
        else:
            positions_proj_1 = torch.mm(K_proj_1, torch.mm(w2c_proj_1, positions_hom))

        positions_proj_1 = positions_proj_1 / positions_proj_1[2, :]
        positions_proj_1[0, :] = (positions_proj_1[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_1[1, :] = (positions_proj_1[1, :] / all_images.shape[1]) * 2 - 1
        positions_proj_1 = positions_proj_1[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj_1 = F.grid_sample(images_proj_1, positions_proj_1, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj_1 = rgb_proj_1.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # radiance estimated from projection onto projector two image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_proj_2, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_2 = torch.mm(K_proj_2, temp)
        else:
            positions_proj_2 = torch.mm(K_proj_2, torch.mm(w2c_proj_2, positions_hom))

        positions_proj_2 = positions_proj_2 / positions_proj_2[2, :]
        positions_proj_2[0, :] = (positions_proj_2[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_2[1, :] = (positions_proj_2[1, :] / all_images.shape[1]) * 2 - 1
        positions_proj_2 = positions_proj_2[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj_2 = F.grid_sample(images_proj_2, positions_proj_2, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj_2 = rgb_proj_2.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # radiance estimated from projection onto camera image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_cam = torch.mm(K_cam, temp)
        else:
            positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

        # import pdb; pdb.set_trace()
        positions_cam = positions_cam / positions_cam[2, :]
        positions_cam[0, :] = (positions_cam[0, :] / all_images.shape[2]) * 2 - 1
        positions_cam[1, :] = (positions_cam[1, :] / all_images.shape[1]) * 2 - 1
        positions_cam = positions_cam[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_cam = F.grid_sample(images_cam, positions_cam, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_cam = rgb_cam.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # compute the ray estimates including the depth
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        comp_rgb_proj_1 = accumulate_along_rays(weights, ray_indices, values=rgb_proj_1, n_rays=n_rays)
        comp_rgb_proj_1 = comp_rgb_proj_1 + self.background_color * (1.0 - opacity)

        comp_rgb_proj_2 = accumulate_along_rays(weights, ray_indices, values=rgb_proj_2, n_rays=n_rays)
        comp_rgb_proj_2 = comp_rgb_proj_2 + self.background_color * (1.0 - opacity)

        comp_rgb_cam = accumulate_along_rays(weights, ray_indices, values=rgb_cam, n_rays=n_rays)
        comp_rgb_cam = comp_rgb_cam + self.background_color * (1.0 - opacity)

        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

        rv = {
            'comp_rgb_proj_1': comp_rgb_proj_1,
            'comp_rgb_proj_2': comp_rgb_proj_2,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
            # 'comp_rgb_depth_proj': comp_rgb_depth_proj,
            # 'comp_rgb_depth_cam': comp_rgb_depth_cam,
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_images,
                fx_pro_1, fy_pro_1, cx_pro_1, cy_pro_1,
                fx_pro_2, fy_pro_2, cx_pro_2, cy_pro_2,
                fx_cam, fy_cam, cx_cam, cy_cam):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_images=all_images,
                                fx_pro_1=fx_pro_1, fy_pro_1=fy_pro_1, cx_pro_1=cx_pro_1, cy_pro_1=cy_pro_1,
                                fx_pro_2=fx_pro_2, fy_pro_2=fy_pro_2, cx_pro_2=cx_pro_2, cy_pro_2=cy_pro_2,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_images=all_images,
                              fx_pro_1=fx_pro_1, fy_pro_1=fy_pro_1, cx_pro_1=cx_pro_1, cy_pro_1=cy_pro_1,
                              fx_pro_2=fx_pro_2, fy_pro_2=fy_pro_2, cx_pro_2=cx_pro_2, cy_pro_2=cy_pro_2,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses


@models.register('neus-geometry-albedo-ambient')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        # for param in self.geometry.parameters():
        #     param.requires_grad = False

        self.albedo = models.make(self.config.albedo.name, self.config.albedo)

        self.ambient = models.make(self.config.ambient.name, self.config.ambient)
        # for param in self.ambient.parameters():
        #     param.requires_grad = False

        self.variance = VarianceNetwork(self.config.variance)
        # for param in self.variance.parameters():
        #     import pdb;pdb.set_trace()
        #     param.requires_grad = False

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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 distribution=None, step=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)


        # if step>3010:
        if False:
            with torch.no_grad():
                ray_indices, t_starts, t_ends = ray_marching(
                    rays_o, rays_d,
                    scene_aabb=self.scene_aabb,
                    grid=self.occupancy_grid if self.config.grid_prune else None,
                    alpha_fn=None,
                    near_plane=None, far_plane=None,
                    render_step_size=self.render_step_size/5,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    alpha_thre=0.0
                )


            tic = time.time()
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            dists = t_ends - t_starts

            # import pdb; pdb.set_trace()


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

            # radiance estimated from projection onto projector image plane
            positions_hom = torch.cat((positions, torch.ones_like(ray_indices).unsqueeze(-1)), dim=1).T

            if self.config.opengl:
                temp = torch.mm(w2c_proj, positions_hom)
                temp[1, :] *= -1
                temp[2, :] *= -1
                positions_proj = torch.mm(K_proj, temp)
            else:
                positions_proj = torch.mm(K_proj, torch.mm(w2c_proj, positions_hom))

            if self.config.opengl:
                temp = torch.mm(w2c_cam, positions_hom)
                temp[1, :] *= -1
                temp[2, :] *= -1
                positions_cam = torch.mm(K_cam, temp)
            else:
                positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

            positions_cam = positions_cam / positions_cam[2, :]
            positions_cam[0, :] = torch.clip(positions_cam[0, :], 0, all_images.shape[2]-1)
            positions_cam[1, :] = torch.clip(positions_cam[1, :], 0, all_images.shape[1]-1)
            positions_cam = torch.floor(positions_cam).long()

            positions_proj = positions_proj / positions_proj[2, :]
            positions_proj[0, :] = torch.clip(positions_proj[0, :], 0, all_images.shape[2]-1)
            positions_proj[1, :] = torch.clip(positions_proj[1, :], 0, all_images.shape[1]-1)
            positions_proj = torch.floor(positions_proj).long()

            toc_1 = time.time()
            # import pdb;pdb.set_trace()

            distr = distribution[
                0, positions_cam[1, :], positions_cam[0, :], positions_proj[0, :]]
            toc_2 = time.time()

            # import pdb;pdb.set_trace()

            # if len(distr) > 0:
            #     samples = torch.multinomial(distr, len(distr)//5, replacement=False)
            # else:
            #     samples = []

            toc_3 = time.time()

            mask_samples = torch.zeros_like(distr)
            # mask_samples[samples] = 1
            mask_samples[distr == 1] = 1
            mask_samples = mask_samples.bool()
            ray_indices = ray_indices[mask_samples]
            t_starts = t_starts[mask_samples]
            # t_ends = torch.roll(t_starts, 1)
            t_ends = t_ends[mask_samples]

            toc_4 = time.time()

            # import pdb;pdb.set_trace()

        else:
            with torch.no_grad():
                # import pdb;pdb.set_trace()

                cam_o = all_c2w[0, :, 3]
                proj_o = all_c2w[1, :, 3]

                if self.config.back_rendering:
                    rays_o = rays_o + 10 * rays_d
                    rays_d = -rays_d
                # rays_o = torch.cat((rays_o, rays_o + 10 * rays_d), 0)
                # rays_d = torch.cat((rays_d, -rays_d), 0)
                # n_rays = n_rays * 2

                # import pdb;pdb.set_trace()
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

        # import pdb; pdb.set_trace()

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


        images_proj = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)


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
        rgb_proj = F.grid_sample(images_proj, positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj = rgb_proj.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

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
        # rgb = self.texture(feature, t_dirs, normal)
        ########################

        # compute the ray estimates including the depth
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)

        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        # out, argmax = scatter_max(weights.squeeze(-1), ray_indices, out=torch.zeros(n_rays, device=ray_indices.device))
        # if t_starts.shape[0] != 0:
        #     argmax[argmax == weights.shape[0]] = weights.shape[0] - 1
        #     depth = (t_starts + t_ends)[argmax] / 2
        # else:
        #     depth = out[:, None]

        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        comp_rgb_proj = accumulate_along_rays(weights, ray_indices, values=rgb_proj, n_rays=n_rays)
        comp_rgb_proj = comp_rgb_proj + self.background_color * (1.0 - opacity)

        comp_rgb_cam = accumulate_along_rays(weights, ray_indices, values=rgb_cam, n_rays=n_rays)
        comp_rgb_cam = comp_rgb_cam + self.background_color * (1.0 - opacity)

        # import pdb;pdb.set_trace()
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

        ########################
        # surface_points_proj[0, :] = (surface_points_proj[0, :] / all_images.shape[2]) * 2 - 1
        # surface_points_proj[1, :] = (surface_points_proj[1, :] / all_images.shape[1]) * 2 - 1
        # # surface_points_proj[0, :] = ((surface_points_proj[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # # surface_points_proj[1, :] = ((surface_points_proj[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        # surface_points_proj = surface_points_proj[:2, :].T.unsqueeze(0).unsqueeze(0)
        #
        # comp_rgb_depth_proj = F.grid_sample(images_proj, surface_points_proj, mode='bilinear', padding_mode='zeros', align_corners=False)
        # comp_rgb_depth_proj = comp_rgb_depth_proj.squeeze(0).squeeze(1).T
        #
        #
        # # image rgb computed from the depth estimate and camera image
        # surface_points_hom = torch.cat((surface_points, torch.ones_like(depth).unsqueeze(-1)), dim=1).T
        #
        # if self.config.opengl:
        #     temp = torch.mm(w2c_cam, surface_points_hom)
        #     temp[1, :] *= -1
        #     temp[2, :] *= -1
        #     surface_points_cam = torch.mm(K_cam, temp)
        # else:
        #     surface_points_cam = torch.mm(K_cam, torch.mm(w2c_cam, surface_points_hom))
        #
        # surface_points_cam = surface_points_cam / surface_points_cam[2, :]
        #
        # surface_points_cam[0, :] = (surface_points_cam[0, :] / all_images.shape[2]) * 2 - 1
        # surface_points_cam[1, :] = (surface_points_cam[1, :] / all_images.shape[1]) * 2 - 1
        # # surface_points_cam[0, :] = ((surface_points_cam[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # # surface_points_cam[1, :] = ((surface_points_cam[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        # surface_points_cam = surface_points_cam[:2, :].T.unsqueeze(0).unsqueeze(0)
        #
        # comp_rgb_depth_cam = F.grid_sample(images_cam, surface_points_cam, mode='bilinear', padding_mode='zeros', align_corners=False)
        # comp_rgb_depth_cam = comp_rgb_depth_cam.squeeze(0).squeeze(1).T
        ################

        ray_positions_hom = torch.cat(((rays_o + 100*rays_d).T, torch.ones_like(depth).unsqueeze(0)), dim=0)
        # ray_positions_hom = torch.cat((surface_points.T, torch.ones_like(depth).unsqueeze(0)), dim=0)

        if self.config.opengl:
            temp = torch.mm(w2c_cam, ray_positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            ray_positions_cam = torch.mm(K_cam, temp)
        else:
            ray_positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, ray_positions_hom))

        ray_positions_cam = ray_positions_cam / ray_positions_cam[2, :]
        ray_positions_cam[0, :] = (ray_positions_cam[0, :] / all_images.shape[2])
        ray_positions_cam[1, :] = (ray_positions_cam[1, :] / all_images.shape[1])


        alb = self.albedo(ray_positions_cam[:2, :].T)
        amb = self.ambient(ray_positions_cam[:2, :].T)


        if self.config.opengl:
            temp = torch.mm(w2c_proj, ray_positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            ray_positions_proj = torch.mm(K_proj, temp)
        else:
            ray_positions_proj = torch.mm(K_proj, torch.mm(w2c_proj, ray_positions_hom))

        ray_positions_proj = ray_positions_proj / ray_positions_proj[2, :]
        ray_positions_proj[0, :] = (ray_positions_proj[0, :] / all_images.shape[2]) * 2 - 1
        ray_positions_proj[1, :] = (ray_positions_proj[1, :] / all_images.shape[1]) * 2 - 1


        ray_positions_proj = ray_positions_proj[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj_samples = F.grid_sample(images_proj, ray_positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj_samples = rgb_proj_samples.squeeze(0).squeeze(1).T
        ################

        # import pdb;pdb.set_trace()
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

        positions_alb = self.albedo(positions_cam[:2, :].T)
        positions_amb = self.ambient(positions_cam[:2, :].T)

#########################

        if self.config.use_normal:
            if self.config.back_rendering:
                comp_alb_cam = accumulate_along_rays(weights, ray_indices,
                                                            values=torch.clamp(positions_alb * torch.cosine_similarity(-normal, proj_o - positions,
                                                                                            dim=1)[:, None], min=0.1, max=1),
                                                            n_rays=n_rays)
            else:
                comp_alb_cam = accumulate_along_rays(weights, ray_indices,
                                                            values=torch.clamp(positions_alb * torch.cosine_similarity(normal, proj_o - positions,
                                                                                            dim=1)[:, None], min=0.1, max=1),
                                                            n_rays=n_rays)
        else:
            comp_alb_cam = accumulate_along_rays(weights, ray_indices,
                                                        values=positions_alb,
                                                        n_rays=n_rays)
        comp_alb_cam = comp_alb_cam + self.background_color * (1.0 - opacity.unsqueeze(-1))

##########################

        # import pdb;pdb.set_trace()
        if self.config.back_rendering:
            if self.config.use_normal:
                # mxx = torch.nn.Softplus(beta=10, threshold=0.5)
                # comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                #                                             values=(rgb_cam-positions_amb)/(positions_alb + 1e-12)/mxx(
                #                                                     torch.cosine_similarity(-normal, proj_o - positions,
                #                                                                             dim=1))[:, None],
                #                                             n_rays=n_rays)
                comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                                                            values=(rgb_cam-positions_amb)/torch.clamp(positions_alb *
                                                                    torch.cosine_similarity(-normal, proj_o - positions,
                                                                                            dim=1)[:, None], min=0.1,
                                                                    max=1),
                                                            n_rays=n_rays)
            else:
                comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                                                            values=(rgb_cam-positions_amb)/(positions_alb + 1e-12),
                                                            n_rays=n_rays)
        else:
            # comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
            #                                             values=(rgb_cam-positions_amb)/(positions_alb + 1e-12)/(
            #                                                 torch.clamp(
            #                                                     torch.cosine_similarity(normal, -rays_d[ray_indices],
            #                                                                             dim=1)[:, None], min=0.05,
            #                                                     max=1)),
            #                                             n_rays=n_rays)

            if self.config.use_normal:
                comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                                                            values=(rgb_cam-positions_amb)/torch.clamp(positions_alb *
                                                                    torch.cosine_similarity(normal, proj_o - positions,
                                                                                            dim=1)[:, None], min=0.1,
                                                                    max=1),
                                                            n_rays=n_rays)
                # comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                #                                             values=(rgb_cam-positions_amb)/(positions_alb + 1e-12)/torch.clamp(
                #                                                     torch.cosine_similarity(normal, proj_o - positions,
                #                                                                             dim=1)[:, None], min=0.2,
                #                                                     max=1).detach(),
                #                                             n_rays=n_rays)
            else:
                comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
                                                            values=(rgb_cam-positions_amb)/(positions_alb + 1e-12),
                                                            n_rays=n_rays)



        # comp_rgb_cam_albedo = comp_rgb_cam_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))
        comp_rgb_cam_albedo = comp_rgb_cam_albedo + rgb_proj_samples * (1.0 - opacity.unsqueeze(-1))


            # Working!
            #     comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
            #                                                 values=(rgb_cam-positions_amb)/(positions_alb + 1e-12),
            #                                                 n_rays=n_rays)
            #     comp_rgb_cam_albedo = comp_rgb_cam_albedo / torch.clamp(torch.cosine_similarity(comp_normal, -rays_d, dim=1)[:, None], min=0.05, max=1)


            # comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices,
            #                                             values=(rgb_cam-positions_amb),
            #                                             n_rays=n_rays)
            # comp_rgb_cam_albedo = comp_rgb_cam_albedo / (comp_alb_cam + 1e-12) / torch.clamp(torch.cosine_similarity(comp_normal, -rays_d, dim=1)[:, None], min=0.05, max=1)


        if self.config.back_rendering:

            if self.config.use_normal:
                comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
                                                         values=rgb_proj*torch.cosine_similarity(-normal, proj_o - positions, dim=1)[:, None] * positions_alb + positions_amb,
                                                         n_rays=n_rays)
                comp_cosine_reflectance = accumulate_along_rays(weights, ray_indices,
                                                         values=torch.cosine_similarity(-normal, proj_o - positions, dim=1)[:, None] * positions_alb,
                                                         n_rays=n_rays)
                comp_cosine = accumulate_along_rays(weights, ray_indices,
                                                         values=torch.cosine_similarity(-normal, proj_o - positions, dim=1)[:, None],
                                                         n_rays=n_rays)
            else:
                comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
                                                             values=rgb_proj * positions_alb + positions_amb,
                                                             n_rays=n_rays)
                comp_cosine_reflectance = accumulate_along_rays(weights, ray_indices,
                                                             values=positions_alb,
                                                             n_rays=n_rays)
                comp_cosine = accumulate_along_rays(weights, ray_indices,
                                                             values=torch.ones_like(positions_alb),
                                                             n_rays=n_rays)
        else:
            if self.config.use_normal:
                comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
                                                         values=rgb_proj*torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None] * positions_alb + positions_amb,
                                                         n_rays=n_rays)
                comp_cosine_reflectance = accumulate_along_rays(weights, ray_indices,
                                                         values=torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None] * positions_alb,
                                                         n_rays=n_rays)
                comp_cosine = accumulate_along_rays(weights, ray_indices,
                                                         values=torch.cosine_similarity(normal, proj_o - positions, dim=1)[:, None],
                                                         n_rays=n_rays)

            else:
                comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
                                                             values=rgb_proj * positions_alb + positions_amb,
                                                             n_rays=n_rays)
                comp_cosine_reflectance = accumulate_along_rays(weights, ray_indices,
                                                             values=positions_alb,
                                                             n_rays=n_rays)
                comp_cosine = accumulate_along_rays(weights, ray_indices,
                                                             values=torch.ones_like(positions_alb),
                                                             n_rays=n_rays)

        comp_rgb_proj_albedo = comp_rgb_proj_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))
        # comp_rgb_proj_albedo = alb * comp_rgb_proj_albedo + amb + self.background_color * (1.0 - opacity.unsqueeze(-1))

        #before
        # comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
        #                                              values=rgb_proj * positions_alb + positions_amb,
        #                                              n_rays=n_rays)
        # comp_rgb_proj_albedo = comp_rgb_proj_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))

        #after
        # import pdb;pdb.set_trace()
        # comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices,
        #                                              values=rgb_proj,
        #                                              n_rays=n_rays)

        # import pdb;
        # pdb.set_trace()
        # *torch.abs(torch.cosine_similarity(normal, positions - rays_o[ray_indices], dim=1)[:, None])
        # import pdb;pdb.set_trace()
        ###########
        # positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T
        #
        # if self.config.opengl:
        #     temp = torch.mm(w2c_proj, positions_hom)
        #     temp[1, :] *= -1
        #     temp[2, :] *= -1
        #     positions_proj = torch.mm(K_proj, temp)
        # else:
        #     positions_proj = torch.mm(K_proj, torch.mm(w2c_proj, positions_hom))
        #
        # positions_proj = positions_proj / positions_proj[2, :]
        # positions_proj[0, :] = (positions_proj[0, :] / all_images.shape[2])
        # positions_proj[1, :] = (positions_proj[1, :] / all_images.shape[1])
        #
        # # import pdb;pdb.set_trace()
        #
        # cam_residuals = torch.abs(rgb_proj * positions_alb - rgb_cam)
        # cam_residuals_weighted = accumulate_along_rays(weights, ray_indices, values=cam_residuals, n_rays=n_rays)




        # import pdb;pdb.set_trace()
        # comp_rgb_proj = comp_rgb_proj[0:n_rays // 2, :]
        # comp_rgb_cam = comp_rgb_cam[0:n_rays // 2, :]
        # comp_rgb_cam_albedo = comp_rgb_cam_albedo[0:n_rays // 2, :]
        # comp_rgb_proj_albedo_2 = comp_rgb_proj_albedo[n_rays // 2 :, :]
        # comp_rgb_proj_albedo = comp_rgb_proj_albedo[0:n_rays // 2, :]
        # comp_normal = comp_normal[0:n_rays // 2, :]
        # opacity = opacity[0:n_rays // 2]
        # depth_2 = depth[n_rays // 2 :]
        # depth = depth[0:n_rays // 2]
        # opacity = opacity[0:n_rays // 2]
        # t_starts = t_starts[0:len(t_starts)// 2, :]
        # alb = alb[0:n_rays // 2, :]
        # amb = amb[0:n_rays // 2, :]
        # surface_points_proj_output= surface_points_proj_output[0:n_rays // 2, :]
        # comp_alb_cam = comp_alb_cam[0:n_rays // 2, :]
        ################

        # import pdb;pdb.set_trace()

        rv = {
            'comp_rgb_proj': comp_rgb_proj,
            # 'comp_rgb_depth_proj': comp_rgb_depth_proj,
            'comp_rgb_proj_albedo': comp_rgb_proj_albedo,
            'comp_rgb_cam': comp_rgb_cam,
            # 'comp_rgb_depth_cam': comp_rgb_depth_cam,
            'comp_rgb_cam_albedo': comp_rgb_cam_albedo,
            # 'comp_rgb_proj_albedo_2': comp_rgb_proj_albedo_2,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            # 'depth_2': depth_2,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'alb': alb,
            'amb': amb,
            'surface_points_proj_output': surface_points_proj_output,
            'comp_alb_cam': comp_alb_cam,
            'comp_cosine_reflectance': comp_cosine_reflectance,
            'comp_cosine': comp_cosine
            # 'cam_res': cam_residuals_weighted
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
                distribution, step):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_c2w=all_c2w,all_images=all_images,
                                fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam,
                                distribution=distribution, step=step)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_c2w=all_c2w, all_images=all_images,
                              fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam,
                              distribution=distribution, step=step)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses

@models.register('neus-geometry-albedo-ambient-color')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        # for param in self.geometry.parameters():
        #     param.requires_grad = False

        self.albedo = models.make(self.config.albedo.name, self.config.albedo)


        self.ambient = models.make(self.config.ambient.name, self.config.ambient)

        for param in self.ambient.parameters():
            param.requires_grad = False

        self.variance = VarianceNetwork(self.config.variance)
        # for param in self.variance.parameters():
        #     param.requires_grad = False

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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 all_w2c=None, all_images=None,
                 fx_pro=None, fy_pro=None, cx_pro=None, cy_pro=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
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

        # import pdb;
        # pdb.set_trace()

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

        images_proj = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

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
        rgb_proj = F.grid_sample(images_proj, positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj = rgb_proj.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

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
        # rgb = self.texture(feature, t_dirs, normal)

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
        surface_points_proj[0, :] = (surface_points_proj[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_proj[1, :] = (surface_points_proj[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_proj[0, :] = ((surface_points_proj[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_proj[1, :] = ((surface_points_proj[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_proj = surface_points_proj[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_proj = F.grid_sample(images_proj, surface_points_proj, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_proj = comp_rgb_depth_proj.squeeze(0).squeeze(1).T


        # image rgb computed from the depth estimate and camera image
        surface_points_hom = torch.cat((surface_points, torch.ones_like(depth).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, surface_points_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            surface_points_cam = torch.mm(K_cam, temp)
        else:
            surface_points_cam = torch.mm(K_cam, torch.mm(w2c_cam, surface_points_hom))

        surface_points_cam = surface_points_cam / surface_points_cam[2, :]

        # if rays_o[0, 0] == 0 and depth.shape[0]==4096 and surface_points_cam[1, 0] >= 600:
        #     import pdb;
        #     pdb.set_trace()
        surface_points_cam[0, :] = (surface_points_cam[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_cam[1, :] = (surface_points_cam[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_cam[0, :] = ((surface_points_cam[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_cam[1, :] = ((surface_points_cam[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_cam = surface_points_cam[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_cam = F.grid_sample(images_cam, surface_points_cam, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_cam = comp_rgb_depth_cam.squeeze(0).squeeze(1).T

        ################
        # import pdb;pdb.set_trace()
        ray_positions_hom = torch.cat(((rays_o + 100*rays_d).T, torch.ones_like(depth).unsqueeze(0)), dim=0)
        # ray_positions_hom = torch.cat((surface_points.T, torch.ones_like(depth).unsqueeze(0)), dim=0)

        if self.config.opengl:
            temp = torch.mm(w2c_cam, ray_positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            ray_positions_cam = torch.mm(K_cam, temp)
        else:
            ray_positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, ray_positions_hom))

        ray_positions_cam = ray_positions_cam / ray_positions_cam[2, :]
        ray_positions_cam[0, :] = (ray_positions_cam[0, :] / all_images.shape[2])
        ray_positions_cam[1, :] = (ray_positions_cam[1, :] / all_images.shape[1])


        alb = self.albedo(ray_positions_cam[:2, :].T)
        amb = self.ambient(ray_positions_cam[:2, :].T)

        # import pdb;pdb.set_trace()
        ################

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

        positions_alb = self.albedo(positions_cam[:2, :].T)
        positions_alb = torch.tile(positions_alb, (1, rgb_cam.shape[-1]//positions_alb.shape[-1]))

        comp_rgb_cam_albedo = accumulate_along_rays(weights, ray_indices, values=torch.clamp(rgb_cam/(positions_alb + 1e-12), min=0, max=3), n_rays=n_rays)
        comp_rgb_cam_albedo = comp_rgb_cam_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))

        comp_rgb_proj_albedo = accumulate_along_rays(weights, ray_indices, values=rgb_proj * positions_alb, n_rays=n_rays)
        comp_rgb_proj_albedo = comp_rgb_proj_albedo + self.background_color * (1.0 - opacity.unsqueeze(-1))

        ###########
        # positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T
        #
        # if self.config.opengl:
        #     temp = torch.mm(w2c_proj, positions_hom)
        #     temp[1, :] *= -1
        #     temp[2, :] *= -1
        #     positions_proj = torch.mm(K_proj, temp)
        # else:
        #     positions_proj = torch.mm(K_proj, torch.mm(w2c_proj, positions_hom))
        #
        # positions_proj = positions_proj / positions_proj[2, :]
        # positions_proj[0, :] = (positions_proj[0, :] / all_images.shape[2])
        # positions_proj[1, :] = (positions_proj[1, :] / all_images.shape[1])
        #
        # # import pdb;pdb.set_trace()
        #
        # cam_residuals = torch.abs(rgb_proj * positions_alb - rgb_cam)
        # cam_residuals_weighted = accumulate_along_rays(weights, ray_indices, values=cam_residuals, n_rays=n_rays)





        ################


        rv = {
            'comp_rgb_proj': comp_rgb_proj,
            'comp_rgb_depth_proj': comp_rgb_depth_proj,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_rgb_cam_albedo': comp_rgb_cam_albedo,
            'comp_rgb_proj_albedo': comp_rgb_proj_albedo,
            'comp_rgb_depth_cam': comp_rgb_depth_cam,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'alb': alb,
            'amb': amb,
            # 'cam_res': cam_residuals_weighted
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_images,
                fx_pro, fy_pro, cx_pro, cy_pro,
                fx_cam, fy_cam, cx_cam, cy_cam):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_images=all_images,
                                fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_images=all_images,
                              fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses

@models.register('neus-geometry-albedo-ambient-dualproj')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)

        self.albedo_1 = models.make(self.config.albedo_1.name, self.config.albedo_1)
        self.albedo_2 = models.make(self.config.albedo_2.name, self.config.albedo_2)


        self.ambient = models.make(self.config.ambient.name, self.config.ambient)
        for param in self.ambient.parameters():
            param.requires_grad = False

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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 all_w2c=None, all_images=None,
                 fx_pro_1=None, fy_pro_1=None, cx_pro_1=None, cy_pro_1=None,
                 fx_pro_2=None, fy_pro_2=None, cx_pro_2=None, cy_pro_2=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
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

        # import pdb;pdb.set_trace()
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        sdf, sdf_grad = self.geometry(positions, with_grad=True, with_feature=False)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[..., None]


        w2c_proj_2 = all_w2c[2, ...]
        w2c_proj_1 = all_w2c[1, ...]
        w2c_cam = all_w2c[0, ...]

        K_proj_1 = np.array([[fx_pro_1, 0, cx_pro_1],
                           [0, fy_pro_1, cy_pro_1],
                           [0, 0, 1]])
        K_proj_1 = torch.from_numpy(K_proj_1).float().to(0)

        K_proj_2 = np.array([[fx_pro_2, 0, cx_pro_2],
                           [0, fy_pro_2, cy_pro_2],
                           [0, 0, 1]])
        K_proj_2 = torch.from_numpy(K_proj_2).float().to(0)

        K_cam = np.array([[fx_cam, 0, cx_cam],
                           [0, fy_cam, cy_cam],
                           [0, 0, 1]])
        K_cam = torch.from_numpy(K_cam).float().to(0)

        images_proj_2 = torch.permute(all_images[2, ...], (2, 0, 1)).unsqueeze(0)
        images_proj_1 = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

        # radiance estimated from projection onto projector one image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_proj_1, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_1 = torch.mm(K_proj_1, temp)
        else:
            positions_proj_1 = torch.mm(K_proj_1, torch.mm(w2c_proj_1, positions_hom))

        positions_proj_1 = positions_proj_1 / positions_proj_1[2, :]
        positions_proj_1[0, :] = (positions_proj_1[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_1[1, :] = (positions_proj_1[1, :] / all_images.shape[1]) * 2 - 1
        positions_proj_1 = positions_proj_1[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj_1 = F.grid_sample(images_proj_1, positions_proj_1, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj_1 = rgb_proj_1.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # radiance estimated from projection onto projector two image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_proj_2, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_2 = torch.mm(K_proj_2, temp)
        else:
            positions_proj_2 = torch.mm(K_proj_2, torch.mm(w2c_proj_2, positions_hom))

        positions_proj_2 = positions_proj_2 / positions_proj_2[2, :]
        positions_proj_2[0, :] = (positions_proj_2[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_2[1, :] = (positions_proj_2[1, :] / all_images.shape[1]) * 2 - 1
        positions_proj_2 = positions_proj_2[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_proj_2 = F.grid_sample(images_proj_2, positions_proj_2, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj_2 = rgb_proj_2.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # radiance estimated from projection onto camera image plane
        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_cam = torch.mm(K_cam, temp)
        else:
            positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, positions_hom))

        # import pdb; pdb.set_trace()
        positions_cam = positions_cam / positions_cam[2, :]
        positions_cam[0, :] = (positions_cam[0, :] / all_images.shape[2]) * 2 - 1
        positions_cam[1, :] = (positions_cam[1, :] / all_images.shape[1]) * 2 - 1
        positions_cam = positions_cam[:2, :].T.unsqueeze(0).unsqueeze(0)
        rgb_cam = F.grid_sample(images_cam, positions_cam, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_cam = rgb_cam.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

        # compute the ray estimates including the depth
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        comp_rgb_proj_1 = accumulate_along_rays(weights, ray_indices, values=rgb_proj_1, n_rays=n_rays)
        comp_rgb_proj_1 = comp_rgb_proj_1 + self.background_color * (1.0 - opacity)

        comp_rgb_proj_2 = accumulate_along_rays(weights, ray_indices, values=rgb_proj_2, n_rays=n_rays)
        comp_rgb_proj_2 = comp_rgb_proj_2 + self.background_color * (1.0 - opacity)

        comp_rgb_cam = accumulate_along_rays(weights, ray_indices, values=rgb_cam, n_rays=n_rays)
        comp_rgb_cam = comp_rgb_cam + self.background_color * (1.0 - opacity)

        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)


        ################

        positions_hom = torch.cat((positions, torch.ones_like(sdf).unsqueeze(-1)), dim=1).T

        #####
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

        #####
        if self.config.opengl:
            temp = torch.mm(w2c_proj_1, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_1 = torch.mm(K_proj_1, temp)
        else:
            positions_proj_1 = torch.mm(K_proj_1, torch.mm(w2c_proj_1, positions_hom))

        positions_proj_1 = positions_proj_1 / positions_proj_1[2, :]
        positions_proj_1[0, :] = (positions_proj_1[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_1[1, :] = (positions_proj_1[1, :] / all_images.shape[1]) * 2 - 1

        ####
        if self.config.opengl:
            temp = torch.mm(w2c_proj_2, positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            positions_proj_2 = torch.mm(K_proj_2, temp)
        else:
            positions_proj_2 = torch.mm(K_proj_2, torch.mm(w2c_proj_2, positions_hom))

        positions_proj_2 = positions_proj_2 / positions_proj_2[2, :]
        positions_proj_2[0, :] = (positions_proj_2[0, :] / all_images.shape[2]) * 2 - 1
        positions_proj_2[1, :] = (positions_proj_2[1, :] / all_images.shape[1]) * 2 - 1

        positions_alb1 = self.albedo_1(positions_cam[:2, :].T)
        positions_alb2 = self.albedo_2(positions_cam[:2, :].T)


        comp_rgb_cam_proj1 = accumulate_along_rays(weights, ray_indices, values=torch.clamp((rgb_cam-positions_alb1 * rgb_proj_1)/(positions_alb2 + 1e-12), min=0, max=1), n_rays=n_rays)
        # comp_rgb_cam_proj1 = accumulate_along_rays(weights, ray_indices, values=torch.clamp((rgb_cam-positions_alb1 * rgb_proj_1), min=0, max=1), n_rays=n_rays)
        comp_rgb_cam_proj1 = comp_rgb_cam_proj1 + self.background_color * (1.0 - opacity.unsqueeze(-1))

        comp_rgb_cam_proj2 = accumulate_along_rays(weights, ray_indices, values=torch.clamp((rgb_cam-positions_alb2 * rgb_proj_2)/(positions_alb1 + 1e-12), min=0, max=1), n_rays=n_rays)
        # comp_rgb_cam_proj2 = accumulate_along_rays(weights, ray_indices, values=torch.clamp((rgb_cam-positions_alb2 * rgb_proj_2), min=0, max=1), n_rays=n_rays)
        comp_rgb_cam_proj2 = comp_rgb_cam_proj2 + self.background_color * (1.0 - opacity.unsqueeze(-1))

        comp_rgb_proj1_proj2 = accumulate_along_rays(weights, ray_indices, values=(rgb_proj_1 * positions_alb1 + rgb_proj_2 * positions_alb2), n_rays=n_rays)
        comp_rgb_proj1_proj2 = comp_rgb_proj1_proj2 + self.background_color * (1.0 - opacity.unsqueeze(-1))

        ###########


        ################
        # import pdb;pdb.set_trace()
        ray_positions_hom = torch.cat(((rays_o + 100*rays_d).T, torch.ones_like(depth).unsqueeze(0)), dim=0)
        # ray_positions_hom = torch.cat((surface_points.T, torch.ones_like(depth).unsqueeze(0)), dim=0)

        if self.config.opengl:
            temp = torch.mm(w2c_cam, ray_positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            ray_positions_cam = torch.mm(K_cam, temp)
        else:
            ray_positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, ray_positions_hom))

        ray_positions_cam = ray_positions_cam / ray_positions_cam[2, :]
        ray_positions_cam[0, :] = (ray_positions_cam[0, :] / all_images.shape[2])
        ray_positions_cam[1, :] = (ray_positions_cam[1, :] / all_images.shape[1])


        alb1 = self.albedo_1(ray_positions_cam[:2, :].T)
        alb2 = self.albedo_2(ray_positions_cam[:2, :].T)
        amb = self.ambient(ray_positions_cam[:2, :].T)
        ####

        rv = {
            'comp_rgb_proj_1': comp_rgb_proj_1,
            'comp_rgb_proj_2': comp_rgb_proj_2,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'comp_rgb_cam_proj1': comp_rgb_cam_proj1,
            'comp_rgb_cam_proj2': comp_rgb_cam_proj2,
            'comp_rgb_proj1_proj2': comp_rgb_proj1_proj2,
            'alb1': alb1,
            'alb2': alb2,
            'amb': amb
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_images,
                fx_pro_1, fy_pro_1, cx_pro_1, cy_pro_1,
                fx_pro_2, fy_pro_2, cx_pro_2, cy_pro_2,
                fx_cam, fy_cam, cx_cam, cy_cam):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_images=all_images,
                                fx_pro_1=fx_pro_1, fy_pro_1=fy_pro_1, cx_pro_1=cx_pro_1, cy_pro_1=cy_pro_1,
                                fx_pro_2=fx_pro_2, fy_pro_2=fy_pro_2, cx_pro_2=cx_pro_2, cy_pro_2=cy_pro_2,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_images=all_images,
                              fx_pro_1=fx_pro_1, fy_pro_1=fy_pro_1, cx_pro_1=cx_pro_1, cy_pro_1=cy_pro_1,
                              fx_pro_2=fx_pro_2, fy_pro_2=fy_pro_2, cx_pro_2=cx_pro_2, cy_pro_2=cy_pro_2,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses


@models.register('neus-geometry-albedo-ambient-with-normal')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        # for param in self.geometry.parameters():
        #     param.requires_grad = False

        self.albedo = models.make(self.config.albedo.name, self.config.albedo)

        self.ambient = models.make(self.config.ambient.name, self.config.ambient)
        for param in self.ambient.parameters():
            param.requires_grad = False

        self.variance = VarianceNetwork(self.config.variance)
        # for param in self.variance.parameters():
        #     param.requires_grad = False

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

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        # update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

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
                 all_w2c=None, all_images=None,
                 fx_pro=None, fy_pro=None, cx_pro=None, cy_pro=None,
                 fx_cam=None, fy_cam=None, cx_cam=None, cy_cam=None):

        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
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

        images_proj = torch.permute(all_images[1, ...], (2, 0, 1)).unsqueeze(0)
        images_cam = torch.permute(all_images[0, ...], (2, 0, 1)).unsqueeze(0)

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
        rgb_proj = F.grid_sample(images_proj, positions_proj, mode='bilinear', padding_mode='zeros', align_corners=None)
        rgb_proj = rgb_proj.squeeze(0).squeeze(1).T
        # rgb = self.texture(feature, t_dirs, normal)

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
        # rgb = self.texture(feature, t_dirs, normal)

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
        surface_points_proj[0, :] = (surface_points_proj[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_proj[1, :] = (surface_points_proj[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_proj[0, :] = ((surface_points_proj[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_proj[1, :] = ((surface_points_proj[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_proj = surface_points_proj[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_proj = F.grid_sample(images_proj, surface_points_proj, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_proj = comp_rgb_depth_proj.squeeze(0).squeeze(1).T


        # image rgb computed from the depth estimate and camera image
        surface_points_hom = torch.cat((surface_points, torch.ones_like(depth).unsqueeze(-1)), dim=1).T

        if self.config.opengl:
            temp = torch.mm(w2c_cam, surface_points_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            surface_points_cam = torch.mm(K_cam, temp)
        else:
            surface_points_cam = torch.mm(K_cam, torch.mm(w2c_cam, surface_points_hom))

        surface_points_cam = surface_points_cam / surface_points_cam[2, :]

        # if rays_o[0, 0] == 0 and depth.shape[0]==4096 and surface_points_cam[1, 0] >= 600:
        #     import pdb;
        #     pdb.set_trace()
        surface_points_cam[0, :] = (surface_points_cam[0, :] / all_images.shape[2]) * 2 - 1
        surface_points_cam[1, :] = (surface_points_cam[1, :] / all_images.shape[1]) * 2 - 1
        # surface_points_cam[0, :] = ((surface_points_cam[0, :] - 0.5) / (all_images.shape[2] - 1.0)) * 2 - 1
        # surface_points_cam[1, :] = ((surface_points_cam[1, :] - 0.5) / (all_images.shape[1] - 1.0)) * 2 - 1
        surface_points_cam = surface_points_cam[:2, :].T.unsqueeze(0).unsqueeze(0)

        comp_rgb_depth_cam = F.grid_sample(images_cam, surface_points_cam, mode='bilinear', padding_mode='zeros', align_corners=False)
        comp_rgb_depth_cam = comp_rgb_depth_cam.squeeze(0).squeeze(1).T

        ################
        # import pdb;pdb.set_trace()
        ray_positions_hom = torch.cat(((rays_o + 100*rays_d).T, torch.ones_like(depth).unsqueeze(0)), dim=0)
        # ray_positions_hom = torch.cat((surface_points.T, torch.ones_like(depth).unsqueeze(0)), dim=0)

        if self.config.opengl:
            temp = torch.mm(w2c_cam, ray_positions_hom)
            temp[1, :] *= -1
            temp[2, :] *= -1
            ray_positions_cam = torch.mm(K_cam, temp)
        else:
            ray_positions_cam = torch.mm(K_cam, torch.mm(w2c_cam, ray_positions_hom))

        ray_positions_cam = ray_positions_cam / ray_positions_cam[2, :]
        ray_positions_cam[0, :] = (ray_positions_cam[0, :] / all_images.shape[2])
        ray_positions_cam[1, :] = (ray_positions_cam[1, :] / all_images.shape[1])


        alb = self.albedo(ray_positions_cam[:2, :].T)
        amb = self.ambient(ray_positions_cam[:2, :].T)
        ################

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

        positions_alb = self.albedo(positions_cam[:2, :].T)

        # import pdb;pdb.set_trace()

        comp_rgb_cam_albedo_normal = \
            accumulate_along_rays(weights, ray_indices,
                                  values=torch.clamp(rgb_cam/((positions_alb + 1e-12) *
                                                              torch.abs(torch.cosine_similarity(normal, positions-rays_o[ray_indices], dim=1)[:, None])),
                                                     min=0, max=1), n_rays=n_rays)
        comp_rgb_cam_albedo_normal = comp_rgb_cam_albedo_normal + self.background_color * (1.0 - opacity.unsqueeze(-1))

        ################

        comp_rgb_proj_normal = accumulate_along_rays(weights, ray_indices,
                                                     values=rgb_proj * torch.abs(torch.cosine_similarity(normal, positions-rays_o[ray_indices], dim=1)[:, None]),
                                                     n_rays=n_rays)
        comp_rgb_proj_normal = comp_rgb_proj_normal + self.background_color * (1.0 - opacity.unsqueeze(-1))


        ################


        rv = {
            'comp_rgb_proj': comp_rgb_proj,
            'comp_rgb_proj_normal': comp_rgb_proj_normal,
            'comp_rgb_depth_proj': comp_rgb_depth_proj,
            'comp_rgb_cam': comp_rgb_cam,
            'comp_rgb_cam_albedo_normal': comp_rgb_cam_albedo_normal,
            'comp_rgb_depth_cam': comp_rgb_depth_cam,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'alb': alb,
            'amb': amb
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad
            })

        return rv

    def forward(self, rays,
                all_w2c, all_images,
                fx_pro, fy_pro, cx_pro, cy_pro,
                fx_cam, fy_cam, cx_cam, cy_cam):
        if self.training:
            out = self.forward_(rays,
                                all_w2c=all_w2c, all_images=all_images,
                                fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                                fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays,
                              all_w2c=all_w2c, all_images=all_images,
                              fx_pro=fx_pro, fy_pro=fy_pro, cx_pro=cx_pro, cy_pro=cy_pro,
                              fx_cam=fx_cam, fy_cam=fy_cam, cx_cam=cx_cam, cy_cam=cy_cam)            # import pdb
            # pdb.set_trace()

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
        # losses.update(self.texture.regularizations(out))
        return losses