import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network



@models.register('2d-implicit-ambient')
class ImplicitAmbient(nn.Module):
    def __init__(self, config):
        super(ImplicitAmbient, self).__init__()
        self.config = config
        self.n_input_dims = self.config.get('n_input_dim', 2)
        self.n_output_dims = self.config.n_output_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims,
                                                               self.config.xyz_encoding_config,
                                                               self.config.mlp_network_config)

    def forward(self, points):
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1],
                                                                                  self.n_output_dims).float()

        return out
