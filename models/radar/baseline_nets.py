# Baseline architecture: VGG-16 + NetVLAD

import torch
import torch.nn as nn

from models.radar.my_vgg import VGGFeatureExtractor, cfgs
from layers.radar.netvlad import NetVLADLoupe


class BaselineModel1(nn.Module):
    # Baseline model: VGG-16/NetVLAD
    def __init__(self):
        super().__init__()
        # Configs to test: D, X, Y, Z
        self.fe = VGGFeatureExtractor(cfg=cfgs['D'], batch_norm=True, padding_mode='constant')
        self.pool = NetVLADLoupe(feature_size=512, cluster_size=64, output_dim=256, gating=False, add_batch_norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, 1, 128, 384) tensor
        x = self.fe(x)
        # x is (batch_size, 512, h, w) tensor

        x = x.flatten(2)
        # x is (batch_size, c=512, h*w) tensor

        x = x.permute(0, 2, 1)
        # x is (batch_size, h*w, c=512) tensor

        # Expects (batch_size, num_points, channels) tensor
        x = self.pool(x)
        # x is (batch_size, out_dim) tensor
        return x
