import torch
import torch.nn as nn


class RadarNet(nn.Module):
    def __init__(self, fe_net: nn.Module, pooling: nn.Module, normalize: bool = False):
        # fe_net: feature extraction network
        # pooling_net: pooling module (e.g. GeM)
        super().__init__()
        self.fe_net = fe_net
        self.pooling = pooling
        self.normalize = normalize

    def forward(self, x: torch.Tensor):
        x = x['radar_batch']
        x = self.fe_net(x)
        x = self.pooling(x)     # (batch_size, n, 1, 1) tensor
        x = x.flatten(1)        # Reshape to (batch_size, n) tensor
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x
