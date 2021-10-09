from typing import Dict, List
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

import layers.lidar.pooling as pooling


class MinkHead(nn.Module):
    def __init__(self, in_levels: List[int], in_channels: List[int], out_channels: int):
        # in_levels: list of levels from the bottom_up trunk which output is taken as an input (starting from 0)
        # in_channels: number of channels in each input_level
        # out_channels: number of output channels
        assert len(in_levels) > 0
        assert len(in_levels) == len(in_channels)

        super().__init__()
        self.in_levels = in_levels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.min_level = min(in_levels)
        self.max_level = max(in_levels)
        assert self.min_level > 0

        # Dictionary in_d[input_level] = number of channels
        self.in_d = {in_level: in_channel for in_level, in_channel in zip(self.in_levels, self.in_channels)}

        self.conv1x1 = nn.ModuleDict()   # 1x1 convolutions in lateral connections
        self.tconv = nn.ModuleDict()    # Top-down transposed convolutions

        for in_level in range(self.min_level+1, self.max_level+1):
            # There's one transposed convolution for each level, except for min_level
            self.tconv[str(in_level)] = ME.MinkowskiConvolutionTranspose(self.out_channels, self.out_channels,
                                                                         kernel_size=2, stride=2, dimension=3)
        for in_level in self.in_d:
            # Create lateral 1x1x convolution for each input level
            self.conv1x1[str(in_level)] = ME.MinkowskiConvolution(self.in_d[in_level], self.out_channels, kernel_size=1,
                                                                  stride=1, dimension=3)

    def forward(self, x: Dict[int, ME.SparseTensor]) -> ME.SparseTensor:
        # x: dictionary of tensors from different trunk levels, indexed by the trunk level

        # TOP-DOWN PASS
        y = self.conv1x1[str(self.max_level)](x[self.max_level])
        #y = x[self.max_level]
        for level in range(self.max_level-1, self.min_level-1, -1):
            y = self.tconv[str(level+1)](y)
            if level in self.in_d:
                # Add the feature map from the corresponding level at bottom-up pass using a skip connection
                y = y + self.conv1x1[str(level)](x[level])

        assert y.shape[1] == self.out_channels

        return y

    def __str__(self):
        n_params = sum([param.nelement() for param in self.parameters()])
        str = f'Input levels: {self.in_levels}   Input channels: {self.in_channels}   #parameters: {n_params}'
        return str


class MinkTrunk(nn.Module):
    # Bottom-up trunk

    def __init__(self, in_channels: int, planes: List[int], layers: List[int] = None,
                 conv0_kernel_size: int = 5, block=BasicBlock, min_out_level: int = 1):
        # min_out_level: minimum level of bottom-up pass to return the feature map. If None, all feature maps from
        #                the level 1 to the top are returned.

        super().__init__()
        self.in_channels = in_channels
        self.planes = planes
        if layers is None:
            self.layers = [1] * len(self.planes)
        else:
            assert len(layers) == len(planes)
            self.layers = layers

        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        assert min_out_level <= len(planes)
        self.min_out_level = min_out_level

        self.num_bottom_up = len(self.planes)
        self.init_dim = planes[0]
        assert len(self.layers) == len(self.planes)

        self.convs = nn.ModuleDict()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleDict()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleDict()  # Bottom-up blocks

        # The first convolution is special case, with larger kernel
        self.inplanes = self.planes[0]
        self.convs[str(0)] = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                                     dimension=3)
        self.bn[str(0)] = ME.MinkowskiBatchNorm(self.inplanes)

        for ndx, (plane, layer) in enumerate(zip(self.planes, self.layers)):
            self.convs[str(ndx + 1)] = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                                                               dimension=3)
            self.bn[str(ndx + 1)] = ME.MinkowskiBatchNorm(self.inplanes)
            self.blocks[str(ndx + 1)] = self._make_layer(self.block, plane, layer)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.weight_initialization()

    def weight_initialization(self):
            for m in self.modules():
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1 ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(ME.MinkowskiConvolution(self.inplanes, planes * block.expansion, kernel_size=1,
                                                               stride=stride, dimension=3),
                                       ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample,
                            dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=3))

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> Dict[int, ME.SparseTensor]:
        # *** BOTTOM-UP PASS ***
        y = {}      # Output feature maps

        x = self.convs[str(0)](x)
        x = self.bn[str(0)](x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        for i in range(1, len(self.layers)+1):
            x = self.convs[str(i)](x)           # Decreases spatial resolution (conv stride=2)
            x = self.bn[str(i)](x)
            x = self.relu(x)
            x = self.blocks[str(i)](x)
            if i >= self.min_out_level:
                y[i] = x

        return y

    def __str__(self):
        n_params = sum([param.nelement() for param in self.parameters()])
        str = f'Planes: {self.planes}   Min out Level: {self.min_out_level}   #parameters: {n_params}'
        return str


class SaliencyRegressor(nn.Module):
    # Regress 1-channel saliency map with values in <0,1> range from the input in_channels feature map
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, in_channels // reduction),
                                 ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(in_channels // reduction, 1),
                                 ME.MinkowskiSigmoid())

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.net(x)


class KeypointDescriptor(nn.Module):
    # Keypoint descriptor decoder
    def __init__(self, in_channels: int, out_channels: int, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        middle_channels = out_channels + (in_channels - out_channels) // 2
        self.normalize = normalize
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, middle_channels),
                                 ME.MinkowskiReLU(inplace=True),
                                 ME.MinkowskiLinear(middle_channels, out_channels))

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.net(x)

        if self.normalize:
            x = ME.MinkowskiFunctional.normalize(x)

        return x


class MinkG(nn.Module):
    def __init__(self, trunk: MinkTrunk, global_head: MinkHead, global_pool_method: str ='GeM',
                 global_normalize: bool = False):
        super().__init__()
        self.trunk = trunk

        self.global_head = global_head
        self.global_pool_method = global_pool_method

        self.global_channels = self.global_head.out_channels
        self.global_pooling = pooling.PoolingWrapper(pool_method=self.global_pool_method,
                                                     in_dim=self.global_channels, output_dim=self.global_channels)
        self.global_normalize = global_normalize

    def forward(self, batch: Dict[str, torch.tensor]):
        x = ME.SparseTensor(batch['lidar_features'], coordinates=batch['lidar_coords'])
        x = self.trunk(x)

        x = self.global_head(x)
        x = self.global_pooling(x)        # (batch_size, features) tensor
        if self.global_normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        return x

    def print_info(self):
        print(f'Model class: {type(self).__name__}')
        n_params = sum([param.nelement() for param in self.parameters()])
        n_trunk_params = sum([param.nelement() for param in self.trunk.parameters()])
        print(f"# parameters - total: {n_params/1000:.1f}   trunk: {n_trunk_params/1000:.1f} [k]")
        n_global_head = sum([param.nelement() for param in self.global_head.parameters()])
        print(f'global descriptor head: {n_global_head/1000:.1f} [k]   pool method: {self.global_pool_method}')
