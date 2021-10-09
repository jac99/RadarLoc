# Based on Pytorch resnet implementation
# FPN feature extractor based on ResNet architecture for cylindrical radar scan processing.
# 1 input channel and optional cylindrical padding for angle dimensionality

from torch import Tensor
import torch.nn as nn
from typing import Type, Union, List
from models.radar.cylindrical_resnet import conv1x1, my_conv, BasicBlock, Bottleneck


class FPNFeatureExtractor(nn.Module):
    def __init__(self, planes: List[int], layers: List[int] = None, conv0_kernel_size: int = 5,
                 block=BasicBlock, min_out_level: int = 1, out_channels: int = 256, cylindrical_padding: bool = False):
        super().__init__()
        assert len(planes) == len(layers)

        self.in_channels = 1
        self.planes = planes
        self.layers = layers
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.min_out_level = min_out_level
        self.cylindrical_padding = cylindrical_padding
        self.out_channels = out_channels
        if self.cylindrical_padding:
            padding_mode = 'cylindrical'
        else:
            padding_mode = 'constant'

        # LAYERS IN BOTTOM-UP BLOCK

        # First convolutional layer has the same number of filters as the first block
        self.inplanes = planes[0]
        self.conv0 = my_conv(self.in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, stride=2,
                             padding_mode=padding_mode)
        self.bn0 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleDict()  # Bottom-up blocks
        for ndx, (plane, layer) in enumerate(zip(self.planes, self.layers)):
            self.blocks[str(ndx + 1)] = self._make_layer(block, planes=plane, blocks=layer, stride=2, padding_mode=padding_mode)

        # LAYERS IN TOP-DOWN BLOCK

        self.conv1x1 = nn.ModuleDict()   # 1x1 convolutions in lateral connections
        self.tconv = nn.ModuleDict()    # Top-down transposed convolutions

        for in_level in range(self.min_out_level + 1, len(layers)+1):
            # There's one transposed convolution for each level, except for min_level
            self.tconv[str(in_level)] = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2,
                                                           stride=2)

        for in_level in range(self.min_out_level, len(layers)+1):
            # Create lateral 1x1x convolution for each input level
            self.conv1x1[str(in_level)] = conv1x1(self.planes[in_level-1], self.out_channels, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, padding_mode: str = 'constant') -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, padding_mode=padding_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding_mode=padding_mode))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        y = {}
        for i in range(1, len(self.blocks) + 1):
            x = self.blocks[str(i)](x)
            if i >= self.min_out_level:
                y[i] = x

        # TOP-DOWN PASS
        top_level = len(self.planes)
        x = self.conv1x1[str(top_level)](y[top_level])
        for level in range(top_level - 1, self.min_out_level - 1, -1):
            x = self.tconv[str(level + 1)](x)
            # Add the feature map from the corresponding level at bottom-up pass using a skip connection
            x = x + self.conv1x1[str(level)](y[level])

        assert x.shape[1] == self.out_channels
        return x


