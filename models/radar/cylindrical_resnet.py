# Based on Pytorch resnet implementation
# ResNet-based feature extractor for cylindrical radar scan processing.
# 1 input channel and circular padding for angle dimensionality

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Union, List, Optional


class CylindricalPad(nn.Module):
    # Circular padding of the last coordinate (width - which corresponds to the angular dimension)
    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        return x


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def my_conv(in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding_mode: str = 'constant') -> nn.Module:
    padding = int(kernel_size // 2)
    if padding_mode == 'constant':
        block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
    elif padding_mode == 'cylindrical':
        block = nn.Sequential(
            CylindricalPad(pad=padding),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(padding, 0), stride=stride, bias=False))
    else:
        raise NotImplementedError(f'Unknown padding mode: {padding_mode}')
    return block


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        padding_mode: str = 'constant'
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = my_conv(inplanes, planes, 3, stride, padding_mode=padding_mode)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = my_conv(planes, planes, 3, padding_mode=padding_mode)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        padding_mode: str = 'constant'
    ) -> None:
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        base_width = 64
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = my_conv(width, width, stride, padding_mode=padding_mode)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], planes: List[int],
                 cylindrical_padding=False) -> None:
        super().__init__()
        assert len(layers) == len(planes)
        assert len(layers) == 4

        self.cylindrical_padding = cylindrical_padding
        if self.cylindrical_padding:
            padding_mode = 'cylindrical'
        else:
            padding_mode = 'constant'

        self.inplanes = 64
        self.base_width = 64
        self.norm_layer = nn.BatchNorm2d
        # Single input channel
        self.conv1 = my_conv(1, self.inplanes, kernel_size=7, stride=2, padding_mode=padding_mode)

        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], padding_mode=padding_mode)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, padding_mode=padding_mode)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, padding_mode=padding_mode)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, padding_mode: str = 'constant') -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, padding_mode=padding_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding_mode=padding_mode))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18(**kwargs: Any) -> ResNetFeatureExtractor:
    planes = [32, 64, 128, 256]
    return ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2], planes, **kwargs)


def resnet34(**kwargs: Any) -> ResNetFeatureExtractor:
    planes = [32, 64, 128, 256]
    return ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3], planes, **kwargs)


def resnet50(**kwargs: Any) -> ResNetFeatureExtractor:
    planes = [32, 64, 128, 256]
    return ResNetFeatureExtractor(Bottleneck, [3, 4, 6, 3], planes, **kwargs)


def resnet101(**kwargs: Any) -> ResNetFeatureExtractor:
    planes = [32, 64, 128, 256]
    return ResNetFeatureExtractor(Bottleneck, [3, 4, 23, 3], planes, **kwargs)


def resnet152(**kwargs: Any) -> ResNetFeatureExtractor:
    planes = [32, 64, 128, 256]
    return ResNetFeatureExtractor(Bottleneck, [3, 8, 36, 3], planes, **kwargs)
