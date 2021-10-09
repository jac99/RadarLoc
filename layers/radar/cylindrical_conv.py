from torch import nn as nn, Tensor
from typing import Union, List


class CylindricalPad(nn.Module):
    # Circular padding of the last coordinate (width - which corresponds to the angular dimension) and
    # constant padding of height coordinate
    def __init__(self, pad: Union[int, List[int]]):
        super().__init__()
        if isinstance(pad, int):
            self.pad = [pad, pad, pad, pad]
        elif isinstance(pad, list):
            assert len(pad) == 4
            self.pad = pad
        else:
            raise NotImplementedError(f"Incorrect padding: {pad}")

    def forward(self, x: Tensor) -> Tensor:
        # Circular padding along the last coordinate (angular). Padding values start from the last coord
        x = nn.functional.pad(x, (self.pad[0], self.pad[1], 0, 0), mode='circular')
        x = nn.functional.pad(x, (0, 0, self.pad[2], self.pad[3]), mode='constant')
        return x


def get_conv(in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding_mode: str = 'constant') -> nn.Module:
    padding = int(kernel_size // 2)
    if padding_mode == 'constant':
        block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
    elif padding_mode == 'cylindrical':
        # Cylindrical convolution = circluar pad along azimuth direction and constant zero padding along range direction
        block = nn.Sequential(
            CylindricalPad(pad=padding),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False))
    else:
        raise NotImplementedError(f'Unknown padding mode: {padding_mode}')
    return block
