import math
import random

import numpy as np
from typing import Tuple
import torch
from torch import Tensor
from scipy.linalg import expm, norm
from torchvision import transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

RADAR_SCAN_MEAN = torch.tensor([0.031])
RADAR_SCAN_STD = torch.tensor([0.048])


def to_image(tensor_image: Tensor, adjust_brightness:bool = False) -> Image:
    # adjust_brigthness: adjust brightness to be in [0..1] range
    unnormalize = transforms.Normalize(-RADAR_SCAN_MEAN / RADAR_SCAN_STD, 1.0 / RADAR_SCAN_STD)
    toPIL = transforms.ToPILImage()
    tensor_image = unnormalize(tensor_image)
    if adjust_brightness:
        scale = 1. / torch.max(tensor_image)
        tensor_image *= scale
    image = toPIL(tensor_image)
    return image

# RADAR IMAGES AUGMENTATIONS


class RadarTrainTransform:
    def __init__(self, aug_mode: int):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            # Baseline augmentation
            t = [transforms.ToTensor(), MyRandomErasing(), RandomNoise('speckle'),
                 transforms.Normalize(mean=RADAR_SCAN_MEAN, std=RADAR_SCAN_STD)]
        elif self.aug_mode == 2:
            # With 180 degree random rotations
            t = [transforms.ToTensor(), RandomRoll(max_theta=180), MyRandomErasing(), RandomNoise('speckle'),
                 transforms.Normalize(mean=RADAR_SCAN_MEAN, std=RADAR_SCAN_STD)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RadarEvalTransform:
    def __init__(self):
        t = [transforms.ToTensor(), transforms.Normalize(mean=RADAR_SCAN_MEAN, std=RADAR_SCAN_STD)]
        self.transform = transforms.Compose(t)

    def __call__(self, e: Image) -> Tensor:
        e = self.transform(e)
        return e


class RadarEvalTransformWithRotation:
    def __init__(self):
        t = [transforms.ToTensor(), RandomRoll(max_theta=180),
             transforms.Normalize(mean=RADAR_SCAN_MEAN, std=RADAR_SCAN_STD)]
        self.transform = transforms.Compose(t)

    def __call__(self, e: Image) -> Tensor:
        e = self.transform(e)
        return e


class RadarTrainSetTransform:
    def __init__(self, aug_mode: int):
        self.aug_mode = aug_mode
        t = [transforms.RandomHorizontalFlip()]
        self.transform = transforms.Compose(t)

    def __call__(self, e: Tensor) -> Tensor:
        e = self.transform(e)
        return e


class MyRandomErasing:
    # My version of random erasing: erases vertical strips
    def __init__(self, p=0.5, scale=(0.02, 0.33)):
        super().__init__()
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale

    @staticmethod
    def get_params(img: Tensor, scale: Tuple[float, float]) -> Tuple[int, int]:
        x1 = random.randrange(int(img.shape[-1] * (1.0 - scale[0])))
        delta = int(img.shape[-1]*(scale[0] + random.random() * (scale[1] - scale[0])))
        x2 = min(x1 + delta, img.shape[-1] - 1)
        return x1, x2

    def __call__(self, img):
        if torch.rand(1) < self.p:
            x1, x2 = self.get_params(img, scale=self.scale)
            if x2 > x1:
                img = F.erase(img, 0, x1, img.shape[-1] - 1, x2-x1, torch.tensor(0.), inplace=False)
        return img


class RandomNoise:
    # Roll along width (angular dimension)
    def __init__(self, mode: str = ''):
        self.mode = mode

    def __call__(self, x: Tensor) -> Tensor:
        if self.mode == 'speckle':
            # mean = 0, std = 0.2
            noise = torch.randn(x.shape) * 0.2
            x = x + x * noise
        elif self.mode == 'gaussian':
            noise = torch.randn(x.shape) * 0.1
            x = x + noise
        else:
            raise NotImplementedError(f"Unknown noise type: {self.mode}")

        return x


class RandomRoll:
    # Roll along width (angular dimension)
    def __init__(self, max_theta: int = 360):
        self.max_theta = max_theta

    def __call__(self, x: Tensor) -> Tensor :
        # x is (batch_size, height=128, width=400) tensor
        # rotate around width axis (400 angular dimenstions)
        width = x.shape[2]
        delta = int(np.random.rand(1) * width * self.max_theta / 360.)
        x = x.roll(delta, dims=2)
        return x


# LIDAR SCANS AUGMENTATIONS

class LidarTrainTransform:
    def __init__(self, aug_mode: int):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            # Baseline augmentation
            t = [LidarJitterPoints(sigma=0.1, clip=0.2), LidarRemoveRandomPoints(r=(0.0, 0.1)),
                 LidarRandomTranslation(max_delta=0.3), LidarRemoveRandomBlock(p=0.4)]
        elif self.aug_mode == 2:
            # With 180 degree random rotations
            t = [LidarJitterPoints(sigma=0.1, clip=0.2), LidarRemoveRandomPoints(r=(0.0, 0.1)),
                 LidarRandomTranslation(max_delta=0.3),
                 LidarRemoveRandomBlock(p=0.4),
                 LidarRandomRotation(max_theta=180, axis=np.array([0, 0, 1]))]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e: Image) -> Tensor:
        if self.transform is not None:
            e = self.transform(e)
        return e


class LidarTrainSetTransform:
    def __init__(self, aug_mode: int):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [LidarRandomRotation(max_theta=5, axis=np.array([0, 0, 1])),
                 LidarRandomFlip([0.25, 0.25, 0.])]
        elif self.aug_mode == 2:
            t = [LidarRandomFlip([0.25, 0.25, 0.])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e: Tensor) -> Tensor:
        if self.transform is not None:
            e = self.transform(e)
        return e


class LidarRandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class LidarRandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=None):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180.) * 2. * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class LidarRandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class LidarJitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class LidarRemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class LidarRemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords

