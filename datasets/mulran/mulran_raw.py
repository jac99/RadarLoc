# Functions and classes operating on a raw Mulran dataset

import numpy as np
import itertools
import os
from typing import List
from sklearn.neighbors import KDTree
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision

from datasets.mulran.utils import read_poses, in_test_split, in_train_split
from misc.point_clouds import PointCloudLoader


class MulranPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -0.9

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = np.fromfile(file_pathname, dtype=np.float32)
        # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
        pc = np.reshape(pc, (-1, 4))[:, :3]
        return pc


class MulranSequence(Dataset):
    """
    Dataset returns readings (point cloud or a radar scan) from one sequence from a raw Mulran dataset
    """
    def __init__(self, dataset_root: str, sequence_name: str, split: str, min_displacement: float = 0.2,
                 use_radar: bool = True, use_lidar: bool = True, polar_folder: str = None):
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)

        assert use_lidar or use_radar
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        sequence_path = os.path.join(self.dataset_root, self.sequence_name)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'
        self.split = split
        self.min_displacement = min_displacement
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.polar_folder = polar_folder
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = 1.

        self.pose_file = os.path.join(sequence_path, 'global_pose.csv')
        assert os.path.exists(self.pose_file), f'Cannot access global pose file: {self.pose_file}'

        timestamps_l = []
        poses_l = []
        rel_filepath_l = []
        sensor_code_l = []

        if self.use_lidar:
            self.rel_lidar_path = os.path.join(self.sequence_name, 'Ouster')
            lidar_path = os.path.join(self.dataset_root, self.rel_lidar_path)
            assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
            self.pc_loader = MulranPointCloudLoader()

            timestamps, poses = read_poses(self.pose_file, lidar_path, 'L', self.pose_time_tolerance)
            timestamps, poses = self.filter(timestamps, poses)
            rel_filepath = [os.path.join(self.rel_lidar_path, str(e) + '.bin') for e in timestamps]
            sensor_code = ['L'] * len(timestamps)

            timestamps_l.append(timestamps)
            poses_l.append(poses)
            rel_filepath_l.append(rel_filepath)
            sensor_code_l.append(sensor_code)

        if self.use_radar:
            self.rel_radar_path = os.path.join(self.sequence_name, self.polar_folder)
            radar_path = os.path.join(self.dataset_root, self.rel_radar_path)
            assert os.path.exists(radar_path), f'Cannot access radar scans: {radar_path}'
            timestamps, poses = read_poses(self.pose_file, radar_path, 'R', self.pose_time_tolerance)
            timestamps, poses = self.filter(timestamps, poses)
            rel_filepath = [os.path.join(self.rel_radar_path, str(e) + '.png') for e in timestamps]
            sensor_code = ['R'] * len(timestamps)

            timestamps_l.append(timestamps)
            poses_l.append(poses)
            rel_filepath_l.append(rel_filepath)
            sensor_code_l.append(sensor_code)

        # Concatenate readings from different sensors
        self.timestamps = np.concatenate(timestamps_l, axis=0)
        self.poses = np.concatenate(poses_l, axis=0)
        self.rel_reading_filepath = list(itertools.chain.from_iterable(rel_filepath_l))
        self.sensor_code = list(itertools.chain.from_iterable(sensor_code_l))

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_reading_filepath)
        assert len(self.timestamps) == len(self.sensor_code)

    def __len__(self):
        return len(self.rel_reading_filepath)

    def __getitem__(self, ndx):
        reading_filepath = os.path.join(self.dataset_root, self.rel_reading_filepath[ndx])
        sensor_code = self.sensor_code[ndx]
        if sensor_code == 'L':
            # Radar reading
            reading = self.pc_loader(reading_filepath)
        elif sensor_code == 'R':
            # Radar image reading. Returns (C, H, W) tensor
            reading = torchvision.io.read_image(reading_filepath)
        else:
            raise NotImplementedError(f"Unknown sensor code: {sensor_code}")

        return {'reading': reading, 'pose': self.poses[ndx], 'ts': self.timestamps[ndx], 'sensor_code': sensor_code}

    def filter(self, ts: np.ndarray, poses: np.ndarray):
        # Filter out scans - retain only scans within a given split with minimum displacement
        positions = poses[:, :2, 3]

        # Retain elements in the given split
        # Only sejong sequence has train/test split
        if self.split != 'all' and self.sequence_name.lower()[:6] == 'sejong':
            if self.split == 'train':
                mask = in_train_split(positions)
            elif self.split == 'test':
                mask = in_test_split(positions)

            ts = ts[mask]
            poses = poses[mask]
            positions = positions[mask]
            #print(f'Split: {self.split}   Mask len: {len(mask)}   Mask True: {np.sum(mask)}')

        # Filter out scans - retain only scans within a given split
        displacement = np.linalg.norm(positions[:-1] - positions[1:], axis=1)
        mask = displacement > self.min_displacement
        # Mask is calculated for elements starting from index 1. Add True at zero position
        mask = np.concatenate([np.array([True]), mask])
        ts = ts[mask]
        poses = poses[mask]

        return ts, poses


class MulranSequences(Dataset):
    """
    Multiple Mulran sequences indexed as a single dataset. Each element is identified by a unique global index.
    """
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str, min_displacement: float = 0.2,
                 use_radar: bool = True, use_lidar: bool = True, polar_folder: str = None):
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)

        assert len(sequence_names) > 0
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split
        self.min_displacement = min_displacement
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.polar_folder = polar_folder

        sequences = []
        for seq_name in self.sequence_names:
            ds = MulranSequence(self.dataset_root, seq_name, split=split, min_displacement=min_displacement,
                                use_radar=use_radar, use_lidar=use_lidar, polar_folder=polar_folder)
            sequences.append(ds)

        self.dataset = ConcatDataset(sequences)

        # Concatenate positions from all sequences
        self.poses = np.zeros((len(self.dataset), 4, 4), dtype=np.float64)
        self.timestamps = np.zeros((len(self.dataset),), dtype=np.int64)
        self.rel_reading_filepath = []
        self.sensor_code = []

        for cum_size, ds in zip(self.dataset.cumulative_sizes, self.dataset.datasets):
            # Consolidated lidar positions, timestamps and relative filepaths
            self.poses[cum_size - len(ds): cum_size, :] = ds.poses
            self.timestamps[cum_size - len(ds): cum_size] = ds.timestamps
            self.rel_reading_filepath.extend(ds.rel_reading_filepath)
            self.sensor_code.extend(ds.sensor_code)

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_reading_filepath)
        assert len(self.timestamps) == len(self.sensor_code)

        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.get_xy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    def get_xy(self):
        # Get X, Y position from (4, 4) pose
        return self.poses[:, :2, 3]

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


if __name__ == '__main__':
    dataset_root = '/media/sf_Datasets/MulRan'
    sequence_names = ['ParkingLot']

    db = MulranSequences(dataset_root, sequence_names, use_radar=True, use_lidar=True)
    print(f'Number of scans in the sequence: {len(db)}')
    e = db[0]

    res = db.find_neighbours_ndx(e['pose'][:2, 3], radius=5)
    print(len(res))
    print('.')



