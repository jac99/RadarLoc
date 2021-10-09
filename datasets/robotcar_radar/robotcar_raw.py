import os
import numpy as np
import argparse
from typing import List, Dict
from sklearn.neighbors import KDTree
import random

from datasets.robotcar_radar.utils import load_poses, find_closest_ndx, in_test_split, in_train_split
from misc.point_clouds import PointCloudLoader

DEBUG = False


class RobotCarPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        # According to RoborCar Radar documentation LiDAR are 1.69 from the ground and z is facing downward
        self.ground_plane_level = -1.6

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = np.fromfile(file_pathname, dtype=np.float32)
        pc = pc.reshape((4, -1))[:3]
        pc = np.transpose(pc, (1, 0))
        # According to RoborCar Radar documentation LiDAR are 1.69 from the ground and z is facing downward
        pc[:, 2] = -pc[:, 2]
        return pc


class RobotCarDataset:
    # Oxford RobotCar dataset
    def __init__(self, dataset_root: str, traversals: List[str], split: str, min_displacemenet: float = 0.2,
                 use_radar: bool = True, use_lidar: bool = True, max_radar: int = None, max_lidar: int = None,
                 radar_subfolder: str = None):
        # max_radar: maximum number of radar scans to retain
        # max_lidar: maximum number of lidar scans to retain
        assert split in ['train', 'test', 'all']

        assert use_radar or use_lidar
        print('Processing RobotCar Radar dataset: {}'.format(dataset_root))
        assert os.path.exists(dataset_root), 'Cannot find RobotCar Radar dataset: {}'.format(dataset_root)
        self.dataset_root = dataset_root
        self.traversals = [os.path.join(dataset_root, e) for e in traversals]
        self.split = split
        self.min_displacement = min_displacemenet
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.max_radar = max_radar
        self.max_lidar = max_lidar
        self.radar_subfolder = radar_subfolder
        self.lidar_subfolder = 'velodyne_left'
        self.radar_timestamps_file = 'radar.timestamps'
        self.lidar_timestamps_file = 'velodyne_left.timestamps'
        self.ts_len = 16    # Timestamp length
        self.radar_file_ext = '.png'
        self.lidar_file_ext = '.bin'
        self.pose_source = 'INS'
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)
        self.pose_time_tolerance = 1.

        assert os.path.exists(self.dataset_root), f"Cannot access dataset root folder: {self.dataset_root}"

        # Load GPS/INS poses from pose_source
        # Timestamps are sorted in the increasing order
        self.sorted_gps_timestamps, self.sorted_gps_positions = self.load_poses()

        self.radar_ts_ndx = {}
        self.lidar_ts_ndx = {}

        for traversal in self.traversals:
            assert os.path.exists(traversal), f"Cannot access traversal: {traversal}"
            if self.use_radar:
                # radar_ts_nds[ts] = (radar_file, traversal_name, chunk)
                self.radar_ts_ndx.update(self.index_scans(traversal, self.radar_subfolder, self.radar_timestamps_file,
                                                          self.radar_file_ext))

            if self.use_lidar:
                # lidar_ts_nds[ts] = (lidar_file, traversal_name, chunk)
                self.lidar_ts_ndx.update(self.index_scans(traversal, self.lidar_subfolder, self.lidar_timestamps_file,
                                                          self.lidar_file_ext))

        if self.use_radar:
            self.radar_sorted_ts_list = list(self.radar_ts_ndx)
            self.radar_sorted_ts_list.sort()
        else:
            self.radar_sorted_ts_list = []

        if self.use_lidar:
            self.lidar_sorted_ts_list = list(self.lidar_ts_ndx)
            self.lidar_sorted_ts_list.sort()
        else:
            self.lidar_sorted_ts_list = []

        print('{} radar scans'.format(len(self.radar_ts_ndx)))
        print('{} lidar scans'.format(len(self.lidar_ts_ndx)))

        # Find positions (easting, northing) of radar scan images
        # Only scans that do not have pose within pose_tolerance are ignored
        self.radar_valid_ts, self.radar_scan_positions = self.find_scan_positions(self.radar_ts_ndx)
        assert len(self.radar_valid_ts) == len(self.radar_scan_positions)

        self.lidar_valid_ts, self.lidar_scan_positions = self.find_scan_positions(self.lidar_ts_ndx)
        assert len(self.lidar_valid_ts) == len(self.lidar_scan_positions)

        # Filter out scans - retain only scans within a given split
        if self.use_radar:
            self.radar_valid_ts, self.radar_scan_positions = self.filter(self.radar_valid_ts, self.radar_scan_positions,
                                                                         self.max_radar)

        if self.use_lidar:
            self.lidar_valid_ts, self.lidar_scan_positions = self.filter(self.lidar_valid_ts, self.lidar_scan_positions,
                                                                         self.max_lidar)

        print('{} filtered radar scans'.format(len(self.radar_valid_ts)))
        print('{} filtered lidar scans'.format(len(self.lidar_valid_ts)))

        # Radar readings preceed lidar readings
        if self.use_radar and self.use_lidar:
            self.all_scan_positions = np.concatenate([self.radar_scan_positions, self.lidar_scan_positions], axis=0)
        elif self.use_radar:
            self.all_scan_positions = self.radar_scan_positions
        elif self.use_lidar:
            self.all_scan_positions = self.lidar_scan_positions

        # Build a kdtree based on X, Y position of all scans (radar and lidar)
        self.kdtree = KDTree(self.all_scan_positions)

    def __len__(self):
        return len(self.radar_valid_ts) + len(self.lidar_valid_ts)

    def __getitem__(self, ndx):
        if ndx < len(self.radar_valid_ts):
            # Get radar scan
            timestamp = self.radar_valid_ts[ndx]
            position = self.radar_scan_positions[ndx]
            scan_file, traversal_name, _ = self.radar_ts_ndx[timestamp]
            sensor_code = 'R'
        else:
            # Get lidar scan (indexed from 0)
            ndx -= len(self.radar_valid_ts)
            timestamp = self.lidar_valid_ts[ndx]
            position = self.lidar_scan_positions[ndx]
            scan_file, traversal_name, _ = self.lidar_ts_ndx[timestamp]
            sensor_code = 'L'

        return timestamp, scan_file, traversal_name, position, sensor_code

    def filter(self, ts: np.ndarray, positions: np.ndarray, max_scans: int):
        # Filter out scans - retain only scans within a given split with minimum displacement and limit the maximum
        # number of elements if needed

        # Retain elements in the given split
        if self.split != 'all':
            if self.split == 'train':
                mask = in_train_split(positions)
            elif self.split == 'test':
                mask = in_test_split(positions)

            ts = ts[mask]
            positions = positions[mask]

        # Filter out scans - retain only scans within a given split
        displacement = np.linalg.norm(positions[:-1] - positions[1:], axis=1)
        mask = displacement > self.min_displacement
        # Mask is calculated for elements starting from index 1. Add True at zero position
        mask = np.concatenate([np.array([True]), mask])
        ts = ts[mask]
        positions = positions[mask]

        # Limit the maximum number of elements
        if max_scans is not None and len(ts) > max_scans:
            retain_ndx = random.sample(range(len(ts)), k=max_scans)
            retain_ndx.sort()
            ts = ts[retain_ndx]
            positions = positions[retain_ndx]

        return ts, positions

    def index_scans(self, traversal: str, subfolder: str, timestamps_file: str, ext: str):
        rel_traversal = os.path.split(traversal)[1]
        print(f'Indexing scans in: {traversal}/{subfolder}...', end='')
        scans_path = os.path.join(traversal, subfolder)
        if not os.path.exists(scans_path):
            print('WARNING: Cannot access scans folder: {}'.format(scans_path))
            return

        timestamp_filepath = os.path.join(traversal, timestamps_file)
        if not os.path.exists(timestamp_filepath):
            print('WARNING: Cannot find timestamps file: {}'.format(timestamp_filepath))
            return

        # Process timestamp file
        with open(timestamp_filepath, 'r') as file:
            content = file.readlines()
            content = [x.rstrip('\n') for x in content]

        count_scans = 0
        ts_ndx = {}
        for line in content:
            assert line[self.ts_len] == ' ', 'Incorrect line in a timestamp file: {}'.format(line)
            ts = int(line[:self.ts_len])
            chunk = line[self.ts_len + 1:]
            scan_file_path = os.path.join(traversal, subfolder, str(ts) + ext)
            if not os.path.exists(scan_file_path):
                print('WARNING: Cannot find scan file: {}'.format(scan_file_path))
                break

            rel_scan_file_path = os.path.join(rel_traversal, subfolder, str(ts) + ext)
            ts_ndx[ts] = (rel_scan_file_path, rel_traversal, chunk)
            count_scans += 1

        print(' {} scans found'.format(count_scans))
        return ts_ndx

    def find_closest_pose(self, ts: int):
        # Find the closest pose given the timestamp
        closest_ndx, delta = find_closest_ndx(self.sorted_gps_timestamps, ts)
        return self.sorted_gps_positions[closest_ndx], delta

    def load_poses(self):
        # Load poses from all traversals
        positions_l1 = []
        timestamps_l1 = []

        for traversal in self.traversals:
            timestamps_l2, positions_l2, ins_status_l2 = load_poses(self.dataset_root, traversal, self.pose_source)
            positions_l1.append(np.array(positions_l2))
            timestamps_l1.append(np.array(timestamps_l2))

        # Concatenate positions and timestamps from all traversals
        positions = np.concatenate(positions_l1, axis=0)
        timestamps = np.concatenate(timestamps_l1, axis=0)

        # Sort by the timestamp
        ndx = np.argsort(timestamps)
        return timestamps[ndx], positions[ndx]

    def find_scan_positions(self, ts_ndx: Dict):
        # max_scans: Maximum number of scans to retain

        count_rejected = 0
        positions = []
        valid_ts = []
        for ts in ts_ndx:
            position, delta = self.find_closest_pose(ts)
            if delta >= self.pose_time_tolerance * 1000000000:
                count_rejected += 1
                continue

            valid_ts.append(ts)
            positions.append((position[0], position[1]))

        print(f'{len(positions)} scans with a pose within {self.pose_time_tolerance}s tolerance.'
              f' {count_rejected} scans without a valid pose.')

        valid_ts = np.array(valid_ts)
        positions = np.array(positions)

        return valid_ts, positions

    def find_neighbours_ndx(self, position: np.ndarray, radius: float):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    sequences = ['2019-01-10-14-36-48-radar-oxford-10k-partial']
    ds = RobotCarDataset(dataset_root=args.dataset_root, traversals=sequences)
