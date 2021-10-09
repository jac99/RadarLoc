# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import Set, List
from typing import Dict
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from datasets.mulran.mulran_raw import MulranPointCloudLoader
from datasets.robotcar_radar.robotcar_raw import RobotCarPointCloudLoader
from misc.point_clouds import PointCloudLoader


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_reading_filepath: str, sensor_code: str,
                 positives: np.ndarray, non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: list of positive elements id
        # negatives: list of negative elements id
        # position: (northing, easting)
        assert sensor_code == 'L' or sensor_code == 'R', f"Unknown sensor type: {sensor_code}"
        assert position.shape == (2,)
        self.id = id
        self.timestamp = timestamp
        self.rel_reading_filepath = rel_reading_filepath
        self.sensor_code = sensor_code
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_reading_filepath: str, sensor_code: str, position: np.ndarray):
        # position: (northing, easting)
        assert sensor_code == 'L' or sensor_code == 'R', f"Unknown sensor type: {sensor_code}"
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_reading_filepath = rel_reading_filepath
        self.sensor_code = sensor_code
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_reading_filepath, self.sensor_code, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path:str, dataset_type: str, query_filename: str,
                 radar_transform, radar_set_transform=None,
                 lidar_transform=None, lidar_set_transform=None):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        # Transforms for Lidar scans and radar images
        self.lidar_transform = lidar_transform
        self.lidar_set_transform = lidar_set_transform
        self.radar_transform = radar_transform
        self.radar_set_transform = radar_set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        # pc_loader must be set in the inheriting class
        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx: int) -> (torch.Tensor, int, str):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_reading_filepath)
        sensor_code = self.queries[ndx].sensor_code
        if sensor_code == 'L':
            reading = self.pc_loader(file_pathname)
            if self.lidar_transform is not None:
                reading = self.lidar_transform(reading)
        elif sensor_code == 'R':
            # Radar image reading. Returns (C, H, W) tensor
            reading = Image.open(file_pathname)
            # Radar transform cannot be None, image must be converted to tensor
            reading = self.radar_transform(reading)
            # reading is (1, height=angle, width=range) tensor
        else:
            raise NotImplementedError(f"Unknown sensor type: {sensor_code}")

        return reading, ndx, sensor_code

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            if pos.position.shape == (2,):
                # pose given as (northing, easting)
                positions[ndx] = pos.position
            else:
                # Full SE(3) pose
                positions[ndx] = pos.position[:2, 3]
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            if pos.position.shape == (2,):
                # pose given as (northing, easting)
                positions[ndx] = pos.position
            else:
                # Full SE(3) pose
                positions[ndx] = pos.position[:2, 3]
        return positions


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if dataset_type == 'mulran':
        return MulranPointCloudLoader()
    elif dataset_type == 'robotcar':
        return RobotCarPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
