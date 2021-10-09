# Warsaw University of Technology

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import TrainingDataset, EvaluationTuple
from datasets.augmentation import RadarTrainTransform, RadarEvalTransform, RadarTrainSetTransform, LidarTrainTransform, \
    LidarTrainSetTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams


def make_datasets(params: TrainingParams):
    # Create training and validation datasets
    datasets = {}
    datasets['train'] = TrainingDataset(params.dataset_folder, params.dataset, params.train_file,
                                        radar_transform=RadarTrainTransform(params.aug_mode),
                                        radar_set_transform=RadarTrainSetTransform(params.aug_mode),
                                        lidar_transform=LidarTrainTransform(params.aug_mode),
                                        lidar_set_transform=LidarTrainSetTransform(params.aug_mode))
    if len(params.val_file) > 0:
        datasets['val'] = TrainingDataset(params.dataset_folder, params.dataset, params.val_file,
                                          radar_transform=RadarEvalTransform())
    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    def collate_fn(data_list):
        # data_list: list of triplets(reading, id, sensor_code)
        lidar = [e for e in data_list if e[2] == 'L']
        radar = [e for e in data_list if e[2] == 'R']

        batch = {}
        # PROCESS LIDAR SCANS
        lidar_readings = [e[0] for e in lidar]
        lidar_indices = [e[1] for e in lidar]

        if len(lidar_indices) > 0:
            if dataset.lidar_set_transform is not None:
                # Apply the same transformation on all dataset elements
                lens = [len(e) for e in lidar_readings]
                lidar_readings = torch.cat(lidar_readings, dim=0)
                lidar_readings = dataset.lidar_set_transform(lidar_readings)
                lidar_readings = lidar_readings.split(lens)

            # Convert to polar (when polar coords are used) and quantize
            # Use the first value returned by quantizer
            lidar_coords = [quantizer(e)[0] for e in lidar_readings]
            lidar_coords = ME.utils.batched_coordinates(lidar_coords)

            # Assign a dummy feature equal to 1 to each point
            batch['lidar_features'] = torch.ones((lidar_coords.shape[0], 1), dtype=torch.float)
            batch['lidar_coords'] = lidar_coords

        # PROCESS RADAR SCANS
        radar_batch = [e[0] for e in radar]
        radar_indices = [e[1] for e in radar]

        if len(radar_indices) > 0:
            radar_batch = torch.stack(radar_batch, dim=0)
            if dataset.radar_set_transform is not None:
                # Apply the same transformation on all dataset elements
                radar_batch = dataset.radar_set_transform(radar_batch)
            batch['radar_batch'] = radar_batch
        # We assume radar scans are first then lidar scans
        indices = radar_indices + lidar_indices

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in indices] for label in indices]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in indices] for
                          label in indices]

        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)
        batch['positives_mask'] = positives_mask
        batch['negatives_mask'] = negatives_mask

        # Positives_mask and negatives_mask which are batch_size x batch_size boolean tensors
        # Masks assume that first elements are radar scans, followed by lidar scans
        return batch

    return collate_fn


def make_dataloaders(params: TrainingParams):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    if params.model_params.lidar_model is not None:
        quantizer = params.model_params.lidar_quantizer
    else:
        quantizer = None

    train_collate_fn = make_collate_fn(datasets['train'],  quantizer)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.batch_size_limit)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler,
                                       collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        assert e.position.shape == (2,)
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray):
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
