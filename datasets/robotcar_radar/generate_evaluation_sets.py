# Test sets for Oxford RobotCar Radar dataset.

import numpy as np
import argparse
from typing import List
import os

from datasets.robotcar_radar.robotcar_raw import RobotCarDataset
from datasets.base_datasets import EvaluationTuple, EvaluationSet
from datasets.dataset_utils import filter_query_elements

DEBUG = False


def get_readings(ds: RobotCarDataset) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    count_skipped = 0
    for timestamp, scan_file, traversal_name, position, sensor_code in ds:
        item = EvaluationTuple(timestamp, scan_file, sensor_code,position)
        elems.append(item)

    return elems


def generate_evaluation_set(map_ds: RobotCarDataset, query_ds: RobotCarDataset, dist_threshold=20) -> EvaluationSet:

    map_set = get_readings(map_ds)
    query_set = get_readings(query_ds)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Oxford RobotCar Radar dataset')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--min_displacement', type=float, default=0.1)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=20)
    args = parser.parse_args()

    # Sequences is a list of (map sequence, query sequence) splits
    # = [('2019-01-18-15-20-12-radar-oxford-10k', '2019-01-18-14-14-42-radar-oxford-10k')]
    sequences = [('2019-01-15-13-06-37-radar-oxford-10k', '2019-01-18-14-14-42-radar-oxford-10k')]

    print(f'Dataset root: {args.dataset_root}')

    for map_sequence, query_sequence in sequences:
        if DEBUG:
            map_sequence = '2019-01-10-14-36-48-radar-oxford-10k-partial'
            query_sequence = '2019-01-10-14-36-48-radar-oxford-10k-partial'
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')
        print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
        print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

        # Evaluation datasets with downsampled radar scans for RadarLoc evaluation
        map_ds = RobotCarDataset(args.dataset_root, [map_sequence], split='all',
                                 min_displacemenet=args.min_displacement, use_radar=True, use_lidar=False,
                                 radar_subfolder='polar_384_128')
        query_ds = RobotCarDataset(args.dataset_root, [query_sequence], split='all',
                                   min_displacemenet=args.min_displacement, use_radar=True, use_lidar=False,
                                   radar_subfolder='polar_384_128')

        test_set = generate_evaluation_set(map_ds, query_ds, dist_threshold=args.dist_threshold)
        pickle_name = f'test_R_{map_sequence}_{query_sequence}_384_128.pickle'

        file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)

        # Evaluation datasets with downsampled radar scans for ScanContext evaluation
        map_ds = RobotCarDataset(args.dataset_root, [map_sequence], split='all',
                                 min_displacemenet=args.min_displacement, use_radar=True, use_lidar=False,
                                 radar_subfolder='polar_120_40')
        query_ds = RobotCarDataset(args.dataset_root, [query_sequence], split='all',
                                   min_displacemenet=args.min_displacement, use_radar=True, use_lidar=False,
                                   radar_subfolder='polar_120_40')

        test_set = generate_evaluation_set(map_ds, query_ds, dist_threshold=args.dist_threshold)
        pickle_name = f'test_R_{map_sequence}_{query_sequence}_120_40.pickle'

        file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)

        # Evaluation datasets with LiDAR point clouds
        map_ds = RobotCarDataset(args.dataset_root, [map_sequence], split='all',
                                 min_displacemenet=args.min_displacement, use_radar=False, use_lidar=True)
        query_ds = RobotCarDataset(args.dataset_root, [query_sequence], split='all',
                                   min_displacemenet=args.min_displacement, use_radar=False, use_lidar=True)

        test_set = generate_evaluation_set(map_ds, query_ds, dist_threshold=args.dist_threshold)
        pickle_name = f'test_L_{map_sequence}_{query_sequence}.pickle'

        file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)

        print('')
