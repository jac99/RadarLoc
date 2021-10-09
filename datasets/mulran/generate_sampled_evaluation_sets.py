# Subsampled test sets for Mulran dataset, where the number of LiDAR scans is similar to the number of Radar images
# Readings from Sejong01 test area form database (map) and readings from Sejong02 test area are queries

import argparse
from typing import List
import os

from datasets.mulran.mulran_raw import MulranSequence
from datasets.base_datasets import EvaluationTuple, EvaluationSet
from datasets.dataset_utils import filter_query_elements

DEBUG = False


def get_readings(sequence: MulranSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_reading_filepath[ndx], sequence.sensor_code[ndx],
                               position)
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, min_displacement: float = 0.2,
                            dist_threshold=20,  map_sensor: str = 'R', query_sensor: str = 'R',
                            polar_folder: str = 'polar_384_128') -> EvaluationSet:
    assert map_sensor in ['R', 'L']
    assert query_sensor in ['R', 'L']

    map_sequence = MulranSequence(dataset_root, map_sequence, split='test', min_displacement=min_displacement,
                                  use_radar=map_sensor=='R', use_lidar=map_sensor=='L', polar_folder=polar_folder)
    query_sequence = MulranSequence(dataset_root, query_sequence, split='test', min_displacement=min_displacement,
                                    use_radar=query_sensor=='R', use_lidar=query_sensor=='L', polar_folder=polar_folder)

    map_set = get_readings(map_sequence)
    query_set = get_readings(query_sequence)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Mulran dataset')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--min_displacement', type=float, default=0.1)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=20)
    args = parser.parse_args()

    # Sequences is a list of (map sequence, query sequence) splits
    sequences = [('Sejong01', 'Sejong02'), ('KAIST01', 'KAIST02'), ('Riverside01', 'Riverside02')]
    print(f'Dataset root: {args.dataset_root}')

    if DEBUG:
        sequences = [('ParkingLot', 'ParkingLot')]

    for map_sequence, query_sequence in sequences:

        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')
        print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
        print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

        test_set = generate_evaluation_set(args.dataset_root, map_sequence, query_sequence,
                                           min_displacement=args.min_displacement, map_sensor='L', query_sensor='L',
                                           dist_threshold=args.dist_threshold)
        test_set.map_set = test_set.map_set[::2]
        test_set.query_set = test_set.query_set[::2]
        print(f"Sampled eval set -- map: {len(test_set.map_set)}   query: {len(test_set.query_set)}")
        pickle_name = f'test_L2L_{map_sequence}_{query_sequence}_reduced.pickle'

        file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)
        print('')
