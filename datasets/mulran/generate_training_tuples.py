# Training tuples generation for MulRan dataset.
# The input is a list of traversals.
# We iterate over all scans from all traversals or
# alternatively we consider scans that are at least x meters away from the previous scan.
# For each anchor, we find positives and negatives in all traversals (including the anchor traversal).

import numpy as np
import argparse
import tqdm
import pickle
import os

from datasets.mulran.mulran_raw import MulranSequences
from datasets.base_datasets import TrainingTuple

DEBUG = False


def generate_training_tuples(ds: MulranSequences, pos_threshold: float = 10, neg_threshold: float = 50):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        anchor_pos = ds.get_xy()[anchor_ndx]

        # Find timestamps of positive and negative elements
        positives = ds.find_neighbours_ndx(anchor_pos, pos_threshold)
        non_negatives = ds.find_neighbours_ndx(anchor_pos, neg_threshold)
        # Remove anchor element from positives, but leave it in non_negatives
        positives = positives[positives != anchor_ndx]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_reading_filepath=ds.rel_reading_filepath[anchor_ndx],
                                           sensor_code=ds.sensor_code[anchor_ndx], positives=positives,
                                           non_negatives=non_negatives, position=anchor_pos)

    print(f'{len(tuples)} tuples generated')

    return tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', type=str, required=True)
    # Thresholds as in PointNetVLAD: we define structurally similar point clouds to be at most 10m apart
    # and those structurally dissimilar to be at least 50m apart
    parser.add_argument('--pos_threshold', default=5)
    parser.add_argument('--neg_threshold', default=20)
    parser.add_argument('--min_displacement', type=float, default=0.1)
    args = parser.parse_args()

    sequences = ['Sejong01', 'Sejong02']
    if DEBUG:
        sequences = ['ParkingLot', 'ParkingLot']

    print(f'Dataset root: {args.dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')

    for use_radar, use_lidar in [(True, False), (False, True)]:
        print(f'Use radar: {use_radar}   Use lidar: {use_lidar}')
        if use_radar:
            sensor_desc = 'R'
            # Folder with downsampled radar scan imgaes
            radar_scan_folder = 'polar_384_128'
        elif use_lidar:
            sensor_desc = 'L'
            radar_scan_folder = None
        else:
            raise NotImplementedError("Only one sensor can be used")

        ds = MulranSequences(args.dataset_root, sequences, split='train', min_displacement=args.min_displacement,
                             use_radar=use_radar, use_lidar=use_lidar, polar_folder=radar_scan_folder)
        train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
        pickle_name = f'train_{sensor_desc}_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
        train_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
        pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))

        ds = MulranSequences(args.dataset_root, sequences, split='test', min_displacement=args.min_displacement,
                             use_radar=use_radar, use_lidar=use_lidar, polar_folder=radar_scan_folder)
        test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)

        # Validation data
        pickle_name = f'val_{sensor_desc}_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
        test_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
        pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))

        print('')
