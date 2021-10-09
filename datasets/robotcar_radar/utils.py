import os
import csv
from bisect import bisect_left
from typing import List
import numpy as np
from scipy.spatial import distance_matrix


# Good traversals with less than 10% of GPS-INS mismatch
TRAVERSALS = ['2019-01-18-15-20-12-radar-oxford-10k',
              '2019-01-18-14-14-42-radar-oxford-10k',
              '2019-01-11-12-26-55-radar-oxford-10k',
              '2019-01-14-14-15-12-radar-oxford-10k',
              '2019-01-17-14-03-00-radar-oxford-10k',
              '2019-01-17-12-48-25-radar-oxford-10k',
              '2019-01-15-13-06-37-radar-oxford-10k',
              '2019-01-17-13-26-39-radar-oxford-10k',
              '2019-01-14-12-41-28-radar-oxford-10k',
              '2019-01-11-13-24-51-radar-oxford-10k',
              '2019-01-15-14-24-38-radar-oxford-10k',
              '2019-01-11-14-37-14-radar-oxford-10k',
              '2019-01-18-12-42-34-radar-oxford-10k',
              '2019-01-10-14-50-05-radar-oxford-10k',
              '2019-01-16-13-09-37-radar-oxford-10k',
              '2019-01-16-14-15-33-radar-oxford-10k']


# For training and test data split - taken from PointNetVLAD. They used radius=150, we use larger radius.
# TEST_RADIUS = 150
TEST_RADIUS = 200
TEST_CENTERS = np.array([[5735712.768124, 620084.402381], [5735611.299219, 620540.270327],
                         [5735237.358209, 620543.094379], [5734749.303802, 619932.693364]])


def load_poses(dataset_root: str, traversal: str, pose_source: str = 'INS'):
    ins_status_l2 = []
    positions_l2 = []
    timestamps_l2 = []
    if pose_source == 'INS':
        # INS + GPS poses
        pose_path = os.path.join(dataset_root, traversal, 'gps', 'ins.csv')
    elif pose_source == 'GPS':
        raise NotImplementedError()
    else:
        raise NotImplementedError(f"Unknown pose source : {pose_source}")

    with open(pose_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', )
        for ndx, row in enumerate(reader):
            if ndx == 0:
                # Skip the header
                assert row == ['timestamp', 'ins_status', 'latitude', 'longitude', 'altitude', 'northing',
                               'easting', 'down', 'utm_zone', 'velocity_north', 'velocity_east',
                               'velocity_down', 'roll', 'pitch', 'yaw']
                continue
            timestamp = int(row[0])
            ins_status = row[1]
            northing = float(row[5])
            easting = float(row[6])
            ins_status_l2.append(ins_status)
            positions_l2.append((northing, easting))
            timestamps_l2.append(timestamp)

    return timestamps_l2, positions_l2, ins_status_l2


def find_closest_ndx(sorted_list: List[int], e: int):
    # Find index of the element closest to e in a sorted list of integers
    pos = bisect_left(sorted_list, e)
    if pos == 0:
        closest_ndx = 0
    elif pos == len(sorted_list):
        closest_ndx = len(sorted_list) - 1
    else:
        before = sorted_list[pos - 1]
        after = sorted_list[pos]
        closest_ndx = pos if after - e < e - before else pos - 1

    delta = sorted_list[closest_ndx] - e if sorted_list[closest_ndx] > e else e - sorted_list[closest_ndx]
    return closest_ndx, delta


def format_time(ts: int):
    # Expects time in microseconds (as 16 digit)
    ms = ts / 1000   # Time in miliseconds
    sec = ms // 1000
    hours = sec // 3600
    days = hours // 24
    # Remaining hours, seconds and ms
    hours = hours % 24
    sec = sec % (3600*24)
    ms = ms % (3600*24)
    return "{}d {}h {}s {}ms".format(days, hours, sec, ms)


def in_train_split(pos: np.ndarray) -> bool:
    # pos: (N, 2) array
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_CENTERS)
    mask = (dist > TEST_RADIUS).all(axis=1)
    return mask


def in_test_split(pos: np.ndarray) -> bool:
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_CENTERS)
    mask = (dist <= TEST_RADIUS).any(axis=1)
    return mask
