# Create images with downsampled radar scans

import argparse
import os
import cv2
import tqdm
import numpy as np


def subsample_scan(scan_path: str, dataset_type: str, output_file_path: str, size: (int, int)):
    assert os.path.exists(scan_path)

    width, height = size
    image = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    if dataset_type == 'mulran':
        assert image.shape == (3360, 400)
    elif dataset_type == 'robotcar':
        # Stripe control bytes
        image = image[:, 11:]
        # Transpose axis, to make the same format as in MulRan
        image = np.transpose(image, (1, 0))
        assert image.shape == (3768, 400)
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    downsampled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    file_name = os.path.split(scan_path)[1]
    cv2.imwrite(os.path.join(output_file_path, file_name), downsampled_image)

    temp_image = downsampled_image.astype(float) / 255.
    mean = np.mean(temp_image)
    std = np.std(temp_image)
    return mean, std


def downsample(dataset_root: str, dataset_type: str, size: (int, int), output_folder: str):
    # List of directories/traversals
    traversals = os.listdir(dataset_root)
    traversals = [os.path.join(dataset_root, e) for e in traversals]
    traversals = [e for e in traversals if os.path.isdir(e)]
    mean_l, std_l = [], []

    for t in traversals:
        if dataset_type == 'mulran':
            scan_folder = os.path.join(t, 'polar')
        elif dataset_type == 'robotcar':
            scan_folder = os.path.join(t, 'radar')
        else:
            raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

        if not os.path.exists(scan_folder):
            print(f"Subfolder: {scan_folder} does not exists")
            continue

        scans = os.listdir(scan_folder)
        scans = [os.path.join(scan_folder, e) for e in scans]
        scans = [e for e in scans if os.path.isfile(e) and os.path.splitext(e)[1] == '.png']
        print(f"{len(scans)} scans in traversal: {t}")

        # Create output folder
        output_file_path = os.path.join(t, output_folder)
        if not os.path.exists(output_file_path):
            os.mkdir(output_file_path)

        assert os.path.exists(output_file_path)

        for scan in tqdm.tqdm(scans):
            mean, std = subsample_scan(scan, dataset_type, output_file_path, size)
            mean_l.append(mean)
            std_l.append(std)

    print(f'Subsampled images mean: {np.mean(mean_l):0.3f}    std:{np.mean(std_l):0.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['mulran', 'robotcar'], default='mulran')
    args = parser.parse_args()

    print(f'Dataset root: {args.dataset_root}')
    print(f'Dataset: {args.dataset}')

    # (384, 128) is for our RadarLoc method, (120, 40) is for ScanContext
    sizes = ((384, 128), (120, 40))
    assert os.path.exists(args.dataset_root)
    for (width, height) in sizes:
        output_folder = f"polar_{width}_{height}"
        print(f"Rescaling to: {width} x {height} (width x height)")
        downsample(args.dataset_root, args.dataset, (width, height), output_folder)
        print('')

