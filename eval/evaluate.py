import argparse
import numpy as np
import tqdm
import os
import random
from typing import List
import PIL.Image as Image
import torch.nn as nn

from datasets.quantization import Quantizer
import torch
import MinkowskiEngine as ME
from models.model_factory import model_factory
from misc.utils import ModelParams
from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from datasets.augmentation import RadarEvalTransform, RadarEvalTransformWithRotation, LidarRandomRotation


class Evaluator:
    def __init__(self, dataset_root: str, dataset: str, eval_set_pickle: str, device: str,
                 radius: List[float] = (5, 10), k: int = 50, quantizer: Quantizer = None, with_rotation: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # repeat_dist_th: list of distance thresholds for repeatability calculation
        # fitness_threshold: ransac estimated fitness threshold to use local features fo reranking step
        # n_k: list of number of keypoints to use as local features

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        assert dataset in ['mulran', 'robotcar']
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.quantizer = quantizer
        self.with_rotation = with_rotation

        if self.with_rotation:
            self.radar_transform = RadarEvalTransformWithRotation()
            self.lidar_transform = LidarRandomRotation(max_theta=360, max_theta2=None, axis=np.array([0, 0, 1]))
        else:
            self.radar_transform = RadarEvalTransform()
            self.lidar_transform = None

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        self.n_samples = len(self.eval_set.query_set)
        self.pc_loader = get_pointcloud_loader(self.dataset)

    def evaluate(self, model):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model: nn.Module):
        model.eval()

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            reading_filepath = os.path.join(self.dataset_root, e.rel_reading_filepath)
            assert os.path.exists(reading_filepath)
            if e.sensor_code == 'L':
                reading = self.pc_loader(reading_filepath)
                if self.lidar_transform is not None:
                    reading = self.lidar_transform(reading)
                coords, _ = self.quantizer(reading)
                bcoords = ME.utils.batched_coordinates([coords])
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
                batch = {'lidar_coords': bcoords.to(self.device), 'lidar_features': feats.to(self.device)}
            elif e.sensor_code == 'R':
                reading = Image.open(reading_filepath)
                # Radar transform cannot be None, image must be converted to tensor
                reading = self.radar_transform(reading)
                batch = {'radar_batch': reading.unsqueeze(0).to(self.device)}
            else:
                raise NotImplementedError(f"Unknown sensor code: {e.sensor_code}")

            with torch.no_grad():
                # Get cross-modal embeddgins
                embedding = model(batch)

            embedding = embedding.detach().cpu().numpy()
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings

    def print_results(self, stats):
        recall = stats['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ", end='')
            for x in recall[r]:
                print("{:0.3f}, ".format(x), end='')
            print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--dataset', type=str, required=True, choices=['mulran', 'robotcar'])
    parser.add_argument('--sensor', type=str, default='R', choices=['R', 'L'], help='R=radar, L=lidar')
    parser.add_argument('--radius',  nargs='+', type=int, default=[5, 10], help='True Positive thresholds in meters')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--with_rotation', dest='with_rotation', action='store_true')
    parser.set_defaults(with_rotation=False)

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Dataset: {args.dataset}')
    print(f'Sensor: {args.sensor}')
    print(f'Radius: {args.radius} [m]')
    print(f'Model config path: {args.model_config}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print(f'Weights: {w}')
    print(f'Evaluate with random rotation: {args.with_rotation}')
    print('')

    model_params = ModelParams(args.model_config)
    model_params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    # List all test files
    assert os.path.exists(args.dataset_root), f'Cannot access dataset root: {args.dataset_root}'
    # List evaluation sets
    eval_sets = os.listdir(args.dataset_root)
    if args.sensor == "R":
        # Radar-based evaluation sets
        assert model_params.radar_model is not None
        eval_sets = [e for e in eval_sets if e.startswith('test_') and e.endswith('384_128.pickle') and '_R_' in e]
        quantizer = None
    elif args.sensor == "L":
        # LiDAR-based evaluation sets
        assert model_params.lidar_model is not None
        eval_sets = [e for e in eval_sets if e.startswith('test_') and '_L_' in e]
        quantizer = model_params.lidar_quantizer
    else:
        raise NotImplementedError(f"Unknown sensor: {args.sensor}")

    stats = {}
    for eval_set in eval_sets:
        print(f'Evaluation set: {eval_set}')
        evaluator = Evaluator(args.dataset_root, args.dataset, eval_set, device, radius=args.radius,
                              quantizer=quantizer, with_rotation=args.with_rotation)

        stats[eval_set] = evaluator.evaluate(model)
        evaluator.print_results(stats[eval_set])

    recall_d = {}
    for eval_set in eval_sets:
        recall = stats[eval_set]['recall']
        for r in recall:
            if r not in recall_d:
                recall_d[r] = []
            recall_d[r].append(recall[r][0])

    print('Mean recall per all test datasets')
    for r in recall_d:
        print(f"Radius: {r}    Mean recall: {np.mean(recall_d[r]):0.3f}")
