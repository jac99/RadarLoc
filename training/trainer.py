# Warsaw University of Technology
# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.

import os
import numpy as np
import torch
import tqdm
import pathlib
import shutil
import wandb
from eval.evaluate import Evaluator
from misc.utils import TrainingParams, get_datetime
from models.loss import make_losses
from models.model_factory import model_factory

from datasets.dataset_utils import make_dataloaders


def print_stats(stats, phase):
    # For triplet loss
    s = '{} - Global loss: {:.6f}    Embedding norm: {:.4f}   Triplets (all/active): {:.1f}/{:.1f}'
    print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'], stats['num_non_zero_triplets']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(params: TrainingParams, debug=False, device='cpu'):
    # wandn_entity_name: Wights & Biases logging service entity name
    # Create model class

    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = 'model_'
    if params.model_params.radar_model is not None:
        model_name += params.model_params.radar_model + '_'
    if params.model_params.lidar_model is not None:
        model_name += params.model_params.lidar_model + '_'

    model_name += s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    shutil.copy(params.params_path, os.path.join(weights_path, 'config.txt'))
    shutil.copy(params.params_path, os.path.join(weights_path, 'model_config.txt'))

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer

    model.to(device)

    # set up dataloaders
    dataloaders = make_dataloaders(params)
    print('Model device: {}'.format(device))

    loss_fn = make_losses(params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)

    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)
    wandb.init(project='RadarLoc', config=params_dict)

    ###########################################################################
    #
    ###########################################################################

    phases = ['train']
    if 'val' in dataloaders:
        phases.append('val')

    # Training statistics
    stats = {e: [] for e in phases}
    stats['eval'] = []

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        for phase in phases:
            if 'train' in phase:
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch
            count_batches = 0

            for batch in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                # Move everything to the device
                batch = {e: batch[e].to(device) for e in batch}

                positives_mask = batch['positives_mask']
                negatives_mask = batch['negatives_mask']
                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()

                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    embeddings = model(batch)
                    assert len(embeddings) == len(positives_mask), f"{len(embeddings) - len(positives_mask)}"
                    assert len(embeddings) == len(negatives_mask), f"{len(embeddings) - len(negatives_mask)}"

                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)
                    batch_stats['loss'] = loss.item()

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                running_stats.append(batch_stats)

            # ******* PHASE END *******
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        metrics = {'train': {}, 'val': {}, 'test': {}}

        metrics['train']['loss'] = stats['train'][-1]['loss']
        if 'num_triplets' in stats['train'][-1]:
            metrics['train']['active_triplets'] = stats['train'][-1]['num_non_zero_triplets']

        if 'val' in phases:
            metrics['val']['loss'] = stats['val'][-1]['loss']
            if 'num_triplets' in stats['val'][-1]:
                metrics['val']['active_triplets'] = stats['val'][-1]['num_non_zero_triplets']

        wandb.log(metrics)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    torch.save(model.state_dict(), final_model_path)

    # Evaluate the final model using all samples
    radius = [5, 10]
    if params.model_params.lidar_model is not None:
        quantizer = params.model_params.lidar_quantizer
    else:
        quantizer = None

    evaluator_test_set = Evaluator(params.dataset_folder, dataset='mulran', eval_set_pickle=params.test_file,
                                   device=device, radius=radius, k=20, quantizer=quantizer,
                                   with_rotation=False)

    global_stats = evaluator_test_set.evaluate(model)

    print('Evaluation results (no rotation):')
    evaluator_test_set.print_results(global_stats)

    evaluator_test_set = Evaluator(params.dataset_folder, dataset='mulran', eval_set_pickle=params.test_file,
                                   device=device, radius=radius, k=20, quantizer=quantizer,
                                   with_rotation=True)

    global_stats_with_rotation = evaluator_test_set.evaluate(model)
    print('Evaluation results (with rotation):')
    evaluator_test_set.print_results(global_stats_with_rotation)

    # Append key experimental metrics to experiment summary file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    _, model_name = os.path.split(model_pathname)
    radar_model_name = params.model_params.radar_model
    lidar_model_name = params.model_params.lidar_model
    prefix = "{}, {}, {}, {}, {}".format(model_name, model_params_name, config_name, radar_model_name, lidar_model_name)
    export_eval_stats("experiment_results.txt", prefix, global_stats, global_stats_with_rotation)


def export_eval_stats(file_name, prefix, stats, stats2):
    # Print results on the final model
    metrics = {}
    metrics['recall_5'] = stats['recall'][5][0]
    metrics['recall_10'] = stats['recall'][10][0]
    metrics['recall_5_R'] = stats2['recall'][5][0]
    metrics['recall_10_R'] = stats2['recall'][10][0]

    with open(file_name, "a") as f:
        s = f"{prefix}, {metrics['recall_5']:0.3f}, {metrics['recall_10']:0.3f}, " \
            f"{metrics['recall_5_R']:0.3f}, {metrics['recall_10_R']:0.3f} \n"
        f.write(s)
    wandb.log(metrics)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
