"""
GigaTIME training loop (ported from db_train.py).
Provides train() and validate() functions called by main.py.
"""
import random
from collections import OrderedDict

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr, spearmanr

from gigatime_analyzer.models.utils import AverageMeter


def calculate_correlations(matrix1, matrix2):
    """Calculate Pearson and Spearman correlation coefficients between two matrices."""
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"
    b, c, h, w = matrix1.shape

    pearson_correlations = []
    spearman_correlations = []

    for channel in range(c):
        pearson_corrs = []
        spearman_corrs = []

        for batch in range(b):
            flat_matrix1 = matrix1[batch, channel].flatten()
            flat_matrix2 = matrix2[batch, channel].flatten()

            valid_indices = ~np.isnan(flat_matrix1.detach().cpu().numpy()) & ~np.isnan(flat_matrix2.detach().cpu().numpy())
            flat_matrix1 = flat_matrix1[valid_indices]
            flat_matrix2 = flat_matrix2[valid_indices]

            if len(flat_matrix1) > 0 and len(flat_matrix2) > 0:
                pearson_corr, _ = pearsonr(flat_matrix1.detach().cpu().numpy(), flat_matrix2.detach().cpu().numpy())
                spearman_corr, _ = spearmanr(flat_matrix1.detach().cpu().numpy(), flat_matrix2.detach().cpu().numpy())
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan

            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)

        pearson_correlations.append(np.nanmean(pearson_corrs))
        spearman_correlations.append(np.nanmean(spearman_corrs))

    return pearson_correlations, spearman_correlations


def split_into_boxes(tensor, box_size):
    batch_size, channels, height, width = tensor.shape
    num_boxes_y = height // box_size
    num_boxes_x = width // box_size
    boxes = tensor.unfold(2, box_size, box_size).unfold(3, box_size, box_size)
    boxes = boxes.contiguous().view(batch_size, channels, num_boxes_y, num_boxes_x, box_size, box_size)
    return boxes


def count_ones(boxes):
    return boxes.sum(dim=(4, 5))


def get_box_metrics(pred, mask, box_size):
    pred_boxes = split_into_boxes(pred, box_size)
    mask_boxes = split_into_boxes(mask, box_size)
    pred_counts = count_ones(pred_boxes)
    mask_counts = count_ones(mask_boxes)

    mse = ((pred_counts.float() - mask_counts.float()) ** 2).mean(dim=0)
    mean_mse_per_channel = mse.mean(dim=(1, 2))
    mean_mse = mse.mean().item()

    pearson, spearman = calculate_correlations(pred_counts, mask_counts)

    return mean_mse_per_channel, pearson, spearman


def sample_data_loader(data_loader, config, sample_fraction=0.1, deterministic=False, what_split="train"):
    dataset = data_loader.dataset
    total_size = len(dataset)
    sample_size = int(total_size * sample_fraction)

    if deterministic:
        sample_indices = [i for i in range(sample_size)]
    else:
        sample_indices = random.sample(range(total_size), sample_size)

    subset = Subset(dataset, sample_indices)

    if what_split == "train":
        sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True,
                                   num_workers=config['num_workers'],
                                   prefetch_factor=6,
                                   drop_last=True)
    else:
        sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=False,
                                   num_workers=config['num_workers'],
                                   prefetch_factor=6,
                                   drop_last=False)
    return sample_loader


def denormalize(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor * std + mean


def train(config, train_loader, model, criterion, optimizer):
    """Run one training epoch. Returns OrderedDict with loss and per-class pearson metrics."""
    avg_meters = {'loss': AverageMeter(), 'pearson': AverageMeter()}
    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]

    model.train()

    pbar = tqdm.tqdm(total=len(train_loader))
    for input, target, name in train_loader:
        downsampled_image = F.interpolate(target, scale_factor=1/8, mode='bilinear', align_corners=False)
        target = F.interpolate(downsampled_image, size=(config["input_h"], config["input_h"]), mode='bilinear', align_corners=False)
        target = target.cuda()

        output_image = model(input.cuda()).cuda()
        loss = criterion(output_image, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pearson, _ = get_box_metrics(output_image, target, box_size=8)

        for class_idx, pearson_value in enumerate(pearson):
            pearson_per_class_meters[class_idx].update(pearson_value, input.size(0))

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['pearson'].update(np.nanmean(pearson), input.size(0))

        pbar.set_postfix({'loss': avg_meters['loss'].avg, 'pearson': avg_meters['pearson'].avg})
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('pearson', avg_meters['pearson'].avg)] +
                       [(f'class_{i}', m.avg) for i, m in enumerate(pearson_per_class_meters)])


def validate(config, val_loader, model, criterion, return_samples=False):
    """
    Run validation. Returns OrderedDict with loss and per-class pearson metrics.
    If return_samples=True, also returns one batch of (input, target, output) tensors
    for visualization under keys 'input', 'target', 'output'.
    """
    avg_meters = {'loss': AverageMeter(), 'pearson': AverageMeter()}
    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]

    model.eval()

    sample_input = None
    sample_target = None
    sample_output = None

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader))
        for input, target, name in val_loader:
            downsampled_image = F.interpolate(target, scale_factor=1/8, mode='bilinear', align_corners=False)
            target = F.interpolate(downsampled_image, size=(config["input_h"], config["input_h"]), mode='bilinear', align_corners=False)
            target = target.cuda()

            output_image = model(input.cuda()).cuda()
            loss = criterion(output_image, target)

            _, pearson, _ = get_box_metrics(output_image, target, box_size=8)

            for class_idx, pearson_value in enumerate(pearson):
                pearson_per_class_meters[class_idx].update(pearson_value, input.size(0))

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['pearson'].update(np.nanmean(pearson), input.size(0))

            if return_samples and sample_input is None:
                sample_input = input
                sample_target = target
                sample_output = output_image

            pbar.set_postfix({'loss': avg_meters['loss'].avg, 'pearson': avg_meters['pearson'].avg})
            pbar.update(1)
        pbar.close()

    result = OrderedDict([('loss', avg_meters['loss'].avg), ('pearson', avg_meters['pearson'].avg)] +
                         [(f'class_{i}', m.avg) for i, m in enumerate(pearson_per_class_meters)])
    if return_samples:
        result['input'] = sample_input
        result['target'] = sample_target
        result['output'] = sample_output

    return result
