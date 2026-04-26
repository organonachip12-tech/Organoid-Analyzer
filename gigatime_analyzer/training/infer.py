"""
GigaTIME inference loop (ported from db_test.py).
Provides validate() and convert_to_csv() functions called by main.py.
"""
import json
import os
import random
import sys
import types
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr, spearmanr

from gigatime_analyzer.models.utils import AverageMeter
from gigatime_analyzer.data.dataset import common_channel_list


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

            valid_indices = ~np.isnan(flat_matrix1.cpu().numpy()) & ~np.isnan(flat_matrix2.cpu().numpy())
            flat_matrix1 = flat_matrix1[valid_indices]
            flat_matrix2 = flat_matrix2[valid_indices]

            if len(flat_matrix1) > 0 and len(flat_matrix2) > 0:
                pearson_corr, _ = pearsonr(flat_matrix1.cpu().numpy(), flat_matrix2.cpu().numpy())
                spearman_corr, _ = spearmanr(flat_matrix1.cpu().numpy(), flat_matrix2.cpu().numpy())
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


def convert_to_csv(pearson_dict, out_path, channel_list):
    """
    Write per-channel Pearson results to CSV at the explicit out_path.

    Parameters:
    pearson_dict: list of AverageMeter objects (one per channel)
    out_path: full path to write the CSV file (including filename)
    channel_list: list of channel names
    """
    pear_per_class_avg = [pear.avg for pear in pearson_dict]
    pear_per_class_std = [pear.std for pear in pearson_dict]
    combined = list(zip(pear_per_class_avg, channel_list, pear_per_class_std))
    sorted_combined = sorted(combined, key=lambda x: x, reverse=True)
    pearson_per_class_avg, channel_names, pearson_per_class_std = zip(*sorted_combined)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = {
        'Channel': channel_names,
        'Pearson_Average Results': pearson_per_class_avg,
        'Pearson_Standard Deviation Results': pearson_per_class_std,
    }
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"Results written to: {out_path}")


def small_tile_preds(big_image, output_image, model, window_size=256, device=None):
    if device is None:
        device = next(model.parameters()).device
    for i in range(0, big_image.shape[2], window_size):
        for j in range(0, big_image.shape[3], window_size):
            window = big_image[:, :, i:i + window_size, j:j + window_size].to(device)
            output = model(window)
            output_image[:, :, i:i + window_size, j:j + window_size] = output
    return output_image


def validate(config, val_loader, model, criterion, output_dir, set_name="silver"):
    """
    Run inference on the test/val loader.
    Writes per-channel Pearson CSV to output_dir/metrics/<set>_Results_per_channel_<date>_test_results.csv.
    Returns OrderedDict with loss and per-channel pearson metrics.
    """
    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    avg_meters = {'loss': AverageMeter(), 'pearson': AverageMeter()}
    pearson_per_class_meters = [AverageMeter() for _ in range(config['num_classes'])]

    window_size = config['window_size']
    model.eval()
    count = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader))

        for data in val_loader:
            count += 1
            input, target_g, name = data

            downsampled_image = F.interpolate(target_g, scale_factor=1/8, mode='bilinear', align_corners=False)
            target = F.interpolate(downsampled_image, size=(512, 512), mode='bilinear', align_corners=False)
            target = target.to(device)

            output_image = torch.zeros_like(target).to(device)
            output_image = small_tile_preds(input, output_image, model, window_size, device=device)

            output_image = output_image > 0.5
            output_image = output_image.float()
            loss = criterion(output_image, target)

            avg_meters['loss'].update(loss.item(), target.size(0))

            _, pearson, spearman = get_box_metrics(output_image, target, box_size=8)

            for class_idx, value in enumerate(pearson):
                if not np.isnan(value):
                    pearson_per_class_meters[class_idx].update(value, target.size(0))

            pbar.set_postfix({'loss': avg_meters['loss'].avg})
            pbar.update(1)

        pbar.close()

    # Write CSV to metrics/ subdirectory
    today_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"{set_name}_Results_per_channel_{today_date}_test_results.csv"
    out_path = os.path.join(output_dir, "metrics", csv_filename)
    convert_to_csv(pearson_per_class_meters, out_path, common_channel_list)

    return OrderedDict(
        [('loss', avg_meters['loss'].avg), ('pearson', avg_meters['pearson'].avg)] +
        [(f'{ch}_pearson', pearson_per_class_meters[i].avg) for i, ch in enumerate(common_channel_list)]
    )


def print_logs(log, exclude_keys=[]):
    for key, value in log.items():
        if key not in exclude_keys:
            try:
                print(f"{key}: {value:.4f}")
            except (TypeError, ValueError):
                print(f"{key}: {value}")


def load_pretrained_model(model, output_dir, hf_token=None):
    """
    Load pretrained weights into model. Checks output_dir/model.pth first,
    then falls back to HuggingFace Hub download.
    """
    ckpt_path = os.path.join(output_dir, "model.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
    else:
        print("Downloading pretrained GigaTIME model from HuggingFace Hub...")
        from huggingface_hub import snapshot_download

        token = hf_token or os.environ.get("HF_TOKEN")
        local_dir = snapshot_download(repo_id="prov-gigatime/GigaTIME", token=token)
        ckpt_path = os.path.join(local_dir, "model.pth")

    # Compatibility shim for older PyTorch serialization
    if "torch.utils.serialization" not in sys.modules:
        sys.modules["torch.utils.serialization"] = types.ModuleType("torch.utils.serialization")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    return model
