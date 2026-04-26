"""
GigaTIME Analyzer CLI entry point.
Usage:
  python gigatime_analyzer/training/main.py --mode preprocess --svs_dir ./data/gigatime/svs
  python gigatime_analyzer/training/main.py --mode train --tiling_dir ./data/gigatime/preprocessed_tiles --metadata ./data/gigatime/preprocessed_tiles/preprocessed_metadata.csv
  python gigatime_analyzer/training/main.py --mode infer  --tiling_dir ./data/gigatime/preprocessed_tiles --metadata ./data/gigatime/preprocessed_tiles/preprocessed_metadata.csv --name my_run
"""
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="GigaTIME Analyzer: H&E to mIF prediction (U-Net++)"
    )

    # Mode
    parser.add_argument(
        "--mode", required=True, choices=["preprocess", "train", "infer"],
        help="Operation mode: preprocess SVS slides, train model, or run inference"
    )

    # Data paths
    parser.add_argument("--tiling_dir", default=None,
                        help="Path to preprocessed tiling directory (input to train/infer)")
    parser.add_argument("--metadata", default=None,
                        help="Path to metadata CSV (preprocessed_metadata.csv or custom)")
    parser.add_argument("--svs_dir", default=None,
                        help="Directory containing SVS slides (required for --mode preprocess)")

    # Model
    parser.add_argument("--arch", default="gigatime", help="Model architecture (default: gigatime)")
    parser.add_argument("--num_classes", type=int, default=23, help="Number of output channels")
    parser.add_argument("--input_channels", type=int, default=3, help="Input channels")
    parser.add_argument("--input_h", type=int, default=512, help="Crop/resize height")
    parser.add_argument("--input_w", type=int, default=512, help="Crop/resize width")
    parser.add_argument("--name", default="gigatime_model",
                        help="Experiment/model name for checkpoint naming")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--scheduler", default="CosineAnnealingLR",
                        choices=["CosineAnnealingLR", "ReduceLROnPlateau", "MultiStepLR", "ConstantLR"])
    parser.add_argument("--loss", default="BCEDiceLoss",
                        choices=["BCEDiceLoss", "LovaszHingeLoss", "BCEWithLogitsLoss", "MSELoss"])
    parser.add_argument("--window_size", type=int, default=256,
                        help="Sub-window size for sliding-window inference")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--early_stopping", type=int, default=-1,
                        help="Early stopping patience (-1 to disable)")
    parser.add_argument("--sampling_prob", type=float, default=1.0,
                        help="Fraction of training data per epoch")
    parser.add_argument("--val_sampling_prob", type=float, default=1.0,
                        help="Fraction of validation data per epoch")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers (use 0 on Windows if errors occur)")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0],
                        help="GPU IDs for DataParallel")

    # Inference specific
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for model download (or set HF_TOKEN env var)")
    parser.add_argument("--set", default="silver", choices=["silver", "gold", "all"],
                        help="Quality filter set for inference (default: silver)")

    # Output
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results and model checkpoints")

    return parser.parse_args()


def _resolve_output_dir(args):
    from gigatime_analyzer.config import RESULTS_DIR
    return args.output_dir or os.path.join(RESULTS_DIR, args.name)


def _run_preprocess(args, output_dir):
    from pathlib import Path
    from gigatime_analyzer.preprocessing import process_slides, write_metadata_csv
    from gigatime_analyzer.config import SVS_DIR, TILES_DIR

    svs_dir = Path(args.svs_dir or SVS_DIR)
    tiles_dir = Path(output_dir)

    print(f"Preprocessing SVS slides from: {svs_dir}")
    print(f"Output tiles to: {tiles_dir}")

    results = process_slides(input_dir=svs_dir, output_dir=tiles_dir)
    total_tiles = sum(r[1] for r in results)
    print(f"Preprocessing complete: {len(results)} slides, {total_tiles} tiles written.")
    if results:
        csv_path = tiles_dir / "preprocessed_metadata.csv"
        print(f"Metadata CSV: {csv_path}")


def _run_train(args, output_dir):
    import json
    from pathlib import Path
    import yaml
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.backends.cudnn as cudnn
    from torch.optim import lr_scheduler
    from collections import OrderedDict
    import pandas as pd
    import torchvision
    from torchvision.utils import save_image

    import albumentations as geometric
    from albumentations.augmentations import transforms as alb_transforms
    from albumentations.core.composition import Compose, OneOf

    from gigatime_analyzer.models.archs import gigatime
    from gigatime_analyzer.models.losses import BCEDiceLoss
    from gigatime_analyzer.models.utils import str2bool
    from gigatime_analyzer.data.dataset import HECOMETDataset_roi, generate_tile_pair_df, common_channel_list
    from gigatime_analyzer.training.train import train, validate, sample_data_loader, denormalize

    config = vars(args)
    config['output_dir'] = output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print(f"=> using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    print("-" * 20)
    for key, val in config.items():
        print(f"{key}: {val}")
    print("-" * 20)

    # Loss
    if config['loss'] == 'MSELoss':
        criterion = nn.MSELoss().to(device)
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif config['loss'] == 'BCEDiceLoss':
        criterion = BCEDiceLoss().to(device)
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")

    if device.type == 'cuda':
        cudnn.benchmark = False

    print(f"=> creating model {config['arch']}")
    model = gigatime(num_classes=config['num_classes'], input_channels=config['input_channels']).to(device)

    if device.type == 'cuda' and len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
        print(f"Using multiple GPUs: {config['gpu_ids']}")

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])

    if config['scheduler'] == 'CosineAnnealingLR':
        sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        sched = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        sched = lr_scheduler.MultiStepLR(optimizer, milestones=[int(config['epochs'] * 0.5), int(config['epochs'] * 0.75)])
    else:
        sched = None

    train_transform = Compose([
        geometric.RandomRotate90(),
        geometric.Flip(),
        OneOf([
            alb_transforms.HueSaturationValue(),
            alb_transforms.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2),
            alb_transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        ], p=1),
        geometric.Resize(config['input_h'], config['input_w']),
        alb_transforms.Normalize(),
    ], is_check_shapes=False)

    val_transform = Compose([
        geometric.Resize(config['input_h'], config['input_w']),
        alb_transforms.Normalize(),
    ], is_check_shapes=False)

    metadata = pd.read_csv(config["metadata"])
    tiling_dir = Path(config["tiling_dir"])
    tile_pair_df = generate_tile_pair_df(metadata=metadata, tiling_dir=tiling_dir)
    tile_pair_df_filtered = tile_pair_df[tile_pair_df.apply(
        lambda x: (
            (x["img_comet_black_ratio"] < 0.3) &
            (x["img_comet_variance"] > 200) &
            (x["img_he_black_ratio"] < 0.3) &
            (x["img_he_variance"] > 200)
        ), axis=1
    )]

    dir_names = tile_pair_df_filtered["dir_name"].unique()
    segment_metric_dict = {}
    for dir_name in dir_names:
        with open(os.path.join(dir_name, "segment_metric.json"), "r") as f:
            segment_metric_dict[dir_name] = json.load(f)

    new_columns = {col: [] for col in next(iter(segment_metric_dict[dir_names[0]].values())).keys()}
    for _, row in tile_pair_df_filtered.iterrows():
        metrics = segment_metric_dict[row["dir_name"]][row["pair_name"]]
        for key, value in metrics.items():
            new_columns[key].append(value)
    for key, values in new_columns.items():
        tile_pair_df_filtered[key] = values

    tile_pair_df_filtered = tile_pair_df_filtered[tile_pair_df_filtered["dice"] > 0.2]

    train_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered,
        transform=train_transform,
        dir_path=config["tiling_dir"],
        window_size=config["window_size"],
        split="train",
        mask_noncell=True,
        cell_mask_label=True,
    )
    val_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered,
        transform=val_transform,
        dir_path=config["tiling_dir"],
        window_size=config["window_size"],
        split="valid",
        standard="silver",
        mask_noncell=True,
        cell_mask_label=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], prefetch_factor=6 if config['num_workers'] > 0 else None,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], prefetch_factor=6 if config['num_workers'] > 0 else None,
        drop_last=False)

    val_loader = sample_data_loader(val_loader, config, config['val_sampling_prob'], deterministic=True, what_split="valid")

    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('pearson', []), ('val_loss', []), ('val_pearson', [])])

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    best_pearson = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print(f"Epoch [{epoch}/{config['epochs']}]")

        train_log = train(config, train_loader, model, criterion, optimizer)
        return_samples = (epoch % 10 == 0)
        val_log = validate(config, val_loader, model, criterion, return_samples=return_samples)

        if return_samples and val_log.get('input') is not None:
            viz_input = denormalize(val_log['input'].to(device), mean, std)
            save_image(torchvision.utils.make_grid(viz_input, nrow=1),
                       os.path.join(output_dir, 'HE_image.png'))
            save_image(torchvision.utils.make_grid(val_log['target'][:, 0, :, :].unsqueeze(1), nrow=1),
                       os.path.join(output_dir, 'target.png'))
            save_image(torchvision.utils.make_grid(val_log['output'][:, 0, :, :].unsqueeze(1), nrow=1),
                       os.path.join(output_dir, 'output.png'))

        if sched is not None:
            if config['scheduler'] == 'CosineAnnealingLR':
                sched.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                sched.step(val_log['loss'])

        trigger += 1
        if val_log['pearson'] > best_pearson:
            # Save raw model weights (unwrap DataParallel if needed)
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, os.path.join(output_dir, 'model.pth'))
            best_pearson = val_log['pearson']
            print("=> saved best model")

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"Training complete. Best val pearson: {best_pearson:.4f}")


def _run_infer(args, output_dir):
    import json
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
    import pandas as pd

    import albumentations as geometric
    from albumentations.augmentations import transforms as alb_transforms
    from albumentations.core.composition import Compose

    from gigatime_analyzer.models.archs import gigatime
    from gigatime_analyzer.models.losses import BCEDiceLoss
    from gigatime_analyzer.data.dataset import HECOMETDataset_roi, generate_tile_pair_df, common_channel_list
    from gigatime_analyzer.training.infer import validate, sample_data_loader, load_pretrained_model

    config = vars(args)
    config['output_dir'] = output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print(f"=> using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    print("-" * 20)
    for key, val in config.items():
        print(f"{key}: {val}")
    print("-" * 20)

    # Loss
    if config['loss'] == 'MSELoss':
        criterion = nn.MSELoss().to(device)
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = BCEDiceLoss().to(device)

    if device.type == 'cuda':
        cudnn.benchmark = False

    print(f"=> creating model {config['arch']}")
    model = gigatime(num_classes=config['num_classes'], input_channels=config['input_channels']).to(device)
    model = load_pretrained_model(model, output_dir, hf_token=config.get('hf_token'))
    if device.type == 'cuda' and len(config.get("gpu_ids", [0])) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])

    val_transform = Compose([
        geometric.Resize(config['input_h'], config['input_w']),
        alb_transforms.Normalize(),
    ])

    metadata = pd.read_csv(config["metadata"])
    tiling_dir = Path(config["tiling_dir"])
    tile_pair_df = generate_tile_pair_df(metadata=metadata, tiling_dir=tiling_dir)

    tile_pair_df_filtered = tile_pair_df[tile_pair_df.apply(
        lambda x: (
            (x["img_comet_black_ratio"] < 0.3) &
            (x["img_comet_variance"] > 200) &
            (x["img_he_black_ratio"] < 0.3) &
            (x["img_he_variance"] > 200)
        ), axis=1
    )]

    dir_names = tile_pair_df_filtered["dir_name"].unique()
    segment_metric_dict = {}
    for dir_name in dir_names:
        with open(os.path.join(dir_name, "segment_metric.json"), "r") as f:
            segment_metric_dict[dir_name] = json.load(f)

    new_columns = {col: [] for col in next(iter(segment_metric_dict[dir_names[0]].values())).keys()}
    for _, row in tile_pair_df_filtered.iterrows():
        metrics = segment_metric_dict[row["dir_name"]][row["pair_name"]]
        for key, value in metrics.items():
            new_columns[key].append(value)
    for key, values in new_columns.items():
        tile_pair_df_filtered[key] = values

    tile_pair_df_filtered = tile_pair_df_filtered[tile_pair_df_filtered["dice"] > 0.2]

    test_dataset = HECOMETDataset_roi(
        all_tile_pair=tile_pair_df,
        tile_pair_df=tile_pair_df_filtered,
        transform=val_transform,
        dir_path=config["tiling_dir"],
        window_size=config["window_size"],
        split="test",
        standard=config["set"],
        mask_noncell=True,
        cell_mask_label=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], prefetch_factor=4 if config['num_workers'] > 0 else None,
        drop_last=False)

    test_loader = sample_data_loader(test_loader, config, config['val_sampling_prob'],
                                     deterministic=True, what_split="valid")

    test_log = validate(config, test_loader, model, criterion,
                        output_dir=output_dir, set_name=config['set'])

    from gigatime_analyzer.training.infer import print_logs
    print_logs(test_log)
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "preprocess":
        _run_preprocess(args, output_dir)
    elif args.mode == "train":
        _run_train(args, output_dir)
    elif args.mode == "infer":
        _run_infer(args, output_dir)


if __name__ == "__main__":
    main()
