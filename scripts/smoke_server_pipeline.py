#!/usr/bin/env python3
"""Fast server smoke test for the PROB baseline pipeline."""

import argparse
import os
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Run one tiny PROB pipeline smoke test.")
    parser.add_argument("--dataset", default="TOWOD", choices=["TOWOD", "OWDETR", "VOC2007"])
    parser.add_argument("--train-set", default="owod_t1_train")
    parser.add_argument("--test-set", default="owod_all_task_test")
    parser.add_argument("--prev-classes", default=0, type=int)
    parser.add_argument("--cur-classes", default=20, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", default="dino_resnet50")
    parser.add_argument("--max-scan", default=20, type=int, help="max samples to scan for a non-empty target")
    parser.add_argument("--backward", action="store_true", help="also run one backward pass")
    parser.add_argument("--data-root", default=os.environ.get("OWOD_DATA_ROOT", "/home/zym/data/OWOD"))
    parser.add_argument("--splits-root", default=os.environ.get("OWOD_SPLITS_ROOT", "data/OWOD"))
    return parser.parse_args()


def build_main_args(cli, get_args_parser):
    smoke_argv = [
        "--dataset", cli.dataset,
        "--data_root", cli.data_root,
        "--splits_root", cli.splits_root,
        "--train_set", cli.train_set,
        "--test_set", cli.test_set,
        "--PREV_INTRODUCED_CLS", str(cli.prev_classes),
        "--CUR_INTRODUCED_CLS", str(cli.cur_classes),
        "--model_type", "prob",
        "--backbone", cli.backbone,
        "--device", cli.device,
        "--batch_size", "1",
        "--num_workers", "0",
        "--wandb_project", "",
        "--output_dir", "",
        "--no_aux_loss",
    ]
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    return parser.parse_args(smoke_argv)


def move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append({
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in target.items()
        })
    return moved


def get_one_batch(dataset, max_scan, utils):
    max_scan = min(max_scan, len(dataset))
    for index in range(max_scan):
        image, target = dataset[index]
        if len(target["labels"]) > 0:
            samples, targets = utils.collate_fn([(image, target)])
            return index, samples, targets
    raise RuntimeError(f"no non-empty target found in first {max_scan} samples")


def main():
    cli = parse_args()

    import numpy as np
    import torch

    import util.misc as utils
    from datasets.coco import make_coco_transforms
    from datasets.torchvision_datasets.open_world import OWDetection
    from main_open_world import get_args_parser, validate_owod_paths
    from models import build_model

    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; rerun with --device cpu only for CPU debugging")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args = build_main_args(cli, get_args_parser)
    validate_owod_paths(args)

    print("[smoke] building dataset")
    dataset = OWDetection(
        args,
        args.data_root,
        splits_root=args.splits_root,
        image_set=args.train_set,
        transforms=make_coco_transforms(args.train_set),
        dataset=args.dataset,
    )
    print(f"[smoke] dataset={args.dataset} train_set={args.train_set} size={len(dataset)}")

    index, samples, targets = get_one_batch(dataset, cli.max_scan, utils)
    device = torch.device(cli.device)
    samples = samples.to(device)
    targets = move_targets_to_device(targets, device)
    print(f"[smoke] sample_index={index} boxes={len(targets[0]['labels'])} device={device}")

    print("[smoke] building PROB model")
    model, criterion, _, _ = build_model(args, mode="prob")
    model.to(device)
    criterion.to(device)
    model.train()
    criterion.train()

    start = time.time()
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    loss = sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)
    if cli.backward:
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] loss={loss.item():.6f}")
    print(f"[smoke] trainable_params={n_parameters}")
    print(f"[smoke] elapsed_sec={time.time() - start:.2f}")
    print("[smoke] OK: PROB baseline pipeline can read data, build model, and run one batch")


if __name__ == "__main__":
    main()
