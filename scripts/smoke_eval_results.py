#!/usr/bin/env python3
"""Fast smoke for eval metrics logging and result-table rendering."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test M-OWODB eval logging and visualization.")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--data-root", default="/home/zym/data/OWOD")
    parser.add_argument("--splits-root", default="data/OWOD")
    parser.add_argument("--coco-path", default="/home/zym/data/coco")
    parser.add_argument("--checkpoint-dir", default="exps/MOWODB/PROB")
    parser.add_argument("--work-dir", default="/tmp/pb_mowodb_eval_smoke")
    parser.add_argument("--num-workers", default="0")
    return parser.parse_args()


def read_ids(path, max_images):
    ids = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            image_id = line.strip()
            if image_id:
                ids.append(image_id)
            if len(ids) >= max_images:
                break
    if not ids:
        raise RuntimeError(f"no image ids found in {path}")
    return ids


def write_smoke_split(source_splits_root, work_dir, dataset, split_name, max_images):
    source = Path(source_splits_root) / "ImageSets" / dataset / "owod_all_task_test.txt"
    ids = read_ids(source, max_images)

    smoke_splits_root = Path(work_dir) / "splits"
    target_dir = smoke_splits_root / "ImageSets" / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{split_name}.txt"
    target.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"[smoke-eval] wrote split {target} with {len(ids)} images")
    return smoke_splits_root


def check_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing expected smoke output: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"empty expected smoke output: {path}")
    return path


def main():
    args = parse_args()
    dataset = "TOWOD"
    split_name = f"smoke_eval_t{args.task}"
    prev_classes = (args.task - 1) * 20
    cur_classes = 20
    checkpoint = Path(args.checkpoint_dir) / f"t{args.task}.pth"
    check_file(checkpoint)

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    smoke_splits_root = write_smoke_split(args.splits_root, work_dir, dataset, split_name, args.max_images)
    run_dir = work_dir / "eval" / f"t{args.task}"
    result_dir = work_dir / "results"

    command = [
        sys.executable,
        "-u",
        "main_open_world.py",
        "--output_dir", str(run_dir),
        "--dataset", dataset,
        "--PREV_INTRODUCED_CLS", str(prev_classes),
        "--CUR_INTRODUCED_CLS", str(cur_classes),
        "--train_set", split_name,
        "--test_set", split_name,
        "--epochs", "1",
        "--lr_drop", "1",
        "--model_type", "prob",
        "--obj_loss_coef", "8e-4",
        "--obj_temp", "1.3",
        "--pretrain", str(checkpoint),
        "--eval",
        "--wandb_project", "",
        "--data_root", args.data_root,
        "--splits_root", str(smoke_splits_root),
        "--coco_path", args.coco_path,
        "--num_workers", args.num_workers,
        "--batch_size", "1",
    ]
    print("[smoke-eval] running tiny eval")
    subprocess.run(command, check=True)

    log_path = check_file(run_dir / "log.txt")
    print(f"[smoke-eval] log={log_path}")

    print("[smoke-eval] rendering result table")
    subprocess.run([
        sys.executable,
        "scripts/render_experiment_results.py",
        "--title", "M-OWODB PROB Eval Smoke",
        "--method", "PROB",
        "--baseline", "PROB",
        "--output-dir", str(result_dir),
        "--run", f"{args.task}:{run_dir}",
    ], check=True)

    html_path = check_file(result_dir / "owod_results.html")
    csv_path = check_file(result_dir / "owod_results.csv")
    manifest_path = check_file(result_dir / "eval_manifest.json")
    print(f"[smoke-eval] html={html_path}")
    print(f"[smoke-eval] csv={csv_path}")
    print(f"[smoke-eval] manifest={manifest_path}")
    print("[smoke-eval] OK")


if __name__ == "__main__":
    main()
