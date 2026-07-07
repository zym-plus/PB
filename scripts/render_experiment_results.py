#!/usr/bin/env python3
"""Render final OWOD result tables for a set of task run directories."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_run(value):
    try:
        task, path = value.split(":", 1)
        task = int(task)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("runs must be formatted as TASK:PATH") from exc
    if task not in (1, 2, 3, 4):
        raise argparse.ArgumentTypeError("TASK must be one of 1, 2, 3, 4")
    return task, path


def check_run_logs(runs):
    missing = []
    empty = []
    invalid = []
    for task, run_path in runs:
        log_path = Path(run_path) / "log.txt"
        if not log_path.exists():
            missing.append((task, log_path))
            continue

        latest = None
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    metrics = record.get("test_metrics")
                    if isinstance(metrics, dict) and metrics:
                        latest = line_no
        except json.JSONDecodeError as exc:
            invalid.append((task, log_path, exc.lineno, exc.msg))
            continue

        if latest is None:
            empty.append((task, log_path))

    if not missing and not empty and not invalid:
        return

    for task, path in missing:
        print(f"missing task {task} log: {path}", file=sys.stderr)
    for task, path in empty:
        print(f"task {task} log has no non-empty test_metrics: {path}", file=sys.stderr)
    for task, path, line_no, message in invalid:
        print(f"invalid JSON in task {task} log {path}:{line_no}: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate OWOD final result HTML/CSV from task log.txt files.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", default="PROB")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-name", default="eval_manifest.json")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"))
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        help="task run directory in TASK:PATH format, for example 1:exps/MOWODB/PROB/t1")
    args = parser.parse_args()

    check_run_logs(args.run)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    manifest = {
        "title": args.title,
        "baseline": args.baseline,
        "gpus": args.gpus,
        "source": "experiment log.txt files",
        "runs": [
            {"method": args.method, "task": task, "path": path}
            for task, path in args.run
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    visualize_script = Path(__file__).resolve().with_name("visualize_results.py")
    subprocess.run([
        sys.executable,
        str(visualize_script),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ], check=True)


if __name__ == "__main__":
    main()
