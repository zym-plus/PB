#!/usr/bin/env python3
"""Build and optionally render a multi-model OWOD comparison manifest."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_run(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError("runs must be formatted as METHOD:MANIFEST_OR_EVAL_DIR")
    method, path = value.split(":", 1)
    method = method.strip()
    path = path.strip()
    if not method:
        raise argparse.ArgumentTypeError("method name cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("manifest path cannot be empty")
    return method, Path(path)


def task_from_run_path(path):
    name = Path(path).name
    if name.startswith("t") and name[1:].isdigit():
        task = int(name[1:])
        if task in (1, 2, 3, 4):
            return task
    return None


def runs_from_eval_manifest(method, manifest_path):
    manifest = read_json(manifest_path)
    runs = []
    for item in manifest.get("runs", []):
        task = int(item.get("task") or task_from_run_path(item.get("path", "")) or 0)
        if task not in (1, 2, 3, 4):
            raise ValueError(f"{manifest_path} has invalid task entry: {item}")
        runs.append({
            "method": method,
            "task": task,
            "path": item["path"],
            "gpus": item.get("gpus", manifest.get("gpus", "")),
            "trained_at": item.get("trained_at", manifest.get("trained_at", "")),
        })
    return runs


def runs_from_run_manifest(method, manifest_path):
    manifest = read_json(manifest_path)
    experiment = manifest.get("experiment", {})
    eval_config = experiment.get("eval") or {}
    eval_root = eval_config.get("output_dir")
    if not eval_root:
        result_dir = Path(manifest_path).parent
        # The current pipeline writes run_manifest.json under results/MOWODB/<method>
        # and eval logs under exps/MOWODB/<method>/eval.
        if len(result_dir.parts) >= 3 and result_dir.parts[-3:-1] == ("results", "MOWODB"):
            eval_root = Path("exps") / "MOWODB" / result_dir.parts[-1] / "eval"
        else:
            eval_root = Path("exps") / "MOWODB" / method / "eval"

    gpu = manifest.get("gpu", "")
    finished_at = manifest.get("finished_at", "")
    return [
        {
            "method": method,
            "task": task,
            "path": str(Path(eval_root) / f"t{task}"),
            "gpus": gpu,
            "trained_at": finished_at,
        }
        for task in range(1, 5)
    ]


def runs_from_eval_dir(method, eval_dir):
    return [
        {
            "method": method,
            "task": task,
            "path": str(Path(eval_dir) / f"t{task}"),
        }
        for task in range(1, 5)
    ]


def runs_from_source(method, path):
    if path.is_dir():
        return runs_from_eval_dir(method, path)

    if not path.exists():
        raise FileNotFoundError(f"comparison input does not exist for {method}: {path}")

    if path.name == "run_manifest.json":
        return runs_from_run_manifest(method, path)
    return runs_from_eval_manifest(method, path)


def check_runs(runs):
    missing = []
    methods = {}
    for item in runs:
        methods.setdefault(item["method"], set()).add(item["task"])
        log_path = Path(item["path"]) / "log.txt"
        if not log_path.exists():
            missing.append((item["method"], item["task"], log_path))

    incomplete = {
        method: sorted(set(range(1, 5)) - tasks)
        for method, tasks in methods.items()
        if tasks != set(range(1, 5))
    }
    if missing or incomplete:
        for method, tasks in incomplete.items():
            print(f"method {method} missing task entries: {tasks}", file=sys.stderr)
        for method, task, path in missing:
            print(f"method {method} task {task} missing log: {path}", file=sys.stderr)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Build a multi-model OWOD comparison manifest.")
    parser.add_argument("--title", default="M-OWODB Model Comparison")
    parser.add_argument("--baseline", default="PROB")
    parser.add_argument("--output", default="results/MOWODB/model_comparison_manifest.json")
    parser.add_argument("--output-dir", default="results/MOWODB/comparison")
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        help="METHOD:run_manifest.json, METHOD:eval_manifest.json, or METHOD:eval_dir")
    parser.add_argument("--render", action="store_true", help="render HTML/CSV after writing the manifest")
    parser.add_argument("--skip-checks", action="store_true", help="skip log.txt existence checks")
    args = parser.parse_args()

    runs = []
    sources = []
    for method, path in args.run:
        source_runs = runs_from_source(method, path)
        runs.extend(source_runs)
        sources.append({"method": method, "path": str(path)})

    if not args.skip_checks:
        check_runs(runs)

    manifest = {
        "title": args.title,
        "baseline": args.baseline,
        "source": "comparison of experiment log.txt files",
        "sources": sources,
        "runs": sorted(runs, key=lambda item: (item["method"], item["task"])),
    }
    write_json(args.output, manifest)
    print(f"manifest: {args.output}")

    if args.render:
        subprocess.run([
            sys.executable,
            "scripts/visualize_results.py",
            "--manifest",
            args.output,
            "--output-dir",
            args.output_dir,
        ], check=True)


if __name__ == "__main__":
    main()
