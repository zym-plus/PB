#!/usr/bin/env python3
"""Extract M-OWODB eval metrics from a tee log and render result tables."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def first_number(text):
    match = NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def number_after_colon(line):
    return first_number(line.split(":", 1)[1]) if ":" in line else None


def dict_value(line, key):
    pattern = re.compile(
        rf"{re.escape(str(key))}\s*:\s*(?:np\.\w+\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    match = pattern.search(line)
    return float(match.group(1)) if match else None


def parse_blocks(log_path):
    blocks = []
    current = None
    current_params = None

    with Path(log_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("number of params:"):
                current_params = number_after_colon(line)
            elif line.startswith("detection mAP50:"):
                if current:
                    blocks.append(current)
                current = {"n_parameters": current_params, "test_metrics": {}}
            elif current is None:
                continue
            elif line.startswith("Wilderness Impact:"):
                current["test_metrics"]["WI"] = dict_value(line, "0.8")
            elif line.startswith("Absolute OSE"):
                current["test_metrics"]["AOSA"] = dict_value(line, "50")
            elif line.startswith("Prev class AP50:"):
                current["test_metrics"]["PK_AP50"] = number_after_colon(line)
            elif line.startswith("Current class AP50:"):
                current["test_metrics"]["CK_AP50"] = number_after_colon(line)
            elif line.startswith("Known AP50:"):
                current["test_metrics"]["K_AP50"] = number_after_colon(line)
            elif line.startswith("Unknown Recall50:"):
                current["test_metrics"]["U_R50"] = number_after_colon(line)

    if current:
        blocks.append(current)

    complete = [
        block for block in blocks
        if {"CK_AP50", "K_AP50", "U_R50"}.issubset(block["test_metrics"])
    ]
    return complete


def write_task_logs(blocks, eval_dir):
    eval_dir = Path(eval_dir)
    if len(blocks) < 4:
        raise SystemExit(f"expected at least 4 completed eval blocks, found {len(blocks)}")

    selected = blocks[-4:]
    for task, block in enumerate(selected, 1):
        run_dir = eval_dir / f"t{task}"
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "epoch": None,
            "n_parameters": block.get("n_parameters"),
            "test_metrics": block["test_metrics"],
        }
        (run_dir / "log.txt").write_text(json.dumps(record) + "\n", encoding="utf-8")
        print(f"wrote {run_dir / 'log.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Create result log.txt files from an eval tee log.")
    parser.add_argument("--log", default="logs/eval_mowodb_prob_gpu2.log")
    parser.add_argument("--eval-dir", default="exps/MOWODB/PROB/eval")
    parser.add_argument("--output-dir", default="results/MOWODB/PROB")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    blocks = parse_blocks(args.log)
    write_task_logs(blocks, args.eval_dir)

    if args.render:
        subprocess.run([
            sys.executable,
            "scripts/render_experiment_results.py",
            "--title", "M-OWODB PROB Evaluation Results",
            "--method", "PROB",
            "--baseline", "PROB",
            "--output-dir", args.output_dir,
            "--run", f"1:{args.eval_dir}/t1",
            "--run", f"2:{args.eval_dir}/t2",
            "--run", f"3:{args.eval_dir}/t3",
            "--run", f"4:{args.eval_dir}/t4",
        ], check=True)


if __name__ == "__main__":
    main()
