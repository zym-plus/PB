#!/usr/bin/env python3
"""Resumable M-OWODB stage pipeline runner.

The runner is deliberately config-driven: model-specific arguments live in the
JSON config, while this script owns stage ordering, checks, status files,
manifest writing, and result rendering.
"""

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def now_iso():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_capture(command):
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return ""


def git_info():
    sha = run_capture(["git", "rev-parse", "HEAD"]) or "unknown"
    branch = run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status = run_capture(["git", "status", "--short"])
    return {
        "sha": sha,
        "branch": branch,
        "status": "clean" if not status else "dirty",
        "status_short": status,
    }


def rel_or_abs(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def line_count(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def file_info(path, with_sha256=False):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    info = {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat(),
    }
    if with_sha256:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
    return info


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def dataset_split_path(config, split_name):
    dataset = config["experiment"]["dataset"]
    splits_root = Path(config["env"]["splits_root"])
    return splits_root / "ImageSets" / dataset / f"{split_name}.txt"


def exemplar_path(config, exemplar_file):
    dataset = config["experiment"]["dataset"]
    splits_root = Path(config["env"]["splits_root"])
    return splits_root / "ImageSets" / dataset / exemplar_file


def require_file(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    return path


def require_dir(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def status_path(stage):
    return Path(stage["output_dir"]) / "stage_status.json"


def stage_log_path(stage):
    return Path(stage["output_dir"]) / "log.txt"


def is_stage_complete(stage, config):
    status_file = status_path(stage)
    checkpoint = Path(stage["expected_checkpoint"])
    log_file = stage_log_path(stage)
    if not (status_file.exists() and checkpoint.exists() and log_file.exists()):
        return False
    try:
        status = read_json(status_file)
    except Exception:
        return False
    if status.get("status") != "finished":
        return False

    exemplar = stage.get("exemplar") or {}
    current_file = exemplar.get("current_file")
    if current_file:
        path = exemplar_path(config, f"{exemplar.get('dir', '')}/{current_file}")
        if not path.exists() or line_count(path) <= 0:
            return False
    return True


def build_stage_command(stage, config):
    common = config["common_args"]
    command = [
        "bash", "./tools/run_dist_launch.sh", "1", sys.executable, "-u", "main_open_world.py",
        "--output_dir", stage["output_dir"],
        "--dataset", common["dataset"],
        "--PREV_INTRODUCED_CLS", str(stage["previous_classes"]),
        "--CUR_INTRODUCED_CLS", str(stage["current_classes"]),
        "--train_set", stage["train_set"],
        "--test_set", common["test_set"],
        "--epochs", str(stage["epochs"]),
        "--model_type", common["model_type"],
        "--obj_loss_coef", str(common["obj_loss_coef"]),
        "--obj_temp", str(common["obj_temp"]),
        "--wandb_name", stage.get("wandb_name", ""),
        "--wandb_project", str(common.get("wandb_project", "")),
        "--data_root", config["env"]["data_root"],
        "--splits_root", config["env"]["splits_root"],
        "--coco_path", config["env"]["coco_path"],
    ]
    if stage.get("lr_drop") is not None:
        command.extend(["--lr_drop", str(stage["lr_drop"])])
    if stage.get("pretrain"):
        command.extend(["--pretrain", stage["pretrain"]])
    if stage.get("lr"):
        command.extend(["--lr", str(stage["lr"])])
    if stage.get("freeze_prob_model"):
        command.append("--freeze_prob_model")
    if stage.get("num_inst_per_class") is not None:
        command.extend(["--num_inst_per_class", str(stage["num_inst_per_class"])])

    exemplar = stage.get("exemplar") or {}
    if exemplar.get("enabled"):
        command.append("--exemplar_replay_selection")
        command.extend(["--exemplar_replay_max_length", str(exemplar["max_length"])])
        command.extend(["--exemplar_replay_dir", exemplar["dir"]])
        if exemplar.get("previous_file"):
            command.extend(["--exemplar_replay_prev_file", exemplar["previous_file"]])
        if exemplar.get("current_file"):
            command.extend(["--exemplar_replay_cur_file", exemplar["current_file"]])
        if exemplar.get("random"):
            command.append("--exemplar_replay_random")

    for item in stage.get("extra_args", []):
        command.append(str(item))
    return command


def build_eval_command(item, config):
    common = config["common_args"]
    eval_config = config["eval"]
    out_dir = Path(eval_config["output_dir"]) / item["name"]
    return [
        "bash", "./tools/run_dist_launch.sh", "1", sys.executable, "-u", "main_open_world.py",
        "--output_dir", str(out_dir),
        "--dataset", common["dataset"],
        "--PREV_INTRODUCED_CLS", str(item["previous_classes"]),
        "--CUR_INTRODUCED_CLS", str(item["current_classes"]),
        "--train_set", "owod_t1_train",
        "--test_set", common["test_set"],
        "--epochs", str(eval_config.get("epochs", 191)),
        "--lr_drop", str(eval_config.get("lr_drop", 35)),
        "--model_type", common["model_type"],
        "--obj_loss_coef", str(common["obj_loss_coef"]),
        "--obj_temp", str(common["obj_temp"]),
        "--pretrain", item["pretrain"],
        "--eval",
        "--wandb_project", "",
        "--data_root", config["env"]["data_root"],
        "--splits_root", config["env"]["splits_root"],
        "--coco_path", config["env"]["coco_path"],
    ]


def check_common_inputs(config, strict=True):
    if not strict:
        return
    data_root = Path(config["env"]["data_root"])
    splits_root = Path(config["env"]["splits_root"])
    require_dir(splits_root / "ImageSets" / config["experiment"]["dataset"], "OWOD split directory")
    require_dir(data_root, "OWOD data root")
    require_dir(data_root / "Annotations", "OWOD annotations directory")
    require_dir(data_root / "JPEGImages", "OWOD JPEGImages directory")
    require_dir(Path(config["env"]["coco_path"]), "COCO data root")


def check_split(config, split_name):
    path = dataset_split_path(config, split_name)
    require_file(path, f"split {split_name}")
    count = line_count(path)
    if count <= 0:
        raise RuntimeError(f"split {split_name} is empty: {path}")
    return count


def check_exemplar_file(config, path_text, description):
    path = exemplar_path(config, path_text)
    require_file(path, description)
    count = line_count(path)
    if count <= 0:
        raise RuntimeError(f"{description} is empty: {path}")
    return path, count


def check_stage_inputs(stage, config, strict=True):
    checks = {}
    train_split = dataset_split_path(config, stage["train_set"])
    if train_split.exists():
        checks["train_count"] = check_split(config, stage["train_set"])
    elif strict:
        require_file(train_split, f"split {stage['train_set']}")
    else:
        checks["train_count"] = "generated"
    test_split_name = config["common_args"]["test_set"]
    test_split = dataset_split_path(config, test_split_name)
    if test_split.exists():
        checks["test_count"] = check_split(config, test_split_name)
    elif strict:
        require_file(test_split, f"split {test_split_name}")
    else:
        checks["test_count"] = "missing_in_dry_run"

    if stage.get("pretrain"):
        if Path(stage["pretrain"]).exists():
            checks["pretrain"] = file_info(stage["pretrain"])
        elif strict:
            require_file(stage["pretrain"], f"pretrain for stage {stage['name']}")
        else:
            checks["pretrain"] = {"path": stage["pretrain"], "exists": False, "produced_by_pipeline": True}

    required_exemplar = stage.get("requires_exemplar_file")
    if required_exemplar:
        path = exemplar_path(config, required_exemplar)
        if path.exists():
            count = line_count(path)
            checks["required_exemplar"] = {"path": str(path), "lines": count}
        elif strict:
            require_file(path, f"required exemplar for stage {stage['name']}")
        else:
            checks["required_exemplar"] = {"path": str(path), "exists": False, "produced_by_pipeline": True}

    exemplar = stage.get("exemplar") or {}
    previous_file = exemplar.get("previous_file")
    if previous_file:
        path_text = f"{exemplar.get('dir', '')}/{previous_file}"
        path = exemplar_path(config, path_text)
        if path.exists():
            count = line_count(path)
            checks["previous_exemplar"] = {"path": str(path), "lines": count}
        elif strict:
            require_file(path, f"previous exemplar for stage {stage['name']}")
        else:
            checks["previous_exemplar"] = {"path": str(path), "exists": False, "produced_by_pipeline": True}
    return checks


def check_stage_outputs(stage, config):
    checkpoint = require_file(stage["expected_checkpoint"], f"expected checkpoint for stage {stage['name']}")
    log_file = require_file(stage_log_path(stage), f"log for stage {stage['name']}")
    result = {
        "checkpoint": file_info(checkpoint),
        "log": file_info(log_file),
    }

    exemplar = stage.get("exemplar") or {}
    current_file = exemplar.get("current_file")
    if current_file:
        path = exemplar_path(config, f"{exemplar.get('dir', '')}/{current_file}")
        require_file(path, f"generated exemplar for stage {stage['name']}")
        count = line_count(path)
        if count <= 0:
            raise RuntimeError(f"generated exemplar is empty for stage {stage['name']}: {path}")
        max_length = int(exemplar["max_length"])
        if count > max_length:
            raise RuntimeError(
                f"generated exemplar exceeds max length for stage {stage['name']}: {count} > {max_length}"
            )
        result["generated_exemplar"] = {"path": str(path), "lines": count, "max_length": max_length}
    return result


def run_with_tee(command, log_path, env, dry_run=False):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(shell_join(command))
    if dry_run:
        return 0

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[pipeline] command: {shell_join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def write_stage_status(stage, status, command, started_at, finished_at=None, error=None, checks=None, outputs=None):
    data = {
        "stage": stage["name"],
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": None,
        "command": shell_join(command),
        "expected_checkpoint": stage.get("expected_checkpoint"),
        "checks": checks or {},
        "outputs": outputs or {},
        "git": git_info(),
    }
    if finished_at and started_at:
        start = dt.datetime.fromisoformat(started_at)
        end = dt.datetime.fromisoformat(finished_at)
        data["duration_sec"] = int((end - start).total_seconds())
    if error:
        data["error"] = error
    write_json(status_path(stage), data)
    return data


def select_stages(stages, stage=None, from_stage=None):
    names = [item["name"] for item in stages]
    if stage and from_stage:
        raise ValueError("use only one of --stage or --from-stage")
    if stage:
        if stage not in names:
            raise ValueError(f"unknown stage: {stage}")
        return [item for item in stages if item["name"] == stage]
    if from_stage:
        if from_stage not in names:
            raise ValueError(f"unknown from-stage: {from_stage}")
        index = names.index(from_stage)
        return stages[index:]
    return stages


def split_counts(config):
    names = sorted({
        config["common_args"]["test_set"],
        *(stage["train_set"] for stage in config["stages"]),
    })
    counts = {}
    for name in names:
        path = dataset_split_path(config, name)
        if path.exists():
            counts[name] = line_count(path)
    return counts


def build_env(args):
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["GPUS_PER_NODE"] = "1"
    env["MASTER_PORT"] = str(args.master_port)
    return env


def gpu_label(args):
    return f"GPU {args.gpu}" if args.gpu else os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")


def run_stage(stage, config, args, env):
    command = build_stage_command(stage, config)
    stage_name = stage["name"]
    print(f"[stage] {stage_name}")

    if args.skip_completed and not args.force and is_stage_complete(stage, config):
        print(f"[skip] {stage_name} already completed")
        return read_json(status_path(stage))

    checks = check_stage_inputs(stage, config, strict=not args.dry_run)
    print(f"[check] {stage_name} train_count={checks['train_count']} test_count={checks['test_count']}")
    if args.dry_run:
        print(shell_join(command))
        return {
            "stage": stage_name,
            "status": "dry_run",
            "command": shell_join(command),
            "checks": checks,
        }

    started_at = now_iso()
    write_stage_status(stage, "running", command, started_at, checks=checks)
    code = run_with_tee(command, Path("logs") / f"pipeline_{stage_name}.log", env)
    finished_at = now_iso()
    if code != 0:
        status = write_stage_status(
            stage, "failed", command, started_at, finished_at, error=f"exit code {code}", checks=checks
        )
        raise subprocess.CalledProcessError(code, command)

    outputs = check_stage_outputs(stage, config)
    return write_stage_status(stage, "finished", command, started_at, finished_at, checks=checks, outputs=outputs)


def eval_status_path(item, config):
    return Path(config["eval"]["output_dir"]) / item["name"] / "stage_status.json"


def eval_log_path(item, config):
    return Path(config["eval"]["output_dir"]) / item["name"] / "log.txt"


def is_eval_complete(item, config):
    status_file = eval_status_path(item, config)
    log_file = eval_log_path(item, config)
    if not (status_file.exists() and log_file.exists()):
        return False
    try:
        return read_json(status_file).get("status") == "finished"
    except Exception:
        return False


def write_eval_status(item, config, status, command, started_at, finished_at=None, error=None):
    out_dir = Path(config["eval"]["output_dir"]) / item["name"]
    data = {
        "stage": f"eval_{item['name']}",
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": None,
        "command": shell_join(command),
        "checkpoint": item["pretrain"],
        "git": git_info(),
    }
    if finished_at and started_at:
        data["duration_sec"] = int((dt.datetime.fromisoformat(finished_at) - dt.datetime.fromisoformat(started_at)).total_seconds())
    if error:
        data["error"] = error
    write_json(out_dir / "stage_status.json", data)
    return data


def run_eval(config, args, env):
    statuses = []
    for item in config["eval"]["checkpoints"]:
        name = item["name"]
        print(f"[eval] {name}")
        if args.skip_completed and not args.force and is_eval_complete(item, config):
            print(f"[skip] eval {name} already completed")
            statuses.append(read_json(eval_status_path(item, config)))
            continue

        command = build_eval_command(item, config)
        if args.dry_run:
            print(shell_join(command))
            statuses.append({"stage": f"eval_{name}", "status": "dry_run", "command": shell_join(command)})
            continue

        require_file(item["pretrain"], f"eval checkpoint for {name}")

        started_at = now_iso()
        write_eval_status(item, config, "running", command, started_at)
        code = run_with_tee(command, Path("logs") / f"pipeline_eval_{name}.log", env)
        finished_at = now_iso()
        if code != 0:
            write_eval_status(item, config, "failed", command, started_at, finished_at, error=f"exit code {code}")
            raise subprocess.CalledProcessError(code, command)
        require_file(eval_log_path(item, config), f"eval log for {name}")
        statuses.append(write_eval_status(item, config, "finished", command, started_at, finished_at))
    return statuses


def render_results(config, args):
    result_dir = config["experiment"]["result_dir"]
    command = [
        sys.executable,
        "scripts/render_experiment_results.py",
        "--title", config["experiment"].get("title", "M-OWODB Pipeline Results"),
        "--method", config["experiment"]["method"],
        "--baseline", config["experiment"].get("baseline", "PROB"),
        "--output-dir", result_dir,
        "--gpus", gpu_label(args),
    ]
    for item in config["eval"]["checkpoints"]:
        command.extend(["--run", f"{item['task']}:{Path(config['eval']['output_dir']) / item['name']}"])

    print("[render]")
    if args.dry_run:
        print(shell_join(command))
        return {"status": "dry_run", "command": shell_join(command)}

    subprocess.run(command, check=True)
    return {
        "status": "finished",
        "command": shell_join(command),
        "html": str(Path(result_dir) / "owod_results.html"),
        "csv": str(Path(result_dir) / "owod_results.csv"),
        "manifest": str(Path(result_dir) / "eval_manifest.json"),
    }


def write_run_manifest(config, args, selected_stages, stage_statuses, eval_statuses, render_status, started_at):
    result_dir = Path(config["experiment"]["result_dir"])
    manifest = {
        "experiment": config["experiment"],
        "git": git_info(),
        "gpu": gpu_label(args),
        "started_at": started_at,
        "finished_at": now_iso(),
        "env": config["env"],
        "split_counts": split_counts(config),
        "selected_stages": [stage["name"] for stage in selected_stages],
        "stage_statuses": stage_statuses,
        "eval_statuses": eval_statuses,
        "render": render_status,
    }
    write_json(result_dir / "run_manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Run the resumable M-OWODB PROB pipeline.")
    parser.add_argument("--config", default="configs/pipeline_mowodb_prob.json")
    parser.add_argument("--gpu", help="physical GPU id exposed through CUDA_VISIBLE_DEVICES, e.g. 2")
    parser.add_argument("--master-port", default="29501")
    parser.add_argument("--stage", help="run exactly one training stage")
    parser.add_argument("--from-stage", help="run from this training stage through the final stage")
    parser.add_argument("--eval-only", action="store_true", help="skip training and run eval/render only")
    parser.add_argument("--no-eval", action="store_true", help="skip automatic eval/render after training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun even if stage status is finished")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-input-checks", action="store_true", help="skip server data directory checks")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.config)
    started_at = now_iso()
    check_common_inputs(config, strict=not args.skip_input_checks)
    selected = [] if args.eval_only else select_stages(config["stages"], args.stage, args.from_stage)
    env = build_env(args)

    print(f"[pipeline] config={args.config}")
    print(f"[pipeline] gpu={gpu_label(args)} dry_run={args.dry_run}")
    print(f"[pipeline] stages={[stage['name'] for stage in selected]}")
    print(f"[pipeline] split_counts={split_counts(config)}")

    stage_statuses = []
    eval_statuses = []
    render_status = None

    for stage in selected:
        stage_statuses.append(run_stage(stage, config, args, env))

    should_eval = args.eval_only or (selected and not args.stage and not args.no_eval)
    if should_eval:
        eval_statuses = run_eval(config, args, env)
        render_status = render_results(config, args)
    else:
        render_status = {"status": "skipped"}

    if not args.dry_run:
        manifest = write_run_manifest(config, args, selected, stage_statuses, eval_statuses, render_status, started_at)
        print(f"[pipeline] run_manifest={Path(config['experiment']['result_dir']) / 'run_manifest.json'}")
        print(f"[pipeline] status=finished stages={len(stage_statuses)} evals={len(eval_statuses)}")
    else:
        print("[pipeline] dry run complete")


if __name__ == "__main__":
    main()
