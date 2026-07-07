# Codex Project Notes

This repository is developed locally first, then synced to GitHub, then pulled on a GPU server for experiments.

## Project Positioning

This project uses PROB as the OWOD experiment pipeline and baseline. It is not a plain PROB reproduction project.

When changing code, treat the existing PROB implementation as:

- the runnable training/evaluation pipeline;
- the baseline model for final result tables and parameter comparisons;
- the control group for later model variants.

New model work should preserve a comparable PROB baseline path, keep metrics extractable from real experiment logs, and make parameter-count differences against PROB explicit.

## Working Flow

1. Edit and test code locally in `/mnt/e/paper/PB`.
2. Commit local changes and push them to the GitHub repository `git@github.com:zym-plus/PB.git`.
3. SSH into the experiment server.
4. Pull the latest code on the server from GitHub.
5. Run GPU experiments on the server, not on the local Windows/WSL checkout.

## Server Runtime

Server repository checkout:

```text
/home/zym/PB
```

Use the existing Python environment on the server:

```bash
source /home/zym/venvs/OWOD/bin/activate
```

Server dataset locations:

```text
/home/zym/data/coco
/home/zym/data/OWOD
```

When changing training, evaluation, or dataset-loading code, keep these server paths in mind. Do not assume datasets live inside this repository.

## Baseline Weights

Local PROB baseline weights are stored at:

```text
/mnt/e/paper/PB/MOWODB
```

This directory contains trained PROB M-OWODB checkpoints such as `t1.pth`, `t2.pth`, `t3.pth`, and `t4.pth`. Treat these files as external experiment artifacts, not source code. Do not upload them to GitHub.

On the GPU server, place these baseline checkpoints outside the code checkout at:

```text
/home/zym/data/prob-results/MOWODB
```

The evaluation scripts currently look for M-OWODB baseline weights under `exps/MOWODB/PROB/*.pth`. Prefer creating symlinks from `exps/MOWODB/PROB/` to `/home/zym/data/prob-results/MOWODB/` rather than copying checkpoints into the GitHub repository.

## GitHub Sync

The local repository should be pushed to:

```bash
git@github.com:zym-plus/PB.git
```

Typical local upload flow:

```bash
git status
git add <changed-files>
git commit -m "<short change summary>"
git push
```

Typical server update flow:

```bash
cd /home/zym/PB
git pull
source scripts/server_env.sh
bash scripts/smoke_server_pipeline.sh
```

`scripts/server_env.sh` activates `/home/zym/venvs/OWOD` when present and exports:

```bash
OWOD_DATA_ROOT=/home/zym/data/OWOD
COCO_PATH=/home/zym/data/coco
PROB_RESULTS_ROOT=/home/zym/data/prob-results
MOWODB_WEIGHTS_DIR=/home/zym/data/prob-results/MOWODB
OWOD_SPLITS_ROOT=<repo>/data/OWOD
```

These can be overridden before running scripts, for example:

```bash
OWOD_DATA_ROOT=/other/OWOD COCO_PATH=/other/coco ./run.sh
```

## Experiment Rules

- Prefer running expensive training and evaluation commands on the GPU server.
- Use `bash scripts/smoke_server_pipeline.sh` as the fastest post-pull check before launching long experiments. It must verify M-OWODB data, `/home/zym/data/prob-results/MOWODB` baseline weights, `exps/MOWODB/PROB` symlinks, the PROB baseline one-batch path, and the final result-table visualization path.
- Before suggesting an experiment command, check whether it needs the COCO or OWOD path and point it at `/home/zym/data/coco` or `/home/zym/data/OWOD`.
- Keep local changes commit-ready so they can be pushed before server runs.
- If the server hostname or repository path is needed and not already known, ask for it instead of guessing.

## M-OWODB Pipeline Rules

Use `scripts/run_mowodb_pipeline.py` as the preferred entry point for M-OWODB PROB training, evaluation, rendering, and resume workflows. The canonical baseline config is:

```text
configs/pipeline_mowodb_prob.json
```

The canonical stage chain is:

```text
t1 -> t2 -> t2_ft -> t3 -> t3_ft -> t4 -> t4_ft -> eval -> render
```

For new model or ablation work:

- Do not edit `configs/pipeline_mowodb_prob.json` in place for non-baseline experiments.
- Copy it to a method-specific config such as `configs/pipeline_mowodb_<method>.json`.
- Give each method its own `exps/MOWODB/<METHOD>/` and `results/MOWODB/<METHOD>/` directories.
- Preserve the same stage order, train/test splits, class counts, and eval protocol unless the research question explicitly requires a protocol change.
- Keep final metrics extractable from real `log.txt` files.
- Keep `results/MOWODB/<METHOD>/run_manifest.json`, `eval_manifest.json`, `owod_results.html`, and `owod_results.csv` as the provenance and reporting artifacts.

Recommended server checks before long runs:

```bash
cd /home/zym/PB
git pull origin main
source scripts/server_env.sh
python -m py_compile scripts/run_mowodb_pipeline.py scripts/build_comparison_manifest.py scripts/render_experiment_results.py scripts/smoke_eval_results.py main_open_world.py datasets/open_world_eval.py
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/smoke_eval_results.py --task 1 --max-images 3 --data-root /home/zym/data/OWOD --splits-root /home/zym/PB/data/OWOD --coco-path /home/zym/data/coco --checkpoint-dir exps/MOWODB/PROB --num-workers 0
python scripts/run_mowodb_pipeline.py --config configs/pipeline_mowodb_prob.json --gpu <gpu_id> --dry-run
```

Use `scripts/build_comparison_manifest.py` for final multi-model tables. Prefer comparing methods from their `run_manifest.json` files:

```bash
python scripts/build_comparison_manifest.py --title "M-OWODB Final Comparison" --baseline PROB --output results/MOWODB/model_comparison_manifest.json --output-dir results/MOWODB/comparison --run PROB:results/MOWODB/PROB/run_manifest.json --run NewModel:results/MOWODB/NewModel/run_manifest.json --render
```

Do not use `--skip-checks` for final reported comparison tables.
