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

The currently available server checkpoint directory is:

```text
/home/zym/data/prob-results/MOWODB/MOWODB
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
