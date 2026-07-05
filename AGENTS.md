# Codex Project Notes

This repository is developed locally first, then synced to GitHub, then pulled on a GPU server for experiments.

## Working Flow

1. Edit and test code locally in `/mnt/e/paper/PB`.
2. Commit local changes and push them to the GitHub repository `git@github.com:zym-plus/PB.git`.
3. SSH into the experiment server.
4. Pull the latest code on the server from GitHub.
5. Run GPU experiments on the server, not on the local Windows/WSL checkout.

## Server Runtime

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
cd <server-repo-path>
git pull
source scripts/server_env.sh
```

`scripts/server_env.sh` activates `/home/zym/venvs/OWOD` when present and exports:

```bash
OWOD_DATA_ROOT=/home/zym/data/OWOD
COCO_PATH=/home/zym/data/coco
OWOD_SPLITS_ROOT=<repo>/data/OWOD
```

These can be overridden before running scripts, for example:

```bash
OWOD_DATA_ROOT=/other/OWOD COCO_PATH=/other/coco ./run.sh
```

## Experiment Rules

- Prefer running expensive training and evaluation commands on the GPU server.
- Before suggesting an experiment command, check whether it needs the COCO or OWOD path and point it at `/home/zym/data/coco` or `/home/zym/data/OWOD`.
- Keep local changes commit-ready so they can be pushed before server runs.
- If the server hostname or repository path is needed and not already known, ask for it instead of guessing.
