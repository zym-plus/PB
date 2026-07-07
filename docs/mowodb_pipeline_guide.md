# M-OWODB PROB Pipeline Guide

This guide describes the resumable M-OWODB PROB baseline pipeline. The pipeline
keeps the original PROB training logic intact and adds a structured runner for
stage selection, preflight checks, skip/resume, evaluation, result tables, and
run manifests.

## Server Setup

Use the server checkout and existing Python environment:

```bash
cd /home/zym/PB
git pull origin main
source scripts/server_env.sh
```

Expected server paths:

```text
repo:        /home/zym/PB
venv:        /home/zym/venvs/OWOD
OWOD data:   /home/zym/data/OWOD
COCO data:   /home/zym/data/coco
splits:      /home/zym/PB/data/OWOD
```

The main pipeline config is:

```text
configs/pipeline_mowodb_prob.json
```

## Stage Order

The full training chain is:

```text
t1 -> t2 -> t2_ft -> t3 -> t3_ft -> t4 -> t4_ft -> eval -> render
```

The final training checkpoints used for evaluation are:

```text
t1: exps/MOWODB/PROB/t1/checkpoint0040.pth
t2: exps/MOWODB/PROB/t2_ft/checkpoint0110.pth
t3: exps/MOWODB/PROB/t3_ft/checkpoint0180.pth
t4: exps/MOWODB/PROB/t4_ft/checkpoint0260.pth
```

## Quick Smoke

Before long runs, run the eval-result smoke on physical GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 \
python scripts/smoke_eval_results.py \
  --task 1 \
  --max-images 3 \
  --data-root /home/zym/data/OWOD \
  --splits-root /home/zym/PB/data/OWOD \
  --coco-path /home/zym/data/coco \
  --checkpoint-dir exps/MOWODB/PROB \
  --num-workers 0
```

Success ends with:

```text
[smoke-eval] OK
```

## Dry Run

Dry run validates the config, data directories, split files, stage selection,
and generated commands without starting training:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 2 \
  --dry-run
```

Dry run from a later stage:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --from-stage t3 \
  --gpu 2 \
  --dry-run
```

Dry run one stage:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --stage t2_ft \
  --gpu 2 \
  --dry-run
```

## Full Training

Run the full pipeline on physical GPU 2:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 2
```

By default, completed stages are skipped when all of these are present:

```text
stage_status.json with status=finished
expected checkpoint
log.txt
generated exemplar file, when the stage produces one
```

Force rerun:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 2 \
  --force
```

## Resume From A Stage

Resume from `t3` through the end:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --from-stage t3 \
  --gpu 2
```

Run only `t2_ft`:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --stage t2_ft \
  --gpu 2
```

When running a single stage, automatic final evaluation is skipped. Use
`--eval-only` after the needed checkpoints exist.

## Eval Only

Run evaluation and render the final table using the configured trained
checkpoints:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --eval-only \
  --gpu 2
```

## Outputs

Per-stage training outputs:

```text
exps/MOWODB/PROB/t1/
exps/MOWODB/PROB/t2/
exps/MOWODB/PROB/t2_ft/
exps/MOWODB/PROB/t3/
exps/MOWODB/PROB/t3_ft/
exps/MOWODB/PROB/t4/
exps/MOWODB/PROB/t4_ft/
```

Per-stage status:

```text
exps/MOWODB/PROB/<stage>/stage_status.json
```

Evaluation logs:

```text
exps/MOWODB/PROB/eval/t1/log.txt
exps/MOWODB/PROB/eval/t2/log.txt
exps/MOWODB/PROB/eval/t3/log.txt
exps/MOWODB/PROB/eval/t4/log.txt
```

Final result files:

```text
results/MOWODB/PROB/owod_results.html
results/MOWODB/PROB/owod_results.csv
results/MOWODB/PROB/eval_manifest.json
results/MOWODB/PROB/run_manifest.json
```

Stage tee logs:

```text
logs/pipeline_t1.log
logs/pipeline_t2.log
logs/pipeline_t2_ft.log
...
logs/pipeline_eval_t1.log
...
```

## Dataset Sizes

The TOWOD split sizes used by this pipeline are:

```text
owod_t1_train:       16551
owod_t2_train:       45520
owod_t3_train:       39402
owod_t4_train:       40260
owod_all_task_test:  10246
```

Fine-tune exemplar files are generated during training under:

```text
data/OWOD/ImageSets/TOWOD/PROB_V1/
```

Expected maximum sizes:

```text
learned_owod_t1_ft.txt: 850
learned_owod_t2_ft.txt: 1743
learned_owod_t3_ft.txt: 2361
learned_owod_t4_ft.txt: 2749
```

## Troubleshooting

Missing pretrain:

```text
missing pretrain for stage t3: exps/MOWODB/PROB/t2_ft/checkpoint0110.pth
```

Resume from the stage that should create that checkpoint, or verify previous
stage completion.

Missing exemplar:

```text
missing required exemplar for stage t2_ft
```

Run or resume the previous exemplar-producing stage.

No final result table:

Check eval logs first:

```bash
find exps/MOWODB/PROB/eval -maxdepth 2 -name log.txt -print -exec tail -n 1 {} \;
```

Then rerun eval only:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --eval-only \
  --gpu 2
```

Unexpected GPU label:

The pipeline records `GPU 2` when invoked with `--gpu 2`. This means physical
GPU id 2 through `CUDA_VISIBLE_DEVICES=2`, not two GPUs.

