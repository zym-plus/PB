# M-OWODB PROB Pipeline Guide

This guide describes the resumable M-OWODB PROB baseline pipeline. The pipeline
keeps the original PROB training logic intact and adds a structured runner for
stage selection, preflight checks, skip/resume, evaluation, result tables, and
run manifests.

The intended research use is:

```text
reproduce PROB baseline -> archive run_manifest/results -> add one model change
-> run the same stage chain -> compare against PROB from real logs
```

Do not treat this as only a convenience script. Treat it as the experiment
contract for M-OWODB work in this repository: stages, data paths, checkpoints,
metrics, GPU identity, and result tables should all be recoverable from the
pipeline outputs.

## What The Pipeline Guarantees

The runner provides the following guarantees when an experiment finishes
successfully:

- every training stage is checked before it starts;
- generated exemplar files are checked before fine-tuning stages consume them;
- completed stages can be skipped safely with `stage_status.json`;
- failed runs can resume from a selected stage;
- evaluation is run from the configured final checkpoints;
- final HTML/CSV tables are generated from real `log.txt` files;
- `run_manifest.json` records git commit, dirty status, GPU label, split counts,
  selected stages, commands, and result locations.

This is enough for controlled baseline reproduction, model comparison, and
ablation bookkeeping. It does not by itself prove statistical significance or
multi-seed robustness; those should be added as separate repeated runs when
needed.

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

Always pull and smoke-test before launching a long run:

```bash
cd /home/zym/PB
git pull origin main
source scripts/server_env.sh
python -m py_compile \
  scripts/run_mowodb_pipeline.py \
  scripts/build_comparison_manifest.py \
  scripts/render_experiment_results.py \
  scripts/smoke_eval_results.py \
  main_open_world.py \
  datasets/open_world_eval.py
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

The stage semantics are:

```text
t1      train first 20 classes and produce t1 exemplars
t2      learn next 20 classes from t1 checkpoint and produce t2 exemplars
t2_ft   fine-tune on t2 exemplar replay set
t3      learn next 20 classes from t2_ft checkpoint and produce t3 exemplars
t3_ft   fine-tune on t3 exemplar replay set
t4      learn final 20 classes from t3_ft checkpoint and produce t4 exemplars
t4_ft   fine-tune on t4 exemplar replay set
eval    evaluate t1/t2/t3/t4 final checkpoints on owod_all_task_test
render  extract metrics from logs and write HTML/CSV tables
```

## Quick Smoke

Before long runs, run the eval-result smoke on one physical GPU. This example
uses physical GPU 1; replace `1` with the GPU id you actually want:

```bash
CUDA_VISIBLE_DEVICES=1 \
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

The smoke is not a quality result. It only verifies that the checkpoint, dataset,
evaluation path, log parser, and result renderer work together.

## Dry Run

Dry run validates the config, data directories, split files, stage selection,
and generated commands without starting training:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 1 \
  --dry-run
```

Dry run from a later stage:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --from-stage t3 \
  --gpu 1 \
  --dry-run
```

Dry run one stage:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --stage t2_ft \
  --gpu 1 \
  --dry-run
```

## Full Training

Run the full pipeline on one physical GPU:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 1
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
  --gpu 1 \
  --force
```

Use `--force` only when you intentionally want to overwrite/recompute an
already completed stage. For normal interrupted runs, prefer `--from-stage` and
the default skip behavior.

## Resume From A Stage

Resume from `t3` through the end:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --from-stage t3 \
  --gpu 1
```

Run only `t2_ft`:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --stage t2_ft \
  --gpu 1
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
  --gpu 1
```

Use this after manually fixing an evaluation/rendering issue, or when training
checkpoints already exist and you only need final metrics.

## Recommended Research Workflow

Use this workflow for each serious experiment:

1. Sync code on the server:

```bash
cd /home/zym/PB
git pull origin main
source scripts/server_env.sh
```

2. Run smoke and dry-run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_eval_results.py \
  --task 1 \
  --max-images 3 \
  --data-root /home/zym/data/OWOD \
  --splits-root /home/zym/PB/data/OWOD \
  --coco-path /home/zym/data/coco \
  --checkpoint-dir exps/MOWODB/PROB \
  --num-workers 0

python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 1 \
  --dry-run
```

3. Run the full baseline or model experiment:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_prob.json \
  --gpu 1
```

4. Confirm the result artifacts:

```bash
ls -lh \
  results/MOWODB/PROB/run_manifest.json \
  results/MOWODB/PROB/eval_manifest.json \
  results/MOWODB/PROB/owod_results.html \
  results/MOWODB/PROB/owod_results.csv

cat results/MOWODB/PROB/owod_results.csv
```

5. Save the command, commit id, config path, and CSV/HTML outputs in your
experiment notes. The manifest records these details, but the note should still
state the research purpose of the run.

## Using It For New Models

For a new method, do not edit the PROB baseline config in place. Copy it and
change only the method-specific parts:

```bash
cp configs/pipeline_mowodb_prob.json configs/pipeline_mowodb_newmodel.json
```

In the copied config, change at least:

```text
experiment.name
experiment.method
experiment.title
experiment.output_root
experiment.result_dir
experiment.wandb_name
all stage output_dir fields
all stage pretrain and expected_checkpoint fields
eval.output_dir
eval.checkpoints[*].pretrain
```

Keep the same stage order, train/test splits, class counts, and eval protocol
unless the research question explicitly requires changing them. If the model
architecture changes, keep metrics extractable from the same `log.txt` format
or update the parser and smoke test in the same commit.

Run the new method with the same interface:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_newmodel.json \
  --gpu 1
```

## Ablation Workflow

Each ablation should have its own config and result directory:

```text
configs/pipeline_mowodb_<method>_<ablation>.json
exps/MOWODB/<METHOD>_<ABLATION>/
results/MOWODB/<METHOD>_<ABLATION>/
```

Change one experimental factor at a time when possible. Examples:

```text
objectness loss coefficient
temperature
feature branch on/off
replay budget
freezing policy
new module enabled/disabled
```

For early debugging, run one stage only:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_newmodel.json \
  --stage t1 \
  --gpu 1
```

For a valid final table, run the complete chain or resume through the final
stage and then run eval/render:

```bash
python scripts/run_mowodb_pipeline.py \
  --config configs/pipeline_mowodb_newmodel.json \
  --from-stage t3 \
  --gpu 1
```

## What Counts As A Complete Experiment

A run is complete only when all of these exist:

```text
all configured final checkpoints
all needed exemplar files for fine-tuning stages
exps/MOWODB/<METHOD>/eval/t1/log.txt
exps/MOWODB/<METHOD>/eval/t2/log.txt
exps/MOWODB/<METHOD>/eval/t3/log.txt
exps/MOWODB/<METHOD>/eval/t4/log.txt
results/MOWODB/<METHOD>/run_manifest.json
results/MOWODB/<METHOD>/eval_manifest.json
results/MOWODB/<METHOD>/owod_results.html
results/MOWODB/<METHOD>/owod_results.csv
```

The CSV is the quick table for inspection. The HTML is the visual report. The
manifests are the provenance records.

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

## Multi-Model Comparison

The final comparison table can combine multiple methods as long as each method
has four eval task directories with `log.txt`:

```text
exps/MOWODB/<METHOD>/eval/t1/log.txt
exps/MOWODB/<METHOD>/eval/t2/log.txt
exps/MOWODB/<METHOD>/eval/t3/log.txt
exps/MOWODB/<METHOD>/eval/t4/log.txt
```

Build and render a comparison from pipeline run manifests:

```bash
python scripts/build_comparison_manifest.py \
  --title "M-OWODB Final Comparison" \
  --baseline PROB \
  --output results/MOWODB/model_comparison_manifest.json \
  --output-dir results/MOWODB/comparison \
  --run PROB:results/MOWODB/PROB/run_manifest.json \
  --run NewModel:results/MOWODB/NewModel/run_manifest.json \
  --render
```

This should be the default comparison path for paper tables because it reads
each method from the pipeline provenance record.

You can also use eval manifests:

```bash
python scripts/build_comparison_manifest.py \
  --title "M-OWODB Final Comparison" \
  --baseline PROB \
  --output results/MOWODB/model_comparison_manifest.json \
  --output-dir results/MOWODB/comparison \
  --run PROB:results/MOWODB/PROB/eval_manifest.json \
  --run NewModel:results/MOWODB/NewModel/eval_manifest.json \
  --render
```

Or pass eval directories directly:

```bash
python scripts/build_comparison_manifest.py \
  --title "M-OWODB Final Comparison" \
  --baseline PROB \
  --output results/MOWODB/model_comparison_manifest.json \
  --output-dir results/MOWODB/comparison \
  --run PROB:exps/MOWODB/PROB/eval \
  --run NewModel:exps/MOWODB/NewModel/eval \
  --render
```

Comparison outputs:

```text
results/MOWODB/model_comparison_manifest.json
results/MOWODB/comparison/owod_results.html
results/MOWODB/comparison/owod_results.csv
```

Recommended final comparison command after running PROB and a new model:

```bash
python scripts/build_comparison_manifest.py \
  --title "M-OWODB Final Comparison" \
  --baseline PROB \
  --output results/MOWODB/model_comparison_manifest.json \
  --output-dir results/MOWODB/comparison \
  --run PROB:results/MOWODB/PROB/run_manifest.json \
  --run NewModel:results/MOWODB/NewModel/run_manifest.json \
  --render
```

The comparison script fails if any method is missing `t1` through `t4` logs,
unless `--skip-checks` is explicitly passed. Do not use `--skip-checks` for
final reported results.

## Reading The Result Table

The generated CSV contains one row per method/task. The most important columns
are:

```text
U_R50      unknown recall at IoU 0.50
PK_AP50    previously known class AP50, when applicable
CK_AP50    currently known class AP50
K_AP50     known class AP50 aggregate
WI         wilderness impact
AOSA       absolute open-set error area
```

For task 1, `PK_AP50` can be blank because there are no previous known classes.
For task 4, unknown recall can be `nan` or zero depending on whether the
benchmark split exposes unknown classes for that final setting. Interpret task
4 open-set columns with the benchmark protocol, not as a parser failure by
default.

## Research Discipline

Use these rules when preparing publishable numbers:

- Keep one clean PROB baseline run for every major code revision that changes
  training, evaluation, losses, dataset loading, or model outputs.
- Never compare a new method against a baseline produced by a different eval
  protocol.
- Keep result tables generated from logs, not copied manually.
- Keep external checkpoint weights outside GitHub.
- Commit the config and code before launching a long server run.
- Record failed experiments too; their `stage_status.json` files are useful for
  diagnosing where the pipeline broke.
- If you change metric extraction, add or rerun the smoke test before trusting
  new CSV/HTML outputs.

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
  --gpu <gpu_id>
```

Unexpected GPU label:

The pipeline records `GPU <gpu_id>` when invoked with `--gpu <gpu_id>`. This
means the physical GPU id exposed through `CUDA_VISIBLE_DEVICES=<gpu_id>`, not
the number of GPUs.
