#!/usr/bin/env bash

source "$(dirname "$0")/../scripts/server_env.sh"

echo running eval ofnano prob-detr, M-OWODB dataset

set -x

EXP_DIR=exps/MOWODB/PROB
EVAL_DIR="${EXP_DIR}/eval"
RESULT_DIR=results/MOWODB/PROB
PY_ARGS=${@:1}
WANDB_NAME=PROB_V1


PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EVAL_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project ""\
    ${PY_ARGS}
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EVAL_DIR}/t2" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t2.pth" --eval --wandb_project ""\
    ${PY_ARGS}

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EVAL_DIR}/t3" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t3.pth" --eval --wandb_project ""\
    ${PY_ARGS}


PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EVAL_DIR}/t4" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t4.pth" --eval --wandb_project ""\
    ${PY_ARGS}

if [ "${RANK:-0}" = "0" ]; then
    python scripts/render_experiment_results.py \
        --title "M-OWODB PROB Evaluation Results" \
        --method PROB --baseline PROB --output-dir "${RESULT_DIR}" \
        --run "1:${EVAL_DIR}/t1" \
        --run "2:${EVAL_DIR}/t2" \
        --run "3:${EVAL_DIR}/t3" \
        --run "4:${EVAL_DIR}/t4"
fi
