#!/usr/bin/env bash

if [ -n "${OWOD_SERVER_ENV_LOADED:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
export OWOD_SERVER_ENV_LOADED=1

OWOD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OWOD_VENV="${OWOD_VENV:-/home/zym/venvs/OWOD}"
export OWOD_DATA_ROOT="${OWOD_DATA_ROOT:-/home/zym/data/OWOD}"
export COCO_PATH="${COCO_PATH:-/home/zym/data/coco}"
export PROB_RESULTS_ROOT="${PROB_RESULTS_ROOT:-/home/zym/data/prob-results}"
export MOWODB_WEIGHTS_DIR="${MOWODB_WEIGHTS_DIR:-${PROB_RESULTS_ROOT}/MOWODB}"
export OWOD_SPLITS_ROOT="${OWOD_SPLITS_ROOT:-${OWOD_REPO_ROOT}/data/OWOD}"

case ":${PYTHONPATH:-}:" in
    *":${OWOD_REPO_ROOT}:"*) ;;
    *) export PYTHONPATH="${OWOD_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

if [ -z "${OWOD_SKIP_VENV:-}" ]; then
    if [ -f "${OWOD_VENV}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${OWOD_VENV}/bin/activate"
    else
        echo "warning: OWOD virtualenv not found at ${OWOD_VENV}" >&2
    fi
fi
