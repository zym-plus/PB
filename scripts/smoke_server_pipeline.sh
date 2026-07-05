#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/server_env.sh

python scripts/smoke_server_pipeline.py "$@"
