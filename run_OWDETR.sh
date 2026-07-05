#!/bin/bash

source "$(dirname "$0")/scripts/server_env.sh"

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/S_OWOD_BENCHMARK.sh
