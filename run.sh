#!/bin/bash

source "$(dirname "$0")/scripts/server_env.sh"

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/M_OWOD_BENCHMARK.sh
