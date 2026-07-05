#!/bin/bash
#
#sBATCH --job-name=owdetr
#SBATCH --gpus=16
#SBATCH --nodes=2
#SBATCH --time=3-00:00:00
#

source "$(dirname "$0")/scripts/server_env.sh"

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh berz deformable_detr 16 configs/S_OWOD_BENCHMARK.sh
