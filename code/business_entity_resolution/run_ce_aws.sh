#!/usr/bin/env bash
# Stage 8b on the GPU box. Usage: bash run_ce_aws.sh <model> <num_gpus> <train_minutes>
#   bash run_ce_aws.sh Qwen/Qwen3-0.6B 1 50      # single A10G / L40S
#   bash run_ce_aws.sh Qwen/Qwen3-1.7B 4 45      # g6e.12xlarge / g5.12xlarge (4 GPUs)
set -euo pipefail
MODEL=${1:-Qwen/Qwen3-0.6B}; N=${2:-1}; MIN=${3:-50}
EXTRA=${EXTRA:-}               # e.g. EXTRA=--gc for the 1.7B model on 24 GB GPUs
T="torchrun --nproc_per_node=$N src/s8_ce_gpu.py"
$T train --model "$MODEL" --data ce/train.parquet --out ce/lora --minutes "$MIN" $EXTRA 2>&1 | tee ce/train.log
$T score --model "$MODEL" --adapter ce/lora --data ce/tune.parquet --out ce/tune_ce.parquet 2>&1 | tee ce/score_tune.log
$T score --model "$MODEL" --adapter ce/lora --data ce/test.parquet --out ce/test_ce.parquet 2>&1 | tee ce/score_test.log
echo "DONE -> download ce/tune_ce.parquet and ce/test_ce.parquet"
