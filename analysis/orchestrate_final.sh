#!/usr/bin/env bash
# Strictly sequential finish (one heavy job at a time):
#   wait for the running train-shard augment -> resume v4 test features (France kept) ->
#   v4 test scoring + submission -> v5: augment test, retrain with name features, score test.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY="$ROOT/.venv/Scripts/python.exe"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match 'augment' }) { exit 0 } else { exit 1 }"; do sleep 20; done
echo "[final] augment finished"
run() { local tag=$1 log=$2; shift 2; echo "[final] $tag: $*"; "$@" >> "$log" 2>&1 || { echo "[final] FAILED $tag"; exit 1; }; }
V4=("ER_WORK_DIR=$ROOT/work_v4" "ER_OUTPUT_DIR=$ROOT/submissions/v4_gpu_fullpop")
V5=("ER_WORK_DIR=$ROOT/work_v5" "ER_OUTPUT_DIR=$ROOT/submissions/v5_namefeats")
run v4-test-features "$ROOT/work_v4/test_prep.log" env "${V4[@]}" "$PY" -c "from src import s3_features; s3_features.build('test', resume=True)"
run v4-score "$ROOT/work_v4/test_finish.log" env "${V4[@]}" "$PY" -m src.run_all --split test --from score
echo "[final] v4 submission written"
run v5-augment "$ROOT/work_v5/v5_run.log" env "${V5[@]}" "$PY" -c "from src import s3_features; s3_features.augment('train'); s3_features.augment('test')"
run v5-train "$ROOT/work_v5/v5_run.log" env "${V5[@]}" "$PY" -m src.run_all --split train --from train
run v5-score "$ROOT/work_v5/v5_run.log" env "${V5[@]}" "$PY" -m src.run_all --split test --from score
echo "[final] done"
