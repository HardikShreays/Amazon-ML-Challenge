#!/usr/bin/env bash
# Resume v4 after the train-features OOM: finish train features (India shards kept), train on the GPU,
# tune the decision layer, then test signatures/blocking/features, then test scoring + output.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK_DIR="$ROOT/work_v4" ER_OUTPUT_DIR="$ROOT/submissions/v4_gpu_fullpop"
PY="$ROOT/.venv/Scripts/python.exe"; L="$ROOT/work_v4"
step() { local log=$1; shift; echo "[orch] $*"; "$@" >> "$L/$log" 2>&1 || { echo "[orch] FAILED: $*"; exit 1; }; }
step train_run.log "$PY" -c "from src import s3_features; s3_features.build('train', resume=True)"
step train_run.log "$PY" -m src.run_all --split train --from train
for s in s1_signatures s2_blocking s3_features; do
  step test_prep.log "$PY" -c "from src import $s; $s.build('test')"
done
step test_finish.log "$PY" -m src.run_all --split test --from score
echo "[orch] done"
