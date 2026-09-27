#!/usr/bin/env bash
# v4: full-population train blocking, K6/K7 keys, empty-address pre-score, leading-zero numbers,
# XGBoost on the GPU. Isolated in work_v4. Usage: analysis/run_v4.sh train|test_prep|test_finish
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK_DIR="$ROOT/work_v4" ER_OUTPUT_DIR="$ROOT/submissions/v4_gpu_fullpop"
PY="$ROOT/.venv/Scripts/python.exe"
case "$1" in
  train)       "$PY" -m src.run_all --split train --from blocking ;;
  test_prep)   for s in s1_signatures s2_blocking s3_features; do
                 t=$(date +%s); echo "[v4] test $s ..."; "$PY" -c "from src import $s; $s.build('test')"
                 echo "[v4] test $s finished in $(( $(date +%s) - t ))s"; done ;;
  test_finish) "$PY" -m src.run_all --split test --from score ;;
esac
