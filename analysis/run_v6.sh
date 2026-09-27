#!/usr/bin/env bash
# v6 = v5 + France pseudo-labels (confident v5 test scores) in both training passes; then score test.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK_DIR="$ROOT/work_v6" ER_OUTPUT_DIR="$ROOT/submissions/v6_france_pseudo" ER_PSEUDO_S1=50000 ER_REUSE_PASS1=1
PY="$ROOT/.venv/Scripts/python.exe"
run() { echo "[v6] $*"; "$@" >> "$ROOT/work_v6/v6_run.log" 2>&1 || { echo "[v6] FAILED $*"; exit 1; }; }
run "$PY" -m src.run_all --split train --from train
run "$PY" -m src.run_all --split test --from score
echo "[v6] done"
