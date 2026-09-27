#!/usr/bin/env bash
# v5 retry: train (GPU with CPU fallback on OOM) + tune, then score test.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK_DIR="$ROOT/work_v5" ER_OUTPUT_DIR="$ROOT/submissions/v5_namefeats"
PY="$ROOT/.venv/Scripts/python.exe"
run() { echo "[final] $*"; "$@" >> "$ROOT/work_v5/v5_run.log" 2>&1 || { echo "[final] FAILED $*"; exit 1; }; }
run "$PY" -m src.run_all --split train --from train
run "$PY" -m src.run_all --split test --from score
echo "[final] done"
