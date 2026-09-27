#!/usr/bin/env bash
# v5 = v4 + name-rarity features. Waits for v4 to finish, then: append the name columns to the train
# shards, retrain (GPU) + re-tune in work_v5/models, and score test with it. work_v5/{train,test} are
# junctions to work_v4's, so blocking and test features are shared; v4's models stay untouched.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/code/business_entity_resolution"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK_DIR="$ROOT/work_v5" ER_OUTPUT_DIR="$ROOT/submissions/v5_namefeats"
PY="$ROOT/.venv/Scripts/python.exe"; L="$ROOT/work_v5"
until grep -qE "^\[orch\] (done|FAILED)" "$ROOT/work_v4/orchestrate.log"; do sleep 20; done
grep -q "^\[orch\] done" "$ROOT/work_v4/orchestrate.log" || { echo "[orch5] v4 failed, not starting"; exit 1; }
step() { echo "[orch5] $*"; "$@" >> "$L/v5_run.log" 2>&1 || { echo "[orch5] FAILED: $*"; exit 1; }; }
step "$PY" -c "from src import s3_features; s3_features.augment('train'); s3_features.augment('test')"
step "$PY" -m src.run_all --split train --from train
step "$PY" -m src.run_all --split test --from score
echo "[orch5] done"
