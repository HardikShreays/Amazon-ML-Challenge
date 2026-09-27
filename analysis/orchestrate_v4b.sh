#!/usr/bin/env bash
# Sequential continuation: wait for the train pipeline, then test blocking/features, then scoring.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; L=work_v4
until grep -q "TRAIN_EXIT" $L/train_run.log; do sleep 15; done
grep -q "TRAIN_EXIT 0" $L/train_run.log || { echo "[orch] train failed"; exit 1; }
echo "[orch] train done; test blocking + features"
analysis/run_v4.sh test_prep > $L/test_prep.log 2>&1 || { echo "[orch] test_prep failed"; exit 1; }
echo "[orch] scoring test"
analysis/run_v4.sh test_finish > $L/test_finish.log 2>&1 || { echo "[orch] test_finish failed"; exit 1; }
echo "[orch] done"
