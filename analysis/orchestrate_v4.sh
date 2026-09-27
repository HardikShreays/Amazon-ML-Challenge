#!/usr/bin/env bash
# Train blocking/features alone (RAM), then test blocking/features on the CPU while the GPU trains,
# then test scoring + decision + output.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
L=work_v4
( sed -i 's/--from signatures/--from blocking/' analysis/run_v4.sh; analysis/run_v4.sh train > $L/train_run.log 2>&1; echo "TRAIN_EXIT $?" >> $L/train_run.log ) &
until grep -qE "^\[run\] train/train \.\.\.|TRAIN_EXIT" $L/train_run.log 2>/dev/null; do sleep 15; done
if grep -q "TRAIN_EXIT" $L/train_run.log; then echo "[orch] train pipeline stopped before training"; exit 1; fi
echo "[orch] GPU training started; launching test blocking + features"
analysis/run_v4.sh test_prep > $L/test_prep.log 2>&1; echo "TEST_PREP_EXIT $?" >> $L/test_prep.log
until grep -q "TRAIN_EXIT" $L/train_run.log; do sleep 15; done
grep -q "TRAIN_EXIT 0" $L/train_run.log && grep -q "TEST_PREP_EXIT 0" $L/test_prep.log || { echo "[orch] a stage failed"; exit 1; }
echo "[orch] scoring test"
analysis/run_v4.sh test_finish > $L/test_finish.log 2>&1; echo "TEST_FINISH_EXIT $?" >> $L/test_finish.log
echo "[orch] done"
