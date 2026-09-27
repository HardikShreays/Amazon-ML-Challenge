"""End-to-end smoke test on a tiny slice of the REAL data — checks the plumbing, not the score.

    python -m src.smoke_test

1. builds <repo>/smoke/dataset/{train,test}/ with the organisers' file names:
     train: SMOKE_TRAIN_S1 random S1 entities + all their true matches + random unlinked distractors
     test:  SMOKE_TEST_S1 random S1 rows (France included) + SMOKE_TEST_S23 random S2/S3 rows
2. runs the unit self-checks, then `run_all --split train` and `run_all --split test` with ER_SMOKE=1
   (tiny XGBoost, small SVD) and ER_*_DIR pointing into smoke/
3. runs the official validator with --check-ids on the smoke submission

Numbers printed here say nothing about leaderboard quality: the pools are ~1000x smaller than the
real ones and the test satellites are random rows, so almost none of them are true matches.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from . import config
from .s0_ingest import read_tsv

SMOKE_TRAIN_S1, SMOKE_TRAIN_DISTRACTORS = 4000, 8000
SMOKE_TEST_S1, SMOKE_TEST_S23 = 2000, 12000
SMOKE_DIR = config.REPO_ROOT / 'smoke'
PKG_DIR = Path(__file__).resolve().parents[1]


def _write_tsv(table, path, columns):
    """Plain tab-separated writer (no quoting), same format as the organisers' files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [table[c].to_pylist() for c in columns]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(columns) + '\n')
        f.writelines('\t'.join(row) + '\n' for row in zip(*cols))


def make_subset(rng):
    """Sample the smoke dataset from the real files."""
    src, dst = config.DATA_DIR, SMOKE_DIR / 'dataset'
    cols = ['entity_id', 'business_name', 'business_address', 'country']

    gt = pacsv.read_csv(src / 'train' / 'train_ground_truth.tsv',
                        parse_options=pacsv.ParseOptions(delimiter='\t', quote_char=False),
                        convert_options=pacsv.ConvertOptions(strings_can_be_null=False, column_types={
                            'source1_entity_id': pa.string(), 'matched_entity_ids': pa.string()}))
    gt = gt.take(pa.array(rng.choice(gt.num_rows, SMOKE_TRAIN_S1, replace=False)))
    linked = {i for s in gt['matched_entity_ids'].to_pylist() for i in s.split(',') if i}
    _write_tsv(gt, dst / 'train' / 'train_ground_truth.tsv', ['source1_entity_id', 'matched_entity_ids'])

    s1 = read_tsv(src / 'train' / 'train_source1.tsv')
    _write_tsv(s1.filter(pc.is_in(s1['entity_id'], value_set=gt['source1_entity_id'])),
               dst / 'train' / 'train_source1.tsv', cols)
    for n in (2, 3):
        t = read_tsv(src / 'train' / f'train_source{n}.tsv')
        is_linked = pc.is_in(t['entity_id'], value_set=pa.array(sorted(linked))).to_numpy(zero_copy_only=False)
        distract = np.flatnonzero(~is_linked)
        pick = np.r_[np.flatnonzero(is_linked), rng.choice(distract, SMOKE_TRAIN_DISTRACTORS // 2, replace=False)]
        _write_tsv(t.take(pa.array(np.sort(pick))), dst / 'train' / f'train_source{n}.tsv', cols)
        del t

    t = read_tsv(src / 'test' / 'test_source1.tsv')
    _write_tsv(t.take(pa.array(np.sort(rng.choice(t.num_rows, SMOKE_TEST_S1, replace=False)))),
               dst / 'test' / 'test_source1.tsv', cols)
    for n in (2, 3):
        t = read_tsv(src / 'test' / f'test_source{n}.tsv')
        _write_tsv(t.take(pa.array(np.sort(rng.choice(t.num_rows, SMOKE_TEST_S23 // 2, replace=False)))),
                   dst / 'test' / f'test_source{n}.tsv', cols)
    print(f'[smoke] dataset written to {dst}')


def run(cmd, env):
    """Run one command from the package folder; raises if it fails."""
    print(f'[smoke] $ {" ".join(cmd)}')
    t0 = time.time()
    subprocess.run(cmd, cwd=PKG_DIR, env=env, check=True)
    print(f'[smoke] ok ({time.time() - t0:.1f}s)')


def main():
    """Build the smoke subset, run the self-checks, both splits end to end and the validator."""
    make_subset(np.random.default_rng(config.SEED))
    env = {**os.environ, 'ER_SMOKE': '1', 'ER_DATA_DIR': str(SMOKE_DIR / 'dataset'),
           'ER_WORK_DIR': str(SMOKE_DIR / 'work'), 'ER_OUTPUT_DIR': str(SMOKE_DIR / 'output')}
    py = sys.executable
    run([py, '-m', 'src.s1_normalise'], env)
    run([py, '-m', 'src.evaluate'], env)
    run([py, '-m', 'src.s6_decide'], env)
    run([py, '-m', 'src.run_all', '--split', 'train', '--force'], env)
    run([py, '-m', 'src.run_all', '--split', 'test', '--force'], env)
    run([py, str(config.VALIDATOR), '--matching', str(SMOKE_DIR / 'output' / 'matching_results.tsv'),
         '--candidate', str(SMOKE_DIR / 'output' / 'candidate_pairs.tsv'),
         '--test-dir', str(SMOKE_DIR / 'dataset' / 'test'), '--check-ids'], env)

    diag = json.loads((SMOKE_DIR / 'work' / 'train' / 'blocking_diagnostics.json').read_text())
    dec = json.loads((SMOKE_DIR / 'work' / 'models' / 'decision.json').read_text())
    print('\n[smoke] PASSED — every stage ran end to end and the validator accepted the output.')
    print(f"[smoke] (tiny data, not meaningful) pair completeness {diag['pair_completeness']:.3f}, "
          f"{diag['cands_per_entity']['mean']:.1f} candidates/entity, tune macro-F0.5 {dec['tune_macro_f05']:.3f} "
          f"with mode={dec['mode']} tau={dec['tau']}")


if __name__ == '__main__':
    main()
