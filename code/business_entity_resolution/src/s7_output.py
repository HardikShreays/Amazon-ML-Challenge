"""Stage 7 — write the two submission files and run the organisers' validator.

    output/matching_results.tsv   source1_entity_id \\t matched_entity_ids     (leaderboard file)
    output/candidate_pairs.tsv    source1_entity_id \\t candidate_entity_ids   (stage-2 candidate set)

One row per S1 entity in the original file order (empty list when nothing matched), ids
comma-separated with no quoting, UTF-8, tab-separated. Written in streaming chunks so the ~70M
candidate ids never sit in memory as Python strings at once.
"""
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from . import config


def write_id_lists(path, column, s1_ids, s23_ids, s1, s23, chunk=200_000):
    """Write `source1_entity_id \\t <column>` with every S1 entity once, in index order."""
    order = np.argsort(s1, kind='stable')
    s1, s23 = np.asarray(s1)[order], np.asarray(s23)[order]
    n = len(s1_ids)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'source1_entity_id\t{column}\n')
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            a, b = np.searchsorted(s1, [lo, hi])
            ids = s23_ids.take(s23[a:b]).to_pylist()
            counts = np.bincount(s1[a:b] - lo, minlength=hi - lo)
            ends = np.cumsum(counts)
            names = s1_ids[lo:hi].to_pylist()
            f.writelines(f'{names[i]}\t{",".join(ids[ends[i] - counts[i]:ends[i]])}\n' for i in range(hi - lo))


def build(split='test'):
    """Write candidate_pairs.tsv (the scored candidate set) and matching_results.tsv (the decided matches) for `split`."""
    out = config.WORK_DIR / split
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_ids = pq.read_table(out / 's1_raw.parquet', columns=['entity_id'])['entity_id'].combine_chunks()
    s23_ids = pq.read_table(out / 's23_raw.parquet', columns=['entity_id'])['entity_id'].combine_chunks()
    cand = pq.read_table(out / 'candidates.parquet', columns=['s1', 's23'])
    match = pq.read_table(out / 'matches.parquet')
    write_id_lists(config.OUTPUT_DIR / 'candidate_pairs.tsv', 'candidate_entity_ids', s1_ids, s23_ids,
                   cand['s1'].to_numpy(), cand['s23'].to_numpy())
    write_id_lists(config.OUTPUT_DIR / 'matching_results.tsv', 'matched_entity_ids', s1_ids, s23_ids,
                   match['s1'].to_numpy(), match['s23'].to_numpy())
    print(f'[s7] wrote {config.OUTPUT_DIR / "matching_results.tsv"} and candidate_pairs.tsv '
          f'({len(s1_ids):,} rows each)')


def validate(check_ids=False):
    """Run the organisers' validator (vendored as src/validate_submission.py); raises unless it prints PASS."""
    cmd = [sys.executable, str(config.VALIDATOR),
           '--matching', str(config.OUTPUT_DIR / 'matching_results.tsv'),
           '--candidate', str(config.OUTPUT_DIR / 'candidate_pairs.tsv'),
           '--test-dir', str(config.DATA_DIR / 'test')] + (['--check-ids'] if check_ids else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        raise SystemExit(f'validator FAILED:\n{res.stdout}\n{res.stderr}')
