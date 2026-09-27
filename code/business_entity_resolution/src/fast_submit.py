"""Fast fallback submission: pass-1 fold models truncated to their first N trees, no pass 2.

Full scoring pushes ~68M test pairs through ~4,100 trees, which takes hours on a laptop CPU. This
scores them with the first --trees trees of each pass-1 fold model, picks (mode, tau) on the TRAIN
tune split scored the same way, then writes and validates the submission with stage 7.

    python -m src.fast_submit --trees 100

Writes work/test/matches.parquet, so a later full run must use `--split test --from score`.
"""
import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, s7_output
from .s3_features import column, load_matrix, shards
from .s4_score import load_models
from .s4_score import predict as predict_model
from .s4_train import TUNE, entity_roles
from .s6_decide import decide, noaddr_mask, search, tune_entities


def predict(models, x, trees):
    """Mean of the pass-1 fold models, each cut to its first `trees` trees."""
    return np.mean([predict_model(m, x, trees) for m in models['pass1']], axis=0).astype(np.float32)


def tune(models, trees):
    """The stage-6 decision search on the TUNE entities, scored with the truncated model -> best dict."""
    entities, gt = tune_entities()
    role, _ = entity_roles(np.load(config.WORK_DIR / 'train' / 'entities.npy'))
    paths = shards('train')
    s1, s23, x = load_matrix(paths, models['features'], role[column(paths, 's1')] == TUNE)
    p = predict(models, x, trees)
    del x
    _, best = search(s1, s23, p, noaddr_mask('train', s1, s23), entities, gt)
    print(f"[fast] tune split ({len(entities):,} entities): macro F0.5 {best['macro_f05']:.4f} with mode={best['mode']} "
          f"tau={best['tau']} tau_noaddr={best['tau_noaddr']}", flush=True)
    return best


def score_test(models, trees):
    """Truncated pass-1 score for every test candidate, shard by shard with progress."""
    paths = shards('test')
    s1s, s23s, ps = [], [], []
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        s1, s23, x = load_matrix([path], models['features'])
        s1s.append(s1); s23s.append(s23); ps.append(predict(models, x, trees))
        el = time.time() - t0
        print(f'[fast] shard {i}/{len(paths)}: {el / 60:.1f} min elapsed, ~{el / i * (len(paths) - i) / 60:.1f} min left',
              flush=True)
    return np.concatenate(s1s), np.concatenate(s23s), np.concatenate(ps)


def main():
    """CLI: tune (mode, tau) on TUNE with truncated pass-1 models, score test, write and validate the submission."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trees', type=int, default=100, help='trees used from each pass-1 fold model')
    args = ap.parse_args()

    models = load_models()
    best = tune(models, args.trees)
    s1, s23, p = score_test(models, args.trees)
    m = decide(s1, s23, p, best['tau'], best['mode'], noaddr_mask('test', s1, s23), best['tau_noaddr'])
    pq.write_table(pa.table({'s1': s1[m], 's23': s23[m]}), config.WORK_DIR / 'test' / 'matches.parquet')
    print(f'[fast] test: {m.sum():,} matches for {len(np.unique(s1)):,} entities with candidates', flush=True)
    s7_output.build('test')
    s7_output.validate()


if __name__ == '__main__':
    main()
