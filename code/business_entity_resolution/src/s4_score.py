"""Stage 4b — batched inference with the trained two-pass matcher (XGBoost, GPU when available).

    pass 1  base features -> mean of the 2 fold models          (per shard, row-independent)
    context competition features on the pass-1 score           (needs every candidate -> whole split)
    pass 2  base + context -> XGBoost -> isotonic calibration   (per shard)

Only (s1, s23, p1) is held for the whole split, never the full feature matrix, so ~100M test
pairs score in a few GB of RAM; the trees run on the GPU.

Output: work/<split>/scores.parquet  (s1, s23, p)  with p a calibrated match probability.
"""
import json
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from . import config
from .s3_features import load_matrix, score_context, shards

MODEL_DIR = config.WORK_DIR / 'models'
PREDICT_BATCH = 1_000_000       # rows per GPU predict call (bounds device memory on a 4 GB card)


def predict(model, x, trees=None):
    """P(match) for a float32 matrix, in GPU-sized batches. `trees` truncates to the first N rounds."""
    rng = (0, trees) if trees else (0, 0)
    out = [model.inplace_predict(x[i:i + PREDICT_BATCH], iteration_range=rng)
           for i in range(0, len(x), PREDICT_BATCH)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def load_booster(path):
    """Load a saved XGBoost model on the configured device (GPU when available)."""
    model = xgb.Booster(model_file=str(path))
    model.set_param({'device': config.XGB_PARAMS['device'], 'nthread': config.N_JOBS})
    return model


def load_models():
    """-> dict(pass1=[fold boosters], pass2=booster, iso=IsotonicRegression)."""
    meta = json.loads((MODEL_DIR / 'meta.json').read_text())
    with open(MODEL_DIR / 'isotonic.pkl', 'rb') as f:
        iso = pickle.load(f)
    return {'pass1': [load_booster(MODEL_DIR / f) for f in meta['pass1']],
            'pass2': load_booster(MODEL_DIR / meta['pass2']), 'iso': iso,
            'features': meta['features_pass1']}                   # the columns the model was trained on


def pass1(models, x):
    """Pass-1 probability: average of the fold models (each saw half of the training entities)."""
    return np.mean([predict(m, x) for m in models['pass1']], axis=0).astype(np.float32)


def pass2(models, x, ctx):
    """Calibrated final probability from base features + pass-1 competition context."""
    raw = predict(models['pass2'], np.hstack([x, ctx]))
    return models['iso'].predict(raw).astype(np.float32)


def build(split):
    """Score every candidate pair of a split."""
    models = load_models()
    paths = shards(split)
    s1s, s23s, p1s = [], [], []
    cols = models['features']
    for p in paths:                                     # pass 1, shard by shard
        s1, s23, x = load_matrix([p], cols)
        s1s.append(s1); s23s.append(s23); p1s.append(pass1(models, x))
    s1, s23, p1 = np.concatenate(s1s), np.concatenate(s23s), np.concatenate(p1s)
    ctx = score_context(s1, s23, p1)                    # competition features over ALL candidates
    del s1s, s23s, p1s

    out, i = [], 0
    for p in paths:                                     # pass 2, same shard order
        _, _, x = load_matrix([p], cols)
        out.append(pass2(models, x, ctx[i:i + len(x)]))
        i += len(x)
    prob = np.concatenate(out) if out else np.zeros(0, np.float32)
    pq.write_table(pa.table({'s1': s1, 's23': s23, 'p': prob}), config.WORK_DIR / split / 'scores.parquet')
    print(f'[s4] {split}: scored {len(prob):,} pairs, {(prob >= 0.5).sum():,} with p >= 0.5')
