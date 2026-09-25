"""Stage 4a — train the LightGBM matcher (two passes) and calibrate it.

Data: the candidate pairs of the TRAIN split (our own blocker's output), labelled with the ground
truth. Negatives are therefore hard by construction — blocking already judged them plausible.

Splits are GROUPED BY S1 ENTITY (a cluster never straddles train/validation):
    train  (1 - VALID_FRAC)   model fitting; 2 folds for out-of-fold pass-1 scores
    cal    (VALID_FRAC / 2)   early stopping + isotonic calibration
    tune   (VALID_FRAC / 2)   untouched until stage 6 tunes the decision layer on it

Pass 1 uses the base features. Its out-of-fold scores feed competition features (rank of the pair
by model score within the entity, margin to the best, rank among S1s claiming the same satellite),
and pass 2 is trained on base + those. Out-of-fold matters: in-sample pass-1 scores would be
over-confident and the pass-2 model would learn to trust them more than it should at test time.

Outputs: work/models/{pass1_f*.txt, pass2.txt, isotonic.pkl, meta.json, importance.csv}
         work/train/valid_scores.parquet  (s1, s23, p, label, role) for cal + tune entities
"""
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .s2_diagnostics import pair_codes
from .s3_features import BASE_FEATURES, SCORE_CONTEXT, load_matrix, score_context, shards
from .s4_score import MODEL_DIR

TRAIN, CAL, TUNE = 0, 1, 2


def entity_roles(entities):
    """Deterministic role (TRAIN / CAL / TUNE) and fold (0/1) for every S1 index."""
    rng = np.random.default_rng(config.SEED)
    u = rng.random(int(entities.max()) + 1)
    role = np.where(u < config.VALID_FRAC / 2, CAL, np.where(u < config.VALID_FRAC, TUNE, TRAIN))
    fold = (rng.random(len(u)) < 0.5).astype(np.int8)
    return role, fold


def subsample(rows, label, pre_rank, rng):
    """Keep every positive and every hard negative (pre-score rank <= HARD_NEG_RANK); downsample the
    easy negatives towards NEG_PER_POS negatives per positive (keeping at least 10% of them)."""
    y, hard = label[rows], pre_rank[rows] <= config.HARD_NEG_RANK
    easy = ~y & ~hard
    budget = config.NEG_PER_POS * y.sum() - (~y & hard).sum()
    rate = np.clip(budget / max(easy.sum(), 1), 0.1, 1.0)
    keep = y | hard | (easy & (rng.random(len(rows)) < rate))
    return rows[keep]


def fit(x, y, xv, yv, name):
    """One LightGBM model with early stopping on validation average precision (classes are imbalanced)."""
    dtrain = lgb.Dataset(x, y, free_raw_data=True)
    dvalid = lgb.Dataset(xv, yv, reference=dtrain)
    model = lgb.train(config.LGB_PARAMS, dtrain, config.NUM_BOOST_ROUND, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(config.EARLY_STOPPING, verbose=False),
                                 lgb.log_evaluation(200)])
    model.save_model(str(MODEL_DIR / f'{name}.txt'), num_iteration=model.best_iteration)
    print(f'[s4] {name}: {model.best_iteration} trees, valid AP {model.best_score["valid_0"]["average_precision"]:.4f}')
    return model


def build():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = config.WORK_DIR / 'train'
    rng = np.random.default_rng(config.SEED)
    s1, s23, x = load_matrix(shards('train'), BASE_FEATURES)
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    label = np.isin(pair_codes(s1, s23), pair_codes(gt.s1, gt.s23))
    role_of, fold_of = entity_roles(np.load(out / 'entities.npy'))
    role, fold = role_of[s1], fold_of[s1]
    pre_rank = x[:, BASE_FEATURES.index('pre_rank')]
    cal = np.flatnonzero(role == CAL)
    print(f'[s4] {len(s1):,} candidate pairs, positive rate {label.mean():.3f}; '
          f'entities: train {np.unique(s1[role == TRAIN]).size:,} / cal {np.unique(s1[cal]).size:,} / '
          f'tune {np.unique(s1[role == TUNE]).size:,}')

    # ---- pass 1: 2-fold out-of-fold scores for train rows, fold-average for cal/tune rows ----
    p1 = np.zeros(len(s1), np.float32)
    pass1_files = []
    for k in (0, 1):
        fit_rows = subsample(np.flatnonzero((role == TRAIN) & (fold != k)), label, pre_rank, rng)
        m = fit(x[fit_rows], label[fit_rows], x[cal], label[cal], f'pass1_f{k}')
        pass1_files.append(f'pass1_f{k}.txt')
        oof = np.flatnonzero((role == TRAIN) & (fold == k))
        p1[oof] = m.predict(x[oof], num_iteration=m.best_iteration)
        val = np.flatnonzero(role != TRAIN)
        p1[val] += 0.5 * m.predict(x[val], num_iteration=m.best_iteration)

    # ---- pass 2: base + competition context on the pass-1 score ----
    x2 = np.hstack([x, score_context(s1, s23, p1)])
    del x
    features2 = BASE_FEATURES + SCORE_CONTEXT
    fit_rows = subsample(np.flatnonzero(role == TRAIN), label, pre_rank, rng)
    m2 = fit(x2[fit_rows], label[fit_rows], x2[cal], label[cal], 'pass2')

    # ---- isotonic calibration on CAL, report on TUNE ----
    val = np.flatnonzero(role != TRAIN)
    raw = m2.predict(x2[val], num_iteration=m2.best_iteration)
    is_cal = role[val] == CAL
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(raw[is_cal], label[val][is_cal])
    p = iso.predict(raw).astype(np.float32)
    tune = ~is_cal
    if label[val][tune].any() and not label[val][tune].all():
        print(f'[s4] tune split: AUC {roc_auc_score(label[val][tune], p[tune]):.4f}, '
              f'AP {average_precision_score(label[val][tune], p[tune]):.4f}')

    with open(MODEL_DIR / 'isotonic.pkl', 'wb') as f:
        pickle.dump(iso, f)
    (MODEL_DIR / 'meta.json').write_text(json.dumps({'pass1': pass1_files, 'pass2': 'pass2.txt',
                                                     'features_pass1': BASE_FEATURES, 'features_pass2': features2}, indent=1))
    imp = pd.DataFrame({'feature': features2, 'gain': m2.feature_importance('gain')}).sort_values('gain', ascending=False)
    imp.to_csv(MODEL_DIR / 'importance.csv', index=False)
    print('[s4] top features by gain:', ', '.join(imp.feature.head(12)))
    pq.write_table(pa.table({'s1': s1[val], 's23': s23[val], 'p': p, 'label': label[val], 'role': role[val].astype(np.int8)}),
                   out / 'valid_scores.parquet')
