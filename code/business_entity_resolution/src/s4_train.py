"""Stage 4a — train the XGBoost matcher (two passes, GPU) and calibrate it.

Data: the candidate pairs of the TRAIN split (our own blocker's output), labelled with the ground
truth. Negatives are therefore hard by construction — blocking already judged them plausible.

Every train S1 entity is blocked (as in test), so each satellite sees all of its real competitors
and the competition features / partition behave exactly as they will on test. Splits are GROUPED
BY S1 ENTITY (a cluster never straddles train/validation):
    train  (1 - VALID_FRAC)   model fitting on a sample of TRAIN_FIT_S1 of them; 2 folds for
                              out-of-fold pass-1 scores
    cal    (VALID_FRAC / 2)   early stopping + isotonic calibration
    tune   (VALID_FRAC / 2)   untouched until stage 6 tunes the decision layer on it

Pass 1 uses the base features. Its out-of-fold scores feed competition features (rank of the pair
by model score within the entity, margin to the best, rank among S1s claiming the same satellite),
and pass 2 is trained on base + those. Out-of-fold matters: in-sample pass-1 scores would be
over-confident and the pass-2 model would learn to trust them more than it should at test time.
Pass-1 scores are computed for EVERY train candidate (fit entities get their out-of-fold model,
all others the fold average, as test does), so the context features see every competitor.

Outputs: work/models/{pass1_f*.ubj, pass2.ubj, isotonic.pkl, meta.json, importance.csv}
         work/train/valid_scores.parquet  (s1, s23, p, label, role) for cal + tune entities
"""
import gc
import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .s2_diagnostics import pair_codes
from .s3_features import BASE_FEATURES, SCORE_CONTEXT, column, load_matrix, score_context, shards
from .s4_score import MODEL_DIR, load_booster, predict

TRAIN, CAL, TUNE = 0, 1, 2


def entity_roles(entities):
    """Deterministic role (TRAIN / CAL / TUNE) and fold (0/1) for every S1 index."""
    rng = np.random.default_rng(config.SEED)
    u = rng.random(int(entities.max()) + 1)
    role = np.where(u < config.VALID_FRAC / 2, CAL, np.where(u < config.VALID_FRAC, TUNE, TRAIN)).astype(np.int8)
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
    """One XGBoost model with early stopping on validation average precision (classes are imbalanced)."""
    params = dict(config.XGB_PARAMS)
    while True:
        dtrain = xgb.QuantileDMatrix(x, y, max_bin=params['max_bin'])
        dvalid = xgb.QuantileDMatrix(xv, yv, ref=dtrain)
        try:
            model = xgb.train(params, dtrain, config.NUM_BOOST_ROUND, evals=[(dvalid, 'valid')],
                              early_stopping_rounds=config.EARLY_STOPPING, verbose_eval=200)
            break
        except xgb.core.XGBoostError as e:              # a 4 GB card shared with the desktop can run out
            if params['device'] == 'cpu' or 'out of memory' not in str(e):
                raise
            del dtrain, dvalid
            gc.collect()
            if params['max_bin'] > 64:                  # coarser histograms first: far less device memory
                params['max_bin'] = 64
                print(f'[s4] {name}: GPU out of memory, retrying on the GPU with max_bin=64', flush=True)
            else:
                params['device'] = 'cpu'
                print(f'[s4] {name}: GPU out of memory, retrying on the CPU', flush=True)
    best, score = model.best_iteration, model.best_score
    model = model[:best + 1]                                     # keep only the trees up to the best round
    model.save_model(str(MODEL_DIR / f'{name}.ubj'))
    print(f'[s4] {name}: {best + 1} trees on {len(y):,} rows, valid AP {score:.4f}', flush=True)
    return model


def masked(n, *row_sets):
    """Boolean row mask of length n with every listed row set, and the row -> matrix position map."""
    mask = np.zeros(n, bool)
    for r in row_sets:
        mask[r] = True
    return mask, np.cumsum(mask) - 1


class Pseudo:
    """Confidently scored test pairs of countries absent from train (France), used as extra training
    rows (config.PSEUDO_S1 > 0). Labels come from a previous model's test scores.parquet: p >= PSEUDO_HI
    is a match, p <= PSEUDO_LO a non-match, anything in between is left out. No ground truth is used.
    Simulated on train with India held out: a US-only model scores India 0.9253, the same model trained
    with India pseudo-labels 0.9314 (in-domain reference 0.9704)."""

    def __init__(self, rng):
        test = config.WORK_DIR / 'test'
        self.paths = shards('test')
        s1 = column(self.paths, 's1')
        sc = pq.read_table(test / 'scores.parquet').to_pandas()
        assert len(sc) == len(s1) and (sc.s1.to_numpy() == s1).all(), 'scores.parquet is not aligned with the test shards'
        country = pq.read_table(test / 's1_sig.parquet', columns=['country'])['country'].to_numpy(zero_copy_only=False)
        seen = set(pq.read_table(config.WORK_DIR / 'train' / 's1_sig.parquet', columns=['country'])['country']
                   .unique().to_pylist())
        self.unseen = ~np.isin(country, sorted(seen))[s1]
        ents = np.unique(s1[self.unseen])
        pick = np.zeros(len(country), bool)
        pick[rng.choice(ents, min(config.PSEUDO_S1, len(ents)), replace=False)] = True
        self.fold_of = (rng.random(len(country)) < 0.5).astype(np.int8)
        p = sc.p.to_numpy()
        conf = (p >= config.PSEUDO_HI) | (p <= config.PSEUDO_LO)
        self.s1, self.s23, self.pre = s1, sc.s23.to_numpy(), column(self.paths, 'pre_rank')
        self.label = p >= config.PSEUDO_HI
        self.rows = np.flatnonzero(self.unseen & pick[s1] & conf)
        self.fold, self.picked = self.fold_of[s1], pick[s1]
        print(f'[s4] pseudo-labels: {pick.sum():,} unseen-country entities, {len(self.rows):,} confident pairs '
              f'({self.label[self.rows].mean():.3f} positive)', flush=True)

    def matrix(self, rows, ctx=None):
        """Base features of the given (sorted) test rows, plus their pass-1 context when `ctx` is given."""
        mask, pos = masked(len(self.s1), rows)
        _, _, x = load_matrix(self.paths, BASE_FEATURES, mask, extra=0 if ctx is None else len(SCORE_CONTEXT))
        if ctx is not None:                              # loaded rows come in sorted order
            x[:, len(BASE_FEATURES):] = ctx[np.searchsorted(self.u, np.flatnonzero(mask))]
        return x[pos[rows]]

    def context(self, models):
        """Pass-1 competition context over the whole unseen-country test population (OOF for picked
        entities) -> array aligned with self.u (the unseen-country rows only)."""
        self.u = np.flatnonzero(self.unseen)
        u = self.u
        x = self.matrix(u)
        pk = np.stack([predict(m, x) for m in models])
        del x
        p1 = np.where(self.picked[u], pk[self.fold[u], np.arange(len(u))], pk.mean(axis=0))
        return score_context(self.s1[u], self.s23[u], p1)


def build():
    """Train the matcher: 2-fold OOF pass 1, competition features on its scores, pass 2, isotonic calibration on CAL; writes models and TUNE/CAL valid_scores.parquet."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = config.WORK_DIR / 'train'
    rng = np.random.default_rng(config.SEED)
    paths = shards('train')
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    entities = np.load(out / 'entities.npy')
    role_of, fold_of = entity_roles(entities)
    train_ents = entities[role_of[entities] == TRAIN]
    if config.TRAIN_FIT_S1 and config.TRAIN_FIT_S1 < len(train_ents):
        train_ents = rng.choice(train_ents, config.TRAIN_FIT_S1, replace=False)
    fit_of = np.zeros(len(role_of), bool)
    fit_of[train_ents] = True

    s1, s23, pre_rank = column(paths, 's1'), column(paths, 's23'), column(paths, 'pre_rank')
    label = np.isin(pair_codes(s1, s23), pair_codes(gt.s1, gt.s23))
    role, fold, fit_row = role_of[s1], fold_of[s1], fit_of[s1]
    cal, val = np.flatnonzero(role == CAL), np.flatnonzero(role != TRAIN)
    print(f'[s4] {len(s1):,} candidate pairs, positive rate {label.mean():.3f}; entities: fit {len(train_ents):,} / '
          f'cal {np.unique(s1[cal]).size:,} / tune {np.unique(s1[role == TUNE]).size:,}', flush=True)

    ps = Pseudo(rng) if config.PSEUDO_S1 else None

    def with_pseudo(x, y, rows_ps, ctx=None):
        """Append pseudo-labelled unseen-country rows (if any) to a fit matrix."""
        if ps is None or not len(rows_ps):
            return x, y
        return np.vstack([x, ps.matrix(rows_ps, ctx)]), np.r_[y, ps.label[rows_ps]]

    # ---- pass 1: 2-fold models on the fit entities (easy negatives thinned) ----
    fold_rows = [subsample(np.flatnonzero(fit_row & (fold != k)), label, pre_rank, rng) for k in (0, 1)]
    ps_rows = [subsample(ps.rows[ps.fold[ps.rows] != k], ps.label, ps.pre, rng) if ps else [] for k in (0, 1)]
    mask, pos = masked(len(s1), cal, *fold_rows)
    _, _, x = load_matrix(paths, BASE_FEATURES, mask)
    if config.REUSE_PASS1 and all((MODEL_DIR / f'pass1_f{k}.ubj').exists() for k in (0, 1)):
        models = [load_booster(MODEL_DIR / f'pass1_f{k}.ubj') for k in (0, 1)]   # same seed -> same rows
        print('[s4] pass 1: reusing the saved fold models', flush=True)
    else:
        models = [fit(*with_pseudo(x[pos[r]], label[r], ps_rows[k]), x[pos[cal]], label[cal], f'pass1_f{k}')
                  for k, r in enumerate(fold_rows)]
    del x
    ps_ctx = ps.context(models) if ps else None

    # ---- pass-1 score for EVERY train candidate (the context needs all competitors) ----
    # model k never saw fold-k fit entities -> out-of-fold for them; everyone else gets the fold average
    p1 = np.empty(len(s1), np.float32)
    i = 0
    for path in paths:
        _, _, xs = load_matrix([path], BASE_FEATURES)
        n = len(xs)
        pk = np.stack([predict(m, xs) for m in models])
        p1[i:i + n] = np.where(fit_row[i:i + n], pk[fold[i:i + n], np.arange(n)], pk.mean(axis=0))
        i += n
    ctx = score_context(s1, s23, p1)
    del models                                          # frees their GPU prediction buffers before pass 2 (4 GB card)
    gc.collect()
    del p1

    # ---- pass 2: base + competition context on the pass-1 score ----
    rows2 = subsample(np.flatnonzero(fit_row), label, pre_rank, rng)
    mask, pos = masked(len(s1), rows2, val)
    _, _, x2 = load_matrix(paths, BASE_FEATURES, mask, extra=len(SCORE_CONTEXT))   # context columns filled in place
    x2[:, len(BASE_FEATURES):] = ctx[mask]
    del ctx
    features2 = BASE_FEATURES + SCORE_CONTEXT
    rows2_ps = subsample(ps.rows, ps.label, ps.pre, rng) if ps else []
    m2 = fit(*with_pseudo(x2[pos[rows2]], label[rows2], rows2_ps, ps_ctx), x2[pos[cal]], label[cal], 'pass2')

    # ---- isotonic calibration on CAL, report on TUNE ----
    raw = predict(m2, x2[pos[val]])
    is_cal = role[val] == CAL
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(raw[is_cal], label[val][is_cal])
    p = iso.predict(raw).astype(np.float32)
    tune = ~is_cal
    if label[val][tune].any() and not label[val][tune].all():
        print(f'[s4] tune split: AUC {roc_auc_score(label[val][tune], p[tune]):.4f}, '
              f'AP {average_precision_score(label[val][tune], p[tune]):.4f}')

    with open(MODEL_DIR / 'isotonic.pkl', 'wb') as f:
        pickle.dump(iso, f)
    (MODEL_DIR / 'meta.json').write_text(json.dumps({'pass1': ['pass1_f0.ubj', 'pass1_f1.ubj'], 'pass2': 'pass2.ubj',
                                                     'features_pass1': BASE_FEATURES, 'features_pass2': features2}, indent=1))
    gain = m2.get_score(importance_type='total_gain')
    imp = pd.DataFrame({'feature': features2, 'gain': [gain.get(f'f{i}', 0.0) for i in range(len(features2))]})
    imp = imp.sort_values('gain', ascending=False)
    imp.to_csv(MODEL_DIR / 'importance.csv', index=False)
    print('[s4] top features by gain:', ', '.join(imp.feature.head(12)))
    pq.write_table(pa.table({'s1': s1[val], 's23': s23[val], 'p': p, 'label': label[val], 'role': role[val]}),
                   out / 'valid_scores.parquet')
