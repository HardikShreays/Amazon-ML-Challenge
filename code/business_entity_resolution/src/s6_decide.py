"""Stage 6 — global decision layer: calibrated pair probabilities -> final match lists.

Three mechanisms (the plan's 6a-6c), compared against a plain threshold on the TUNE split:

  threshold   accept p >= tau (capped at MAX_MATCHES per entity)                  <- baseline
  partition   + each satellite goes to at most ONE S1: its highest-probability claimant.
              EDA: 0 of 7.6M training satellites has two owners, so every second claim is a
              guaranteed false positive.
  full        + per-entity expected-F0.5 cut. Survivors of one entity sorted p1 >= p2 >= ...:
                  E[F0.5(k)] ~ 1.25 * sum(p_1..p_k) / (0.25 * k + sum(p_all))     k >= 1
                  E[F0.5(0)] = prod(1 - p_i)       (probability the entity is a true singleton)
              choose the argmax k. k = 0 is how singletons earn their 1.0.

`tau` (and the mode) are picked by direct macro-F0.5 grid search on the TUNE entities, which the
model never trained or calibrated on.

train: writes work/models/decision.json      test: writes work/test/matches.parquet (s1, s23)
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .evaluate import macro_f05
from .s4_score import MODEL_DIR
from .s4_train import TUNE, entity_roles

MODES = ('threshold', 'partition', 'full')


def _owner_mask(s23, p):
    """True for the single highest-probability claimant of every satellite record."""
    order = np.argsort(-p, kind='stable')
    first = np.unique(s23[order], return_index=True)[1]
    mask = np.zeros(len(p), bool)
    mask[order[first]] = True
    return mask


def decide(s1, s23, p, tau, mode):
    """Boolean mask over the pairs that are emitted as matches."""
    s1, s23, p = np.asarray(s1), np.asarray(s23), np.asarray(p, np.float64)
    live = _owner_mask(s23, p) if mode != 'threshold' else np.ones(len(p), bool)

    idx = np.flatnonzero(live)
    order = idx[np.lexsort((-p[idx], s1[idx]))]            # per entity, descending probability
    g, v = s1[order], p[order]
    if not len(g):
        return np.zeros(len(p), bool)
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    k = np.arange(len(g)) - np.repeat(starts, sizes) + 1    # 1-based position in the entity's list
    allowed = (v >= tau) & (k <= config.MAX_MATCHES)
    if mode != 'full':
        accept = allowed
    else:
        cum = np.cumsum(v)
        cum_k = cum - np.repeat(cum[starts] - v[starts], sizes)          # prefix sums within entity
        total = np.repeat(np.add.reduceat(v, starts), sizes)              # expected true cluster size
        ef = np.where(allowed, 1.25 * cum_k / (0.25 * k + total), -1.0)
        ef0 = np.exp(np.add.reduceat(np.log1p(-np.minimum(v, 1 - 1e-9)), starts))  # P(no true match)
        best = np.maximum.reduceat(ef, starts)
        # smallest k reaching the best expected F0.5; k* = 0 when "no match" is expected to score higher
        k_star = np.minimum.reduceat(np.where(ef == np.repeat(best, sizes), k, 10 ** 9), starts)
        k_star = np.where(ef0 >= best, 0, k_star)
        accept = allowed & (k <= np.repeat(k_star, sizes))
    mask = np.zeros(len(p), bool)
    mask[order[accept]] = True
    return mask


def tune():
    """Grid-search (mode, tau) on the TUNE entities and store the winner."""
    out = config.WORK_DIR / 'train'
    vs = pq.read_table(out / 'valid_scores.parquet').to_pandas()
    vs = vs[vs.role == TUNE]
    blocked = np.load(out / 'entities.npy')
    role, _ = entity_roles(blocked)
    entities = blocked[role[blocked] == TUNE]               # includes entities with no candidates at all
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    gt = gt[np.isin(gt.s1, entities)]

    rows = []
    for mode in MODES:
        for tau in config.TAU_GRID:
            m = decide(vs.s1, vs.s23, vs.p, tau, mode)
            rows.append({'mode': mode, 'tau': tau, 'macro_f05': macro_f05(vs.s1[m], vs.s23[m], gt.s1, gt.s23, entities),
                         'matches_per_entity': m.sum() / max(len(entities), 1)})
    table = pd.DataFrame(rows)
    best = table.loc[table.macro_f05.idxmax()]
    oracle = macro_f05(vs.s1[vs.label], vs.s23[vs.label], gt.s1, gt.s23, entities)   # perfect model on our candidates
    summary = {'mode': best['mode'], 'tau': float(best['tau']), 'tune_macro_f05': float(best['macro_f05']),
               'blocking_ceiling_f05': oracle, 'tune_entities': int(len(entities)),
               'best_per_mode': {md: float(table[table['mode'] == md].macro_f05.max()) for md in MODES}}
    (MODEL_DIR / 'decision.json').write_text(json.dumps(summary, indent=2))
    table.to_csv(MODEL_DIR / 'tau_sweep.csv', index=False)
    print(f"[s6] best per mode on TUNE: { {k: round(v, 4) for k, v in summary['best_per_mode'].items()} }")
    print(f"[s6] chosen: mode={summary['mode']} tau={summary['tau']} -> macro F0.5 {summary['tune_macro_f05']:.4f} "
          f"(ceiling with perfect matcher on these candidates: {oracle:.4f})")
    return summary


def apply(split):
    """Apply the tuned decision layer to a split's scores -> matches.parquet."""
    d = json.loads((MODEL_DIR / 'decision.json').read_text())
    sc = pq.read_table(config.WORK_DIR / split / 'scores.parquet').to_pandas()
    m = decide(sc.s1, sc.s23, sc.p, d['tau'], d['mode'])
    pq.write_table(pa.table({'s1': sc.s1.to_numpy()[m], 's23': sc.s23.to_numpy()[m]}),
                   config.WORK_DIR / split / 'matches.parquet')
    n_s1 = pq.ParquetFile(config.WORK_DIR / split / 's1_raw.parquet').metadata.num_rows
    print(f"[s6] {split}: {m.sum():,} matches for {n_s1:,} entities ({m.sum() / n_s1:.2f} per entity, "
          f"mode={d['mode']}, tau={d['tau']})")


if __name__ == '__main__':
    # tiny self-check: satellite 9 is claimed by entities 0 and 1 -> only entity 0 (higher p) keeps it;
    # entity 2 has one weak candidate -> expected-F0.5 prefers predicting "no match"
    s1, s23, p = [0, 0, 1, 1, 2], [9, 8, 9, 7, 6], [0.95, 0.9, 0.6, 0.8, 0.3]
    assert decide(s1, s23, p, 0.5, 'threshold').tolist() == [True, True, True, True, False]
    assert decide(s1, s23, p, 0.5, 'partition').tolist() == [True, True, False, True, False]
    assert decide(s1, s23, p, 0.1, 'full').tolist() == [True, True, False, True, False]
    print('s6_decide self-check OK')
