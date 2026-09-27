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
model never trained or calibrated on. Ties go to 'partition': it only removes guaranteed false
positives, so it can never hurt, yet it never fires on TUNE (no satellite has two owners there).
A second threshold `tau_noaddr` (>= tau) then applies to pairs where either side has an empty
address; those name-only matches are the least precise slice (audit: ~0.75 vs 0.98 overall).

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
PREFER = {'partition': 0, 'threshold': 1, 'full': 2}     # tie-break order between equal scores
TIE_DECIMALS = 4                # scores equal to 4 decimals (~ a handful of entities of 110k) are ties


def _owner_mask(s23, p):
    """True for the single highest-probability claimant of every satellite record."""
    order = np.argsort(-p, kind='stable')
    first = np.unique(s23[order], return_index=True)[1]
    mask = np.zeros(len(p), bool)
    mask[order[first]] = True
    return mask


def noaddr_mask(split, s1, s23):
    """True for pairs where the S1 or the satellite address is empty."""
    w = config.WORK_DIR / split
    a = pq.read_table(w / 's1_sig.parquet', columns=['addr_empty'])['addr_empty'].to_numpy()
    b = pq.read_table(w / 's23_sig.parquet', columns=['addr_empty'])['addr_empty'].to_numpy()
    return (a[np.asarray(s1)] == 1) | (b[np.asarray(s23)] == 1)


def decide(s1, s23, p, tau, mode, noaddr=None, tau_noaddr=None):
    """Boolean mask over the pairs that are emitted as matches. With `noaddr` (bool per pair) and
    `tau_noaddr`, pairs lacking an address must reach the stricter `tau_noaddr` instead of `tau`."""
    s1, s23, p = np.asarray(s1), np.asarray(s23), np.asarray(p, np.float64)
    tau = np.full(len(p), tau) if noaddr is None or tau_noaddr is None else np.where(noaddr, tau_noaddr, tau)
    live = _owner_mask(s23, p) if mode != 'threshold' else np.ones(len(p), bool)

    idx = np.flatnonzero(live)
    order = idx[np.lexsort((-p[idx], s1[idx]))]            # per entity, descending probability
    g, v = s1[order], p[order]
    if not len(g):
        return np.zeros(len(p), bool)
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    k = np.arange(len(g)) - np.repeat(starts, sizes) + 1    # 1-based position in the entity's list
    allowed = (v >= tau[order]) & (k <= config.MAX_MATCHES)
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


def tune_entities():
    """(TUNE entity indices, their ground-truth pairs) from the train split."""
    out = config.WORK_DIR / 'train'
    blocked = np.load(out / 'entities.npy')
    role, _ = entity_roles(blocked)
    entities = blocked[role[blocked] == TUNE]               # includes entities with no candidates at all
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    return entities, gt[np.isin(gt.s1, entities)]


def search(s1, s23, p, noaddr, entities, gt):
    """Grid-search (mode, tau), then tau_noaddr >= tau, by macro-F0.5 -> (sweep table, best dict)."""
    s1, s23 = pd.Series(np.asarray(s1)), pd.Series(np.asarray(s23))

    def score(mask):
        """Macro F0.5 of the pairs selected by `mask` against the tune ground truth."""
        return macro_f05(s1[mask], s23[mask], gt.s1, gt.s23, entities)

    rows = []
    for mode in MODES:
        for tau in config.TAU_GRID:
            m = decide(s1, s23, p, tau, mode)
            rows.append({'mode': mode, 'tau': tau, 'tau_noaddr': tau, 'macro_f05': score(m),
                         'matches_per_entity': m.sum() / max(len(entities), 1)})
    table = pd.DataFrame(rows)
    ranked = table.assign(pref=table['mode'].map(PREFER), key=table['macro_f05'].round(TIE_DECIMALS))
    ranked = ranked.sort_values(['key', 'pref', 'macro_f05'], ascending=[False, True, False])
    best = {k: v for k, v in ranked.iloc[0].to_dict().items() if k not in ('pref', 'key')}
    for tn in sorted({t for t in config.TAU_GRID + [0.97, 0.99] if t > best['tau']}):
        m = decide(s1, s23, p, best['tau'], best['mode'], noaddr, tn)
        f = score(m)
        table.loc[len(table)] = {'mode': best['mode'], 'tau': best['tau'], 'tau_noaddr': tn, 'macro_f05': f,
                                 'matches_per_entity': m.sum() / max(len(entities), 1)}
        if f > best['macro_f05']:
            best.update(tau_noaddr=tn, macro_f05=f)
    return table, best


def tune():
    """Grid-search the decision layer on the TUNE entities and store the winner."""
    out = config.WORK_DIR / 'train'
    vs = pq.read_table(out / 'valid_scores.parquet').to_pandas()
    vs = vs[vs.role == TUNE].reset_index(drop=True)
    entities, gt = tune_entities()
    table, best = search(vs.s1, vs.s23, vs.p, noaddr_mask('train', vs.s1, vs.s23), entities, gt)
    oracle = macro_f05(vs.s1[vs.label], vs.s23[vs.label], gt.s1, gt.s23, entities)   # perfect model on our candidates
    base = table[table.tau_noaddr == table.tau]
    summary = {'mode': best['mode'], 'tau': float(best['tau']), 'tau_noaddr': float(best['tau_noaddr']),
               'tune_macro_f05': float(best['macro_f05']),
               'blocking_ceiling_f05': oracle, 'tune_entities': int(len(entities)),
               'best_per_mode': {md: float(base[base['mode'] == md].macro_f05.max()) for md in MODES}}
    (MODEL_DIR / 'decision.json').write_text(json.dumps(summary, indent=2))
    table.to_csv(MODEL_DIR / 'tau_sweep.csv', index=False)
    print(f"[s6] best per mode on TUNE: { {k: round(v, 4) for k, v in summary['best_per_mode'].items()} }")
    print(f"[s6] chosen: mode={summary['mode']} tau={summary['tau']} tau_noaddr={summary['tau_noaddr']} -> "
          f"macro F0.5 {summary['tune_macro_f05']:.4f} (ceiling with perfect matcher on these candidates: {oracle:.4f})")
    return summary


def apply(split):
    """Apply the tuned decision layer to a split's scores -> matches.parquet."""
    d = json.loads((MODEL_DIR / 'decision.json').read_text())
    sc = pq.read_table(config.WORK_DIR / split / 'scores.parquet').to_pandas()
    noaddr = noaddr_mask(split, sc.s1, sc.s23) if 'tau_noaddr' in d else None
    m = decide(sc.s1, sc.s23, sc.p, d['tau'], d['mode'], noaddr, d.get('tau_noaddr'))
    pq.write_table(pa.table({'s1': sc.s1.to_numpy()[m], 's23': sc.s23.to_numpy()[m]}),
                   config.WORK_DIR / split / 'matches.parquet')
    n_s1 = pq.ParquetFile(config.WORK_DIR / split / 's1_raw.parquet').metadata.num_rows
    print(f"[s6] {split}: {m.sum():,} matches for {n_s1:,} entities ({m.sum() / n_s1:.2f} per entity, "
          f"mode={d['mode']}, tau={d['tau']}, tau_noaddr={d.get('tau_noaddr')})")


if __name__ == '__main__':
    # tiny self-check: satellite 9 is claimed by entities 0 and 1 -> only entity 0 (higher p) keeps it;
    # entity 2 has one weak candidate -> expected-F0.5 prefers predicting "no match"
    s1, s23, p = [0, 0, 1, 1, 2], [9, 8, 9, 7, 6], [0.95, 0.9, 0.6, 0.8, 0.3]
    assert decide(s1, s23, p, 0.5, 'threshold').tolist() == [True, True, True, True, False]
    assert decide(s1, s23, p, 0.5, 'partition').tolist() == [True, True, False, True, False]
    assert decide(s1, s23, p, 0.1, 'full').tolist() == [True, True, False, True, False]
    print('s6_decide self-check OK')
