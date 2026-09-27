"""Stage 8c — merge the cross-encoder into the decision layer and write the submission (runs locally).

Inside the grey zone the pair probability is replaced by one of:
    gbdt    the v5 GBDT probability (baseline, for reference)
    ce      the cross-encoder alone
    stack   logistic regression on [logit p_gbdt, logit_ce, address-missing flag]
Each is evaluated on the TUNE entities (stack is 2-fold cross-fitted by entity so the number is
honest), then (mode, tau, tau_noaddr) is grid-searched exactly as in stage 6. The best variant is
refitted on all of TUNE and applied to test -> OUTPUT_DIR/{matching_results,candidate_pairs}.tsv.

Run: ER_WORK_DIR=<work_v5> ER_OUTPUT_DIR=<submissions/v6_ce> python -m src.s8_ce_merge
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from . import config
from .s4_train import TUNE
from .s6_decide import decide, noaddr_mask, search, tune_entities
from .s7_output import validate, write_id_lists
from .s8_ce_export import OUT, ZONE


def logit(p):
    """Log-odds of a probability, clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def features(p, lc, noaddr):
    """Stacker inputs: [logit GBDT probability, cross-encoder logit, address-missing flag]."""
    return np.c_[logit(p), lc, noaddr.astype(float)]


def zone_scores(base, ce_path):
    """Grey-zone mask over `base` plus the cross-encoder logits aligned to it."""
    z = ((base.p >= ZONE[0]) & (base.p <= ZONE[1])).to_numpy()
    ce = pd.read_parquet(ce_path)
    assert len(ce) == z.sum() and (ce.s1.to_numpy() == base.s1.to_numpy()[z]).all() \
        and (ce.s23.to_numpy() == base.s23.to_numpy()[z]).all(), f'{ce_path} does not line up with the grey zone'
    return z, ce.logit_ce.to_numpy(np.float64)


def main():
    """Pick gbdt / ce / stack on TUNE, re-tune the decision layer, apply it to test and write + validate the submission."""
    vs = pq.read_table(config.WORK_DIR / 'train' / 'valid_scores.parquet').to_pandas()
    vs = vs[vs.role == TUNE].reset_index(drop=True)
    na = noaddr_mask('train', vs.s1, vs.s23)
    ents, gt = tune_entities()
    z, lc = zone_scores(vs, OUT / 'tune_ce.parquet')
    pz, yz, naz, s1z = vs.p.to_numpy()[z], vs.label.to_numpy()[z], na[z], vs.s1.to_numpy()[z]

    fold = np.random.default_rng(7).random(int(vs.s1.max()) + 1)[s1z] < 0.5
    stack = np.empty(z.sum())
    for f in (True, False):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(features(pz, lc, naz)[fold != f], yz[fold != f])
        stack[fold == f] = lr.predict_proba(features(pz, lc, naz)[fold == f])[:, 1]
    variants = {'gbdt': pz, 'ce': 1 / (1 + np.exp(-lc)), 'stack': stack}

    acc = {k: ((v >= 0.5) == yz).mean() for k, v in variants.items()}
    print(f'[s8] grey-zone pair accuracy on TUNE: { {k: round(v, 4) for k, v in acc.items()} }', flush=True)
    results = {}
    for name, v in variants.items():
        p = vs.p.to_numpy(np.float64).copy()
        p[z] = v
        _, best = search(vs.s1, vs.s23, p, na, ents, gt)
        results[name] = best
        print(f"[s8] {name:6s} macro F0.5 {best['macro_f05']:.4f}  mode={best['mode']} tau={best['tau']} "
              f"tau_noaddr={best['tau_noaddr']}", flush=True)
    pick = max(results, key=lambda k: results[k]['macro_f05'])
    best = results[pick]
    print(f'[s8] using {pick}', flush=True)

    # ---- test ----
    sc = pq.read_table(config.WORK_DIR / 'test' / 'scores.parquet').to_pandas()
    zt, lct = zone_scores(sc, OUT / 'test_ce.parquet')
    nat = noaddr_mask('test', sc.s1, sc.s23)
    p = sc.p.to_numpy(np.float64)
    if pick == 'ce':
        p[zt] = 1 / (1 + np.exp(-lct))
    elif pick == 'stack':
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(features(pz, lc, naz), yz)
        p[zt] = lr.predict_proba(features(p[zt], lct, nat[zt]))[:, 1]
        print(f'[s8] stacker coefficients {lr.coef_.round(3).tolist()} intercept {lr.intercept_.round(3).tolist()}')
    m = decide(sc.s1, sc.s23, p, best['tau'], best['mode'], nat, best['tau_noaddr'])
    s1, s23 = sc.s1.to_numpy()[m], sc.s23.to_numpy()[m]
    n_s1 = pq.ParquetFile(config.WORK_DIR / 'test' / 's1_raw.parquet').metadata.num_rows
    print(f'[s8] test: {m.sum():,} matches ({m.sum() / n_s1:.2f} per S1)', flush=True)

    w = config.WORK_DIR / 'test'
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_ids = pq.read_table(w / 's1_raw.parquet', columns=['entity_id'])['entity_id'].combine_chunks()
    s23_ids = pq.read_table(w / 's23_raw.parquet', columns=['entity_id'])['entity_id'].combine_chunks()
    write_id_lists(config.OUTPUT_DIR / 'candidate_pairs.tsv', 'candidate_entity_ids', s1_ids, s23_ids,
                   sc.s1.to_numpy(), sc.s23.to_numpy())
    write_id_lists(config.OUTPUT_DIR / 'matching_results.tsv', 'matched_entity_ids', s1_ids, s23_ids, s1, s23)
    (config.OUTPUT_DIR / 'ce_decision.json').write_text(json.dumps(
        {'variant': pick, **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best.items()},
         'all': {k: float(r['macro_f05']) for k, r in results.items()}, 'grey_accuracy': acc}, indent=2, default=float))
    validate()


if __name__ == '__main__':
    main()
