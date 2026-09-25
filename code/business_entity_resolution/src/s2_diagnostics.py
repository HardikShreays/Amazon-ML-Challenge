"""Stage 2b — blocking diagnostics (train split only, needs ground truth).

Pair completeness is the recall CEILING of the whole pipeline: no model can recover a true pair that
blocking never produced. Target from the plan: >= 0.98 at <= 40 candidates per entity.

    pair completeness = true pairs kept / true pairs of the blocked entities
    reduction ratio   = 1 - candidates / (entities x satellites)
    per-key recall    = share of true pairs each key produced (and produced *alone*)
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .s2_blocking import KEY_NAMES


def pair_codes(s1, s23):
    """One int64 per (s1, s23) pair, for fast set membership with np.isin."""
    return (np.asarray(s1, np.int64) << 32) | np.asarray(s23, np.int64)


def report(split='train'):
    out = config.WORK_DIR / split
    cand = pq.read_table(out / 'candidates.parquet', columns=['s1', 's23', 'kbits']).to_pandas()
    entities = np.load(out / 'entities.npy')
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    gt = gt[np.isin(gt.s1, entities)]
    n_s23 = pq.ParquetFile(out / 's23_raw.parquet').metadata.num_rows
    country = pq.read_table(out / 's1_sig.parquet', columns=['country'])['country'].to_numpy(zero_copy_only=False)

    cand_code = pair_codes(cand.s1, cand.s23)
    order = np.argsort(cand_code)
    pos = np.searchsorted(cand_code, pair_codes(gt.s1, gt.s23), sorter=order)
    pos = order[np.minimum(pos, len(order) - 1)]
    found = cand_code[pos] == pair_codes(gt.s1, gt.s23)
    bits = np.where(found, cand.kbits.to_numpy()[pos], 0)

    per_entity = np.bincount(np.searchsorted(entities, cand.s1), minlength=len(entities))
    stats = {
        'entities': int(len(entities)),
        'candidates': int(len(cand)),
        'true_pairs': int(len(gt)),
        'pair_completeness': float(found.mean()) if len(gt) else float('nan'),
        'reduction_ratio': 1 - len(cand) / (len(entities) * n_s23),
        'cands_per_entity': {'mean': float(per_entity.mean()), 'p95': float(np.percentile(per_entity, 95)),
                             'max': int(per_entity.max()), 'zero': float((per_entity == 0).mean())},
        'positive_rate': float(found.sum() / max(len(cand), 1)),
        'per_key_recall': {name: float(((bits & b) > 0).mean()) for b, name in KEY_NAMES.items()},
        'per_key_only': {name: float((bits == b).mean()) for b, name in KEY_NAMES.items()},
        'pair_completeness_by_country': {c: float(found[country[gt.s1] == c].mean())
                                         for c in pd.unique(country[gt.s1])},
    }
    (out / 'blocking_diagnostics.json').write_text(json.dumps(stats, indent=2))
    print(f"[s2] pair completeness {stats['pair_completeness']:.4f} | "
          f"{stats['cands_per_entity']['mean']:.1f} candidates/entity (p95 {stats['cands_per_entity']['p95']:.0f}) | "
          f"reduction ratio {stats['reduction_ratio']:.6f}")
    print('[s2] per-key recall:', {k: round(v, 3) for k, v in stats['per_key_recall'].items()})
    print('[s2] found ONLY by key:', {k: round(v, 4) for k, v in stats['per_key_only'].items()})
    return stats
