"""Labelled counterpart of audit_test.py: the same slices on the TRAIN tune split, where the ground
truth is known, so each slice gets a measured precision instead of an eyeball verdict.

    python analysis/audit_slices_labelled.py

Uses the same scorer and decision as the fast submission (pass-1 first 100 trees, threshold 0.65).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'business_entity_resolution'))
from src import config  # noqa: E402
from src.fast_submit import predict  # noqa: E402
from src.s2_diagnostics import pair_codes  # noqa: E402
from src.s3_features import BASE_FEATURES, load_matrix, shards  # noqa: E402
from src.s4_score import load_models  # noqa: E402
from src.s4_train import TUNE, entity_roles  # noqa: E402
from src.s6_decide import decide  # noqa: E402

W = config.WORK_DIR / 'train'
TREES, TAU = 100, 0.65
KEEP = ['name_tset', 'lead_eq', 'sfx_compat', 'b_addr_empty', 'b_non_latin', 'acronym']

blocked = np.load(W / 'entities.npy')
role, _ = entity_roles(blocked)
entities = blocked[role[blocked] == TUNE]
s1, s23, x = load_matrix(shards('train'), BASE_FEATURES)
keep = role[s1] == TUNE
df = pd.DataFrame({'s1': s1[keep], 's23': s23[keep], 'p': predict(load_models(), x[keep], TREES)})
for c in KEEP:
    df[c] = x[keep, BASE_FEATURES.index(c)]
del x
gt = pq.read_table(W / 'gt_pairs.parquet').to_pandas()
gt = gt[np.isin(gt.s1, entities)]
df['label'] = np.isin(pair_codes(df.s1, df.s23), pair_codes(gt.s1, gt.s23))

sig1 = pq.read_table(W / 's1_sig.parquet', columns=['name_core', 'sfx', 'nums', 'addr_empty', 'country']).to_pandas()
sig23 = pq.read_table(W / 's23_sig.parquet', columns=['name_core', 'nums']).to_pandas()
df['country'] = sig1.country.to_numpy()[df.s1]
df['cs'], df['cq'] = sig1.name_core.to_numpy()[df.s1], sig23.name_core.to_numpy()[df.s23]
lead = lambda v: np.array([int(n.split()[0]) if n else -1 for n in v], np.int64)  # noqa: E731
df['s_pno'], df['r_pno'] = lead(sig1.nums.to_numpy()[df.s1]), lead(sig23.nums.to_numpy()[df.s23])
df['addr_missing'] = ((sig1.addr_empty.to_numpy()[df.s1] == 1) | (df.b_addr_empty == 1)).astype(int)
df['pno_abs'] = np.where((df.r_pno >= 0) & (df.s_pno >= 0), (df.r_pno - df.s_pno).abs(), -1)

m = df[decide(df.s1, df.s23, df.p, TAU, 'threshold')].copy()
n_true = gt.groupby('s1').size()
print(f'tune: {len(entities):,} entities, {len(m):,} matched pairs, overall precision {m.label.mean():.4f}, '
      f'recall {m.label.sum() / len(gt):.4f}')


def report(name, sl):
    print(f'{name:<62} {len(sl):>9,} pairs  precision {sl.label.mean() if len(sl) else float("nan"):.4f}  '
          f'| {sl.groupby("country").label.agg(["size", "mean"]).round(3).to_dict("index") if len(sl) else ""}')


a = m[m.addr_missing == 0].copy()
a['_eq'] = a.groupby('s1').lead_eq.transform('max') == 1
report('A  diff house no. (<=50) while S1 has a same-number match', a[a._eq & (a.lead_eq == 0) & (a.pno_abs > 0) & (a.pno_abs <= 50)])
report('B  name_tset < 40 AND different house number', m[(m.name_tset < 40) & (m.lead_eq == 0) & (m.addr_missing == 0)])
report('B2 name_tset < 40 (any address)', m[m.name_tset < 40])
k = m.groupby('s1').size()
big = m[m.s1.isin(k[k >= config.MAX_MATCHES].index)]
report(f'C  S1s with >= {config.MAX_MATCHES} matches', big)
print(f'   ... those S1s have {n_true.reindex(big.s1.unique()).fillna(0).mean():.1f} true matches on average (cap {config.MAX_MATCHES})')
report('F  address missing on either side', m[m.addr_missing == 1])
report('G  same core name, conflicting suffix (sfx_compat == 3)', m[(m.cq == m.cs) & (m.sfx_compat == 3)])
own = m.groupby('s23').s1.transform('size')
report('H  satellite matched to >= 2 S1s', m[own >= 2])
report('   non-Latin record matched', m[m.b_non_latin == 1])
report('   acronym match', m[m.acronym == 1])

best = df.sort_values('p', ascending=False).drop_duplicates('s1')
empty = best[~best.s1.isin(m.s1)]
band = empty[(empty.p >= 0.5) & (empty.p < TAU)]
print(f'\nD  S1 predicted EMPTY, best candidate p in [0.5, {TAU}): {len(band):,} S1s; best candidate is a TRUE match '
      f'for {band.label.mean():.3f}; entity truly has matches: {band.s1.isin(gt.s1).mean():.3f}')
for lo, hi in [(0.3, 0.5), (0.5, 0.6), (0.6, 0.65)]:
    b = empty[(empty.p >= lo) & (empty.p < hi)]
    print(f'   p in [{lo}, {hi}): {len(b):,} S1s, best candidate true {b.label.mean():.3f}, '
          f'non-Latin {b.b_non_latin.mean():.2f}, acronym {b.acronym.mean():.2f}')
print(f'   empty S1s that truly have matches: {empty.s1.isin(gt.s1).mean():.3f} of {len(empty):,} empty-with-candidates')
