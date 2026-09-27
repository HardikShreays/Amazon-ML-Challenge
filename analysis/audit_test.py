"""Qualitative audit of the test submission across countries (read-only; produces no labels).

Adapted from a teammate's polars audit for the v3 pipeline to this repo's files. Slices where an
inconsistency would show if present; prints slice sizes (by country) and a few examples each.

    python analysis/audit_test.py [N]          # N examples x 3 per slice (default 5)

Scores: the submission came from src/fast_submit.py (pass-1 models, first 100 trees), so `p` here is
that same score, recomputed once and cached in work/test/audit_pairs.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'business_entity_resolution'))
from src import config  # noqa: E402
from src.fast_submit import predict  # noqa: E402
from src.s3_features import BASE_FEATURES, load_matrix, shards  # noqa: E402
from src.s4_score import load_models  # noqa: E402

T = config.WORK_DIR / 'test'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TREES = 100
TAU = 0.65
KEEP = ['name_tset', 'core_tset', 'lead_eq', 'addr_tset', 'num_tset', 'sfx_compat', 'b_addr_empty', 'pre_rank']


def pair_table():
    """(s1, s23, p, KEEP features) for every test candidate, cached after the first run."""
    cache = T / 'audit_pairs.parquet'
    if cache.exists():
        return pq.read_table(cache).to_pandas()
    models, parts = load_models(), []
    for path in shards('test'):
        s1, s23, x = load_matrix([path], BASE_FEATURES)
        d = {'s1': s1, 's23': s23, 'p': predict(models, x, TREES)}
        d.update({c: x[:, BASE_FEATURES.index(c)] for c in KEEP})
        parts.append(pa.table(d))
    t = pa.concat_tables(parts)
    pq.write_table(t, cache)
    return t.to_pandas()


def lead(nums):
    """First number of the folded address as int (-1 when absent)."""
    return np.array([int(n.split()[0]) if n else -1 for n in nums], np.int64)


print('[audit] loading ...', flush=True)
s1_raw = pq.read_table(T / 's1_raw.parquet')
s23_raw = pq.read_table(T / 's23_raw.parquet', columns=['entity_id', 'business_name', 'business_address', 'src'])
s1_sig = pq.read_table(T / 's1_sig.parquet', columns=['name_core', 'sfx', 'nums', 'addr_empty']).to_pandas()
s23_sig = pq.read_table(T / 's23_sig.parquet', columns=['name_core', 'sfx', 'nums', 'addr_empty'])
country = s1_raw['country'].to_numpy(zero_copy_only=False)
pairs = pair_table()

m = pq.read_table(T / 'matches.parquet').to_pandas().merge(pairs, on=['s1', 's23'], how='left')
m['country'] = country[m.s1.to_numpy()]
sub = s23_sig.take(pa.array(m.s23.to_numpy())).to_pandas()
m['cq'], m['r_sfx'], m['r_pno'] = sub.name_core.to_numpy(), sub.sfx.to_numpy(), lead(sub.nums)
m['cs'], m['s_sfx'] = s1_sig.name_core.to_numpy()[m.s1], s1_sig.sfx.to_numpy()[m.s1]
m['s_pno'] = lead(s1_sig.nums.to_numpy()[m.s1])
m['addr_missing'] = ((s1_sig.addr_empty.to_numpy()[m.s1] == 1) | (m.b_addr_empty == 1)).astype(int)
m['pno_abs'] = np.where((m.r_pno >= 0) & (m.s_pno >= 0), (m.r_pno - m.s_pno).abs(), -1)
print(f'[audit] {len(m):,} matched pairs for {len(np.unique(m.s1)):,} S1s; {len(pairs):,} candidate pairs', flush=True)


def texts(df):
    """Attach raw S1 / record strings (only for the rows that will be printed)."""
    a = s1_raw.take(pa.array(df.s1.to_numpy())).to_pandas()
    b = s23_raw.take(pa.array(df.s23.to_numpy())).to_pandas()
    return df.assign(s_name=a.business_name.to_numpy(), s_addr=a.business_address.to_numpy(),
                     r_name=b.business_name.to_numpy(), r_addr=b.business_address.to_numpy(),
                     r_src=b.src.to_numpy())


COLS = ['country', 's_name', 's_addr', 'r_name', 'r_addr', 'p']


def show(title, df, cols=COLS, sort=None):
    by = df.groupby('country').size().to_dict() if len(df) else {}
    print(f'\n### {title}: {len(df):,} pairs | by country {by}')
    if len(df):
        ex = df.sample(min(N * 3, len(df)), random_state=1)
        ex = texts(ex.sort_values(sort) if sort else ex)
        with pd.option_context('display.max_colwidth', 44, 'display.width', 260, 'display.max_columns', 20):
            print(ex[cols].to_string(index=False))


# A: matched records of one S1 whose house number differs while another matched record has the same number
a = m[m.addr_missing == 0].copy()
a['_s1_has_eq'] = a.groupby('s1').lead_eq.transform('max') == 1
show('A  matched with a DIFFERENT house number although the S1 also has same-number matches (possible near-miss)',
     a[a._s1_has_eq & (a.lead_eq == 0) & (a.pno_abs > 0) & (a.pno_abs <= 50)])
# B: little name overlap AND different number
show('B  low name similarity (<40) AND different house number',
     m[(m.name_tset < 40) & (m.lead_eq == 0) & (m.addr_missing == 0)])
# C: large clusters (MAX_MATCHES caps at 8)
k = m.groupby('s1').size()
show(f'C  S1s with >= {config.MAX_MATCHES} matches (listing their pairs)', m[m.s1.isin(k[k >= config.MAX_MATCHES].index)],
     sort=['s1'])
# D: empty S1s whose best candidate is just under the threshold
best = pairs.sort_values('p', ascending=False).drop_duplicates('s1')
empty = best[~best.s1.isin(m.s1)].copy()
empty['country'] = country[empty.s1.to_numpy()]
show(f'D  S1 predicted EMPTY but best candidate p in [0.5, {TAU})', empty[empty.p >= 0.5])
print(f'    (S1s with no candidate at all: {len(country) - pairs.s1.nunique():,})')
# E: sibling distractors: same first core token, record's extra core words are real S1-vocabulary words
vocab =pd.DataFrame({'country': country, 'w': s1_sig.name_core.str.split().map(lambda t: list(set(t)))}) \
          .explode('w').groupby(['country', 'w']).size().rename('vf')
cq, cs = m.cq.str.split(), m.cs.str.split()
first_eq = np.array([bool(x) and bool(y) and x[0] == y[0] for x, y in zip(cq, cs)])
xq = [sorted(set(x) - set(y)) for x, y in zip(cq, cs)]
e = m.assign(xq=xq)[first_eq & np.array([len(x) > 0 for x in xq])]
ev = e[['country', 'xq']].explode('xq').reset_index()
ev['vf'] = vocab.reindex(pd.MultiIndex.from_arrays([ev.country, ev.xq])).fillna(0).to_numpy()
real = ev.groupby('index').vf.apply(lambda v: (v >= 3).all())
show("E  matched although record's extra words are REAL S1-vocabulary words (sibling-distractor pattern)",
     e.loc[real[real].index])
# F: address-missing matches
show('F  address-missing records matched on name only', m[m.addr_missing == 1])
# G: same core name, conflicting legal/generic suffix (README risk: France 'Club' vs 'Centre')
show('G  same core name but CONFLICTING suffix sets (sfx_compat == 3)', m[(m.cq == m.cs) & (m.sfx_compat == 3)])
# H: satellite records claimed by more than one S1 (never happens in the training ground truth)
own = m.groupby('s23').s1.transform('size')
show('H  satellite record matched to >= 2 different S1s', m[own >= 2], sort=['s23'])
# R: plain random France matches and France empties, for eyeballing
show('R1 random France matches', m[m.country == 'France'])
fe = empty[empty.country == 'France']
show('R2 random France S1s predicted EMPTY (with their best candidate)', fe)
