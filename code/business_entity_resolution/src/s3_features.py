"""Stage 3 — pairwise features for every candidate pair (~50, all float32).

All features are symmetric in spirit, language-agnostic and country-free (blocking already
restricts pairs to one country, so country equality would be a constant).

A. name      rapidfuzz ratios on canonical / core / squash forms, squash common-prefix (the
             domain-name feature), token Jaccard / containment, acronym, legal-suffix compatibility
B. address   token ratios (order-invariant), numeric-set Jaccard / containment / lead equality /
             near-miss (157th vs 157nd), IDF-weighted overlap, empty-address flags
C. context   blocking key bits, pre-score, ANN similarity, and competition features: rank of this
             pair within its S1 entity, margin to the entity's best, rank of this S1 among all S1s
             competing for the same satellite. (Model-score versions are added in stage 4.)

Output: work/<split>/features/part-*.parquet  (s1, s23, <features>)
"""
import pickle
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein, Prefix
from rapidfuzz.process import cpdist

from . import config

SIG = ['idx', 'name_c', 'name_core', 'sfx', 'squash', 'phon', 'addr_tok', 'nums', 'is_domain', 'non_latin', 'addr_empty']

# (feature name, column, rapidfuzz scorer) — computed pairwise in C++
FUZZY = [
    ('name_ratio', 'name_c', fuzz.ratio), ('name_partial', 'name_c', fuzz.partial_ratio),
    ('name_tsort', 'name_c', fuzz.token_sort_ratio), ('name_tset', 'name_c', fuzz.token_set_ratio),
    ('core_ratio', 'name_core', fuzz.ratio), ('core_tset', 'name_core', fuzz.token_set_ratio),
    ('sq_ratio', 'squash', fuzz.ratio), ('sq_partial', 'squash', fuzz.partial_ratio),
    ('sq_jw', 'squash', JaroWinkler.normalized_similarity), ('sq_lcs', 'squash', LCSseq.normalized_similarity),
    ('sq_prefix', 'squash', Prefix.similarity), ('phon_ratio', 'phon', fuzz.ratio),
    ('addr_ratio', 'addr_tok', fuzz.ratio), ('addr_partial', 'addr_tok', fuzz.partial_ratio),
    ('addr_tsort', 'addr_tok', fuzz.token_sort_ratio), ('addr_tset', 'addr_tok', fuzz.token_set_ratio),
    ('num_tset', 'nums', fuzz.token_set_ratio),
]
PY = ['core_jacc', 'core_contain', 'acronym', 'sfx_compat', 'num_jacc', 'num_contain', 'lead_eq', 'lead_in',
      'num_near', 'addr_jacc', 'idf_cov_a', 'idf_cov_b', 'idf_shared_sum', 'idf_shared_max']
CONTEXT = ['k1', 'k2', 'k3', 'k4', 'k5', 'n_keys', 'prescore', 'ann_sim', 'pre_rank', 'n_cand', 'pre_margin',
           'pre_rev_rank', 'n_comp']
OTHER = ['sq_prefix_frac', 'name_len_diff', 'name_ntok_diff', 'addr_ntok_a', 'addr_ntok_b', 'n_nums_a', 'n_nums_b',
         'b_is_domain', 'b_non_latin', 'b_addr_empty', 'b_src']
BASE_FEATURES = [f for f, _, _ in FUZZY] + PY + OTHER + CONTEXT

_W = {}


def _init(idf_path):
    with open(idf_path, 'rb') as f:
        d = pickle.load(f)
    _W.update(idf=d['idf'], default=d['default'])


def _sfx_compat(a, b):
    """0 same legal suffix, 1 one side has none, 2 partial overlap, 3 conflicting (llc vs private limited)."""
    if a == b:
        return 0
    if not a or not b:
        return 1
    return 2 if set(a.split()) & set(b.split()) else 3


def _py_batch(args):
    """Worker: set-based features that have no vectorised equivalent (see PY for the order)."""
    idf, default = _W['idf'], _W['default']
    out = np.zeros((len(args[0]), len(PY)), np.float32)
    for i, (ca, cb, sa, sb, na, nb, aa, ab) in enumerate(zip(*args)):
        ta, tb = set(ca.split()), set(cb.split())
        inter = len(ta & tb)
        acro = float(len(ta) >= 2 and ''.join(t[0] for t in ca.split()) == cb.replace(' ', '')) or \
            float(len(tb) >= 2 and ''.join(t[0] for t in cb.split()) == ca.replace(' ', ''))
        la, lb = na.split(), nb.split()
        xa, xb = set(la), set(lb)
        ninter = len(xa & xb)
        near = float(any(len(p) >= 2 and len(q) >= 2 and p != q and Levenshtein.distance(p, q) <= 1
                         for p in la[:4] for q in lb[:4]))
        wa, wb = set(aa.split()), set(ab.split())
        shared = [idf.get(t, default) for t in wa & wb]
        sa_idf = sum(idf.get(t, default) for t in wa) or 1.0
        sb_idf = sum(idf.get(t, default) for t in wb) or 1.0
        out[i] = (inter / max(len(ta | tb), 1), inter / max(min(len(ta), len(tb)), 1), acro, _sfx_compat(sa, sb),
                  ninter / max(len(xa | xb), 1), ninter / max(len(xa), 1),
                  float(bool(la) and bool(lb) and la[0] == lb[0]), float(bool(la) and la[0] in xb), near,
                  len(wa & wb) / max(len(wa | wb), 1), sum(shared) / sa_idf, sum(shared) / sb_idf,
                  sum(shared), max(shared, default=0.0))
    return out


def _ntok(values):
    return np.fromiter((len(v.split()) for v in values), np.float32, len(values))


def pair_features(a, b, pool):
    """Feature matrix (dict name -> float32 array) for aligned lists of S1 (a) and satellite (b) signatures."""
    f = {}
    for name, col, scorer in FUZZY:
        f[name] = cpdist(a[col], b[col], scorer=scorer, workers=-1).astype(np.float32)
    step = max(1, len(a['squash']) // (config.N_JOBS * 4) + 1)
    cols = ('name_core', 'sfx', 'nums', 'addr_tok')
    args = [tuple(x[c][i:i + step] for c in cols for x in (a, b)) for i in range(0, len(a['squash']), step)]
    py = np.concatenate(pool.map(_py_batch, args))
    for j, name in enumerate(PY):
        f[name] = py[:, j]
    len_a = np.fromiter(map(len, a['squash']), np.float32, len(a['squash']))
    len_b = np.fromiter(map(len, b['squash']), np.float32, len(b['squash']))
    f['sq_prefix_frac'] = f['sq_prefix'] / np.maximum(np.minimum(len_a, len_b), 1)
    f['name_len_diff'] = np.abs(np.fromiter(map(len, a['name_c']), np.float32) - np.fromiter(map(len, b['name_c']), np.float32))
    f['name_ntok_diff'] = np.abs(_ntok(a['name_c']) - _ntok(b['name_c']))
    f['addr_ntok_a'], f['addr_ntok_b'] = _ntok(a['addr_tok']), _ntok(b['addr_tok'])
    f['n_nums_a'], f['n_nums_b'] = _ntok(a['nums']), _ntok(b['nums'])
    f['b_is_domain'] = np.asarray(b['is_domain'], np.float32)
    f['b_non_latin'] = np.asarray(b['non_latin'], np.float32)
    f['b_addr_empty'] = np.asarray(b['addr_empty'], np.float32)
    return f


def group_rank(group, value):
    """1-based rank of each row within its group by descending value (ties by position)."""
    return pd.Series(value).groupby(group).rank(ascending=False, method='first').to_numpy(np.float32)


def context_features(cand):
    """Group-C features from the pre-score: how does this pair compare with its competitors?"""
    s1, s23, pre = cand['s1'].to_numpy(), cand['s23'].to_numpy(), cand['prescore'].to_numpy()
    kb = cand['kbits'].to_numpy()
    for j in range(5):
        cand[f'k{j + 1}'] = ((kb >> j) & 1).astype(np.float32)
    cand['n_keys'] = cand[['k1', 'k2', 'k3', 'k4', 'k5']].sum(axis=1).astype(np.float32)
    cand['pre_rank'] = group_rank(s1, pre)
    cand['n_cand'] = cand.groupby('s1')['s1'].transform('size').astype(np.float32)
    cand['pre_margin'] = (cand.groupby('s1')['prescore'].transform('max') - pre).astype(np.float32)
    cand['pre_rev_rank'] = group_rank(s23, pre)
    cand['n_comp'] = cand.groupby('s23')['s23'].transform('size').astype(np.float32)
    return cand


def score_context(df, col):
    """Competition features on a MODEL score (stage-4 second pass): same idea as context_features, plus
    the partition hint 'is this S1 the best claimant of this satellite?'."""
    p = df[col].to_numpy()
    g1, g23 = df.groupby('s1')[col], df.groupby('s23')[col]
    return pd.DataFrame({
        f'{col}_rank': group_rank(df['s1'].to_numpy(), p),
        f'{col}_margin': (g1.transform('max') - p).to_numpy(np.float32),
        f'{col}_sum': g1.transform('sum').to_numpy(np.float32),               # expected cluster size
        f'{col}_n_above': pd.Series(p > 0.5).groupby(df['s1'].to_numpy()).transform('sum').to_numpy(np.float32),
        f'{col}_rev_rank': group_rank(df['s23'].to_numpy(), p),
        f'{col}_rev_margin': (g23.transform('max') - p).to_numpy(np.float32),
    }, index=df.index)


def _gather(table, local_rows):
    """Signature columns of the given rows as python lists (strings) / numpy arrays (flags)."""
    t = table.take(pa.array(local_rows))
    return {c: t[c].to_pylist() for c in SIG if c != 'idx'}


def build(split):
    """Compute features for all candidate pairs of a split, one parquet shard per chunk."""
    out = config.split_dir(split)
    feat_dir = out / 'features'
    shutil.rmtree(feat_dir, ignore_errors=True)
    feat_dir.mkdir()
    cand = context_features(pq.read_table(out / 'candidates.parquet').to_pandas())
    s1_country = pq.read_table(out / 's1_sig.parquet', columns=['country'])['country'].to_numpy(zero_copy_only=False)
    cand['country'] = s1_country[cand['s1'].to_numpy()]

    part = 0
    with Pool(config.N_JOBS, initializer=_init, initargs=(out / 'idf.pkl',)) as pool:
        for country, cc in cand.groupby('country', sort=True):
            t1 = pq.read_table(out / 's1_sig.parquet', columns=SIG, filters=[('country', '=', country)])
            t23 = pq.read_table(out / 's23_sig.parquet', columns=SIG + ['src'], filters=[('country', '=', country)])
            i1, i23 = t1['idx'].to_numpy(), t23['idx'].to_numpy()
            for start in range(0, len(cc), config.FEATURE_CHUNK):
                ch = cc.iloc[start:start + config.FEATURE_CHUNK]
                r23 = np.searchsorted(i23, ch['s23'].to_numpy())
                a = _gather(t1, np.searchsorted(i1, ch['s1'].to_numpy()))
                b = _gather(t23, r23)
                f = pair_features(a, b, pool)
                f['b_src'] = t23['src'].take(pa.array(r23)).to_numpy().astype(np.float32)
                for c in CONTEXT:
                    f[c] = ch[c].to_numpy(np.float32)
                table = pa.table({'s1': ch['s1'].to_numpy(), 's23': ch['s23'].to_numpy(),
                                  **{k: f[k] for k in BASE_FEATURES}})
                pq.write_table(table, feat_dir / f'part-{part:04d}.parquet')
                part += 1
            print(f'[s3] {split}/{country}: features for {len(cc):,} pairs')


def load(split, columns=None):
    """All feature shards of a split as one DataFrame (float32 features)."""
    return pq.read_table(config.WORK_DIR / split / 'features', columns=columns).to_pandas()
