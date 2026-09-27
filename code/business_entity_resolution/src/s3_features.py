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
import math
import pickle
import shutil
from collections import Counter

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
CONTEXT = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'n_keys', 'prescore', 'ann_sim', 'pre_rank', 'n_cand', 'pre_margin',
           'pre_rev_rank', 'n_comp']
OTHER = ['sq_prefix_frac', 'name_len_diff', 'name_ntok_diff', 'addr_ntok_a', 'addr_ntok_b', 'n_nums_a', 'n_nums_b',
         'b_is_domain', 'b_non_latin', 'b_addr_empty', 'b_src']
# D. name rarity (config.NAME_FEATS): a name-only match on a unique name is near-certain, on a name
#    shared by dozens of S1s it is a coin flip; nothing above says which. IDF over core-name tokens of
#    the split, plus how many S1s / satellites of the country carry exactly this squashed name.
NAME = ['nidf_shared_sum', 'nidf_shared_max', 'nidf_cov_a', 'nidf_cov_b', 'nidf_miss_max_a', 'nidf_miss_max_b',
        'sq_n_s1_a', 'sq_n_s1_b', 'sq_n_s23_b']
BASE_FEATURES = [f for f, _, _ in FUZZY] + PY + OTHER + CONTEXT + (NAME if config.NAME_FEATS else [])

_W = {}


def _init(idf_path):
    """Worker initialiser: load the address-token IDF table once per process."""
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
    """Number of whitespace tokens of each string, as float32."""
    return np.fromiter((len(v.split()) for v in values), np.float32, len(values))


def name_stats(split):
    """Name-token IDF over S1 + S2/S3 core names of the split, and exact squash counts per country
    for S1 and for satellites (unsupervised; cached in work/<split>/name_stats.pkl)."""
    path = config.WORK_DIR / split / 'name_stats.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    df, n, sq = Counter(), 0, {}
    for side, name in (('s1', 's1_sig.parquet'), ('s23', 's23_sig.parquet')):
        t = pq.read_table(config.WORK_DIR / split / name, columns=['name_core', 'squash', 'country'])
        for core in t['name_core'].to_pylist():
            df.update(set(core.split()))
        n += t.num_rows
        keys, counts = np.unique(_name_keys(t['country'].to_pylist(), t['squash'].to_pylist()), return_counts=True)
        sq[side] = (keys, counts)                               # sorted hashes of (country, squash) -> count
        del t
    stats = {'idf': {k: math.log(n / (1 + v)) for k, v in df.items()}, 'default': math.log(n), **sq}
    with open(path, 'wb') as f:
        pickle.dump(stats, f)
    return stats


def _name_keys(countries, squashes):
    """int64 hash of (country, squashed name), used to count how many records share an exact name."""
    return pd.util.hash_array(np.asarray([f'{c}|{q}' for c, q in zip(countries, squashes)], dtype=object))


def _count(table, keys):
    """How many records carry each key (0 when absent), from a (sorted keys, counts) pair."""
    k, c = table
    pos = np.minimum(np.searchsorted(k, keys), len(k) - 1)
    return np.where(k[pos] == keys, c[pos], 0)


def name_features(a, b, stats, country):
    """Group-D features (see NAME) for aligned S1 (a) / satellite (b) signature lists."""
    idf, default = stats['idf'], stats['default']
    n = len(a['name_core'])
    ka, kb = _name_keys([country] * n, a['squash']), _name_keys([country] * n, b['squash'])
    out = np.zeros((n, len(NAME)), np.float32)
    out[:, 6] = np.log1p(_count(stats['s1'], ka))
    out[:, 7] = np.log1p(_count(stats['s1'], kb))
    out[:, 8] = np.log1p(_count(stats['s23'], kb))
    for i, (ca, cb) in enumerate(zip(a['name_core'], b['name_core'])):
        ta, tb = set(ca.split()), set(cb.split())
        wa = {t: idf.get(t, default) for t in ta}
        wb = {t: idf.get(t, default) for t in tb}
        shared = [wa[t] for t in ta & tb]
        out[i, :6] = (sum(shared), max(shared, default=0.0), sum(shared) / (sum(wa.values()) or 1.0),
                      sum(shared) / (sum(wb.values()) or 1.0), max((w for t, w in wa.items() if t not in tb), default=0.0),
                      max((w for t, w in wb.items() if t not in ta), default=0.0))
    return {name: out[:, j] for j, name in enumerate(NAME)}


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


def group_stats(group, value):
    """Per-row statistics of its group, numpy only (tens of millions of rows, no pandas groupby):
    rank (1-based, by descending value), group max, group sum, group size, #values > 0.5."""
    order = np.lexsort((-value, group))
    g, v = group[order], value[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1] if len(g) else np.zeros(0, int)
    sizes = np.diff(np.r_[starts, len(g)])
    sorted_stats = {
        'rank': np.arange(len(g)) - np.repeat(starts, sizes) + 1,
        'max': np.repeat(v[starts], sizes),                      # sorted descending -> first is the max
        'sum': np.repeat(np.add.reduceat(v, starts) if len(g) else v, sizes),
        'size': np.repeat(sizes, sizes),
        'above': np.repeat(np.add.reduceat(v > 0.5, starts) if len(g) else v, sizes),
    }
    out = {}
    for k, sv in sorted_stats.items():
        out[k] = np.empty(len(g), np.float32)
        out[k][order] = sv
    return out


def context_features(cand):
    """Group-C features from the pre-score: how does this pair compare with its competitors?"""
    s1, s23, pre = cand['s1'].to_numpy(), cand['s23'].to_numpy(), cand['prescore'].to_numpy()
    kb = cand['kbits'].to_numpy()
    for j in range(7):
        cand[f'k{j + 1}'] = ((kb >> j) & 1).astype(np.float32)
    cand['n_keys'] = cand[[f'k{j + 1}' for j in range(7)]].sum(axis=1).astype(np.float32)
    by_s1, by_s23 = group_stats(s1, pre), group_stats(s23, pre)
    cand['pre_rank'], cand['n_cand'] = by_s1['rank'], by_s1['size']
    cand['pre_margin'] = by_s1['max'] - pre
    cand['pre_rev_rank'], cand['n_comp'] = by_s23['rank'], by_s23['size']
    return cand


SCORE_CONTEXT = ['p1', 'p1_rank', 'p1_margin', 'p1_sum', 'p1_n_above', 'p1_rev_rank', 'p1_rev_margin']


def score_context(s1, s23, p1):
    """Competition features on the pass-1 MODEL score (stage-4 second pass), as an (n, 7) matrix in
    SCORE_CONTEXT order. p1_sum is the entity's expected cluster size; p1_rev_* tell the model
    whether this S1 is the best claimant of the satellite (the partition property, as a feature)."""
    a, b = group_stats(s1, p1), group_stats(s23, p1)
    return np.column_stack([p1, a['rank'], a['max'] - p1, a['sum'], a['above'],
                            b['rank'], b['max'] - p1]).astype(np.float32)


def _gather(table, local_rows):
    """Signature columns of the given rows as python lists (strings) / numpy arrays (flags)."""
    t = table.take(pa.array(local_rows))
    return {c: t[c].to_pylist() for c in SIG if c != 'idx'}


def build(split, resume=False):
    """Compute features for all candidate pairs of a split, one parquet shard per chunk.
    resume=True keeps the shards already written and skips every country whose shards all exist."""
    out = config.split_dir(split)
    feat_dir = out / 'features'
    if not resume:
        shutil.rmtree(feat_dir, ignore_errors=True)
    feat_dir.mkdir(exist_ok=True)
    cand = pq.read_table(out / 'candidates.parquet')           # stays in Arrow; one country at a time goes to pandas
    names, codes = np.unique(pq.read_table(out / 's1_sig.parquet', columns=['country'])['country']
                             .to_numpy(zero_copy_only=False), return_inverse=True)
    country_of_pair = codes.astype(np.int16)[cand['s1'].to_numpy()]

    stats = name_stats(split) if config.NAME_FEATS else None
    part = 0
    for code, country in enumerate(names):
        sel = country_of_pair == code
        n_parts = -(-int(sel.sum()) // config.FEATURE_CHUNK)
        if resume and all((feat_dir / f'part-{p:04d}.parquet').exists() for p in range(part, part + n_parts)):
            print(f'[s3] {split}/{country}: {n_parts} shards already written, skipping')
            part += n_parts
            continue
        # one country at a time: a satellite only competes with S1s of its own country, so the
        # context features are complete, and peak memory is bounded by the largest country. The
        # worker pool only exists while pair features are computed (each worker holds the IDF table).
        cc = context_features(cand.filter(pa.array(sel)).to_pandas())
        del sel
        with config.worker_pool(_init, (out / 'idf.pkl',)) as pool:
            t1 = pq.read_table(out / 's1_sig.parquet', columns=SIG, filters=[('country', '=', country)])
            t23 = pq.read_table(out / 's23_sig.parquet', columns=SIG + ['src'], filters=[('country', '=', country)])
            i1, i23 = t1['idx'].to_numpy(), t23['idx'].to_numpy()
            for start in range(0, len(cc), config.FEATURE_CHUNK):
                ch = cc.iloc[start:start + config.FEATURE_CHUNK]
                r23 = np.searchsorted(i23, ch['s23'].to_numpy())
                a = _gather(t1, np.searchsorted(i1, ch['s1'].to_numpy()))
                b = _gather(t23, r23)
                f = pair_features(a, b, pool)
                if stats is not None:
                    f.update(name_features(a, b, stats, country))
                f['b_src'] = t23['src'].take(pa.array(r23)).to_numpy().astype(np.float32)
                for c in CONTEXT:
                    f[c] = ch[c].to_numpy(np.float32)
                table = pa.table({'s1': ch['s1'].to_numpy(), 's23': ch['s23'].to_numpy(),
                                  **{k: f[k] for k in BASE_FEATURES}})
                pq.write_table(table, feat_dir / f'part-{part:04d}.parquet')
                part += 1
            print(f'[s3] {split}/{country}: features for {len(cc):,} pairs')
        del cc


def augment(split):
    """Append the group-D (NAME) columns to feature shards written without them, in place. Needs no
    worker pool and no fuzzy matching, so it costs minutes instead of a full feature pass."""
    stats = name_stats(split)
    w = config.WORK_DIR / split
    sig = {side: pq.read_table(w / name, columns=['idx', 'name_core', 'squash', 'country'])
           for side, name in (('s1', 's1_sig.parquet'), ('s23', 's23_sig.parquet'))}
    for path in shards(split):
        t = pq.read_table(path)
        if all(c in t.column_names for c in NAME):
            continue
        a = sig['s1'].take(t['s1'])                      # sig row == record index (idx is 0..n-1 in order)
        b = sig['s23'].take(t['s23'])
        f = {n: np.zeros(t.num_rows, np.float32) for n in NAME}
        countries = a['country'].to_numpy(zero_copy_only=False)
        for country in np.unique(countries):
            rows = np.flatnonzero(countries == country)
            fa = {c: a[c].take(pa.array(rows)).to_pylist() for c in ('name_core', 'squash')}
            fb = {c: b[c].take(pa.array(rows)).to_pylist() for c in ('name_core', 'squash')}
            for n, v in name_features(fa, fb, stats, country).items():
                f[n][rows] = v
        for n in NAME:
            t = t.append_column(n, pa.array(f[n]))
        pq.write_table(t, path)
    print(f'[s3] {split}: name-rarity columns present in all shards')


def shards(split):
    """Feature shard paths of a split, in write order."""
    return sorted((config.WORK_DIR / split / 'features').glob('part-*.parquet'))


def column(paths, col):
    """One column over all shards, concatenated in shard order."""
    return np.concatenate([pq.read_table(p, columns=[col])[col].to_numpy() for p in paths])


def load_matrix(paths, cols, mask=None, extra=0):
    """(s1, s23, X float32 [n, len(cols) + extra]) from feature shards, preallocated to avoid double
    copies; the `extra` trailing columns are left for the caller to fill. `mask` (bool, aligned with the
    rows of `paths` in shard order) loads only those rows, so a sample of a split whose full matrix would
    not fit in RAM can still be read."""
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    bounds = np.r_[0, np.cumsum(sizes)]
    rows = [None if mask is None else np.flatnonzero(mask[bounds[k]:bounds[k + 1]]) for k in range(len(paths))]
    n = sum(s if r is None else len(r) for s, r in zip(sizes, rows))
    s1, s23 = np.empty(n, np.int32), np.empty(n, np.int32)
    x = np.empty((n, len(cols) + extra), np.float32)
    i = 0
    for p, r in zip(paths, rows):
        t = pq.read_table(p, columns=['s1', 's23'] + cols)
        if r is not None:
            t = t.take(pa.array(r))
        m = t.num_rows
        s1[i:i + m], s23[i:i + m] = t['s1'].to_numpy(), t['s23'].to_numpy()
        for j, c in enumerate(cols):
            x[i:i + m, j] = t[c].to_numpy()
        i += m
    return s1, s23, x
