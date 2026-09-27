"""Stage 1c — per-record signatures, computed once and reused by blocking and features.

Columns written to work/<split>/{s1,s23}_sig.parquet (one row per raw row, same order):
    idx        int32 row index (S1 index or S2/S3 index)
    country    raw country label (open set — never hardcoded)
    name_c     canonical name, transliteration map applied, domain TLD dropped
    name_core  name_c without legal/generic suffix tokens (e.g. 'segura classic armour')
    sfx        sorted legal tokens that were stripped ('corp', 'private limited', ...)
    squash     core name with all spaces removed ('seguraclassicarmour')
    phon       metaphone of the first two core tokens
    addr_tok   canonical alphabetic address tokens after abbreviation folding
    nums       numeric runs of the address without leading zeros, in order, de-duplicated ('14243 157')
    is_domain / non_latin / addr_empty   int8 flags
plus `src` (2/3) for satellites.

Also writes idf.pkl: address-token IDF over S1+S2+S3 of THIS split. Deviation from the plan (which
fits IDF on train only): IDF is unsupervised, so computing it on the haystack being searched is not
label leakage, and it is the only way unseen French tokens ('rue', 'avenue') get sensible weights.
"""
import math
import pickle
import re
from collections import Counter

import jellyfish
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, s1_dictionaries
from .s1_normalise import TLDS, canon, has_indic, is_domain

_NUM = re.compile(r'\d+')
_W = {}   # per-worker dictionaries, filled by _init


def _init(abbr, translit, legal):
    """Worker initialiser: share the abbreviation, transliteration and legal-suffix maps."""
    _W.update(abbr=abbr, translit=translit, legal=legal, longs=set(abbr['*'].values()))


def signature(name, addr, country):
    """Signature tuple for one record (see module docstring for the fields)."""
    tr, legal = _W['translit'], s1_dictionaries.legal_for(_W['legal'], country)
    toks = [tr.get(t, t) for t in canon(name).split()]
    dom = is_domain(name)
    if dom:
        toks = [t for t in toks if t not in TLDS] or toks
    core = [t for t in toks if t not in legal] or toks
    sfx = ' '.join(sorted({t for t in toks if t in legal}))
    phon = ' '.join(jellyfish.metaphone(t) for t in core[:2])

    atoks = [tr.get(t, t) for t in canon(addr).split()]
    merged, i = [], 0
    while i < len(atoks):                          # 'north carolina' -> 'northcarolina' (matches nc)
        if i + 1 < len(atoks) and atoks[i] + atoks[i + 1] in _W['longs']:
            merged.append(atoks[i] + atoks[i + 1]); i += 2
        else:
            merged.append(atoks[i]); i += 1
    ab = _W['abbr'].get(country, _W['abbr']['*'])            # per-country map (unseen countries: + mined extras)
    alpha = [ab.get(t, t) for t in merged if t.isalpha() and len(t) > 1]
    nums = list(dict.fromkeys(n.lstrip('0') or '0' for n in _NUM.findall(' '.join(merged))))   # 0040 == 40
    return (' '.join(toks), ' '.join(core), sfx, ''.join(core), phon, ' '.join(alpha), ' '.join(nums),
            int(dom), int(has_indic(name)), int(not addr.strip()))


SIG_COLS = ['name_c', 'name_core', 'sfx', 'squash', 'phon', 'addr_tok', 'nums', 'is_domain', 'non_latin', 'addr_empty']


def _sig_batch(args):
    """Worker: signatures for a list of (name, addr, country) + document frequencies of address tokens."""
    names, addrs, countries = args
    rows = [signature(n, a, c) for n, a, c in zip(names, addrs, countries)]
    df = Counter()
    for r in rows:
        df.update(set(r[5].split()))
    return rows, df


def _write(pool, raw_path, out_path, extra_cols):
    """Stream one raw parquet through the worker pool in CHUNK_ROWS batches."""
    writer, df_total, offset = None, Counter(), 0
    for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=config.CHUNK_ROWS):
        names = batch.column('business_name').to_pylist()
        addrs = batch.column('business_address').to_pylist()
        countries = batch.column('country').to_pylist()
        step = max(1, math.ceil(len(names) / (config.N_JOBS * 4)))
        parts = pool.map(_sig_batch, [(names[i:i + step], addrs[i:i + step], countries[i:i + step])
                                      for i in range(0, len(names), step)])
        rows = [r for p in parts for r in p[0]]
        for p in parts:
            df_total.update(p[1])
        cols = list(zip(*rows))
        data = {'idx': np.arange(offset, offset + len(rows), dtype=np.int32),
                'country': batch.column('country')}
        for c in extra_cols:
            data[c] = batch.column(c)
        for name, values in zip(SIG_COLS, cols):
            data[name] = pa.array(values, type=pa.int8() if name in ('is_domain', 'non_latin', 'addr_empty') else pa.string())
        table = pa.table(data)
        writer = writer or pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        offset += len(rows)
    writer.close()
    return df_total, offset


def build(split):
    """Compute signatures for S1 and S2/S3 of a split and the address IDF table."""
    out = config.split_dir(split)
    abbr, translit, legal = s1_dictionaries.load(split)
    with config.worker_pool(_init, (abbr, translit, legal)) as pool:
        df1, n1 = _write(pool, out / 's1_raw.parquet', out / 's1_sig.parquet', [])
        df2, n2 = _write(pool, out / 's23_raw.parquet', out / 's23_sig.parquet', ['src'])
    df1.update(df2)
    n = n1 + n2
    idf = {t: math.log(n / (1 + c)) for t, c in df1.items()}
    with open(out / 'idf.pkl', 'wb') as f:
        pickle.dump({'idf': idf, 'default': math.log(n)}, f)
    print(f'[s1] {split}: signatures for {n1:,} S1 + {n2:,} S2/S3 rows, {len(idf):,} address tokens in IDF')


def load_idf(split):
    """-> (idf dict, default idf for unseen tokens)."""
    with open(config.WORK_DIR / split / 'idf.pkl', 'rb') as f:
        d = pickle.load(f)
    return d['idf'], d['default']
