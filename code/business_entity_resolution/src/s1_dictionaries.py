"""Stage 1b — dictionaries mined from the provided data (nothing hand-written, no external data).

1. abbreviation map (address tokens) — from TRAIN ground-truth pairs: a short token in one address
   that is a same-first-letter subsequence of an unmatched token (or joined bigram) in its true
   partner's address: dr→drive, rd→road, mh→maharashtra, nc→northcarolina.
2. transliteration map (name + address tokens) — from TRAIN ground-truth pairs whose satellite side
   is in an Indic script: romanised token → best fuzzy Latin token of the partner: praivet→private.
3. legal / generic suffix set — per split, from S1 names only (no labels): trailing tokens and
   trailing bigrams used by >= LEGAL_MIN_SHARE of a country's S1 names. This is how the unseen
   French sarl / sas / eurl / sci are picked up on the test split.

Outputs: work/train/maps.json (1+2), work/<split>/legal.json (3).
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from . import config
from .s1_normalise import canon, has_indic

_INDIC_RUN = re.compile(r'[ऀ-ൿ]+')


def _is_subseq(short, long):
    """True if `short` can be read off `long` left-to-right and both start with the same letter."""
    if short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def mine_abbreviations(addr_pairs, min_count):
    """addr_pairs: iterable of (canon_addr_a, canon_addr_b) for true pairs -> {short: long}."""
    counts, totals = Counter(), Counter()
    for a, b in addr_pairs:
        t1 = [t for t in a.split() if t.isalpha()]
        t2 = [t for t in b.split() if t.isalpha()]
        s1, s2 = set(t1), set(t2)
        u1, u2 = s1 - s2, s2 - s1
        longs1 = u1 | {x + y for x, y in zip(t1, t1[1:])}
        longs2 = u2 | {x + y for x, y in zip(t2, t2[1:])}
        for shorts, longs in ((u1, longs2), (u2, longs1)):
            for s in shorts:
                if 2 <= len(s) <= 4:
                    totals[s] += 1
                    for lg in longs:
                        if len(lg) >= len(s) + 2 and lg not in s1 & s2 and _is_subseq(s, lg):
                            counts[(s, lg)] += 1
    return _dominant(counts, totals, min_count)


def mine_transliterations(pairs, min_count):
    """pairs: iterable of (raw_indic_text, canon_latin_text) -> {romanised_token: latin_token}."""
    counts, totals = Counter(), Counter()
    for raw, latin in pairs:
        latin_toks = set(latin.split())
        for rom in {canon(w) for w in _INDIC_RUN.findall(raw)}:
            if not rom or rom in latin_toks:
                continue
            totals[rom] += 1
            best = process.extractOne(rom, latin_toks, scorer=fuzz.ratio, score_cutoff=60)
            if best:
                counts[(rom, best[0])] += 1
    return _dominant(counts, totals, min_count)


def _dominant(counts, totals, min_count):
    """Keep, per source token, its most frequent target if seen >= min_count times and in >= 50%
    of that token's occurrences (so ambiguous tokens are left alone)."""
    best = {}
    for (s, t), c in counts.most_common():
        if s not in best and c >= min_count and c >= 0.5 * totals[s]:
            best[s] = t
    return best


def mine_legal(names_canon, countries, min_share=config.LEGAL_MIN_SHARE):
    """Trailing tokens / trailing bigram tokens frequent among one country's S1 names."""
    by_country = defaultdict(list)
    for n, c in zip(names_canon, countries):
        by_country[c].append(n.split())
    legal = set()
    for toks_list in by_country.values():
        uni = Counter(t[-1] for t in toks_list if t)
        bi = Counter((t[-2], t[-1]) for t in toks_list if len(t) >= 2)
        n = len(toks_list)
        legal |= {t for t, v in uni.items() if v / n >= min_share}
        legal |= {w for (a, b), v in bi.items() if v / n >= min_share and b in legal for w in (a, b)}
    return sorted(t for t in legal if not t.isdigit() and len(t) <= 12)


def build(split):
    """Mine the train-only maps (train split) and the per-split legal suffix set."""
    out = config.split_dir(split)
    s1 = pq.read_table(out / 's1_raw.parquet', columns=['business_name', 'country'])
    names = [canon(n) for n in s1['business_name'].to_pylist()]
    legal = mine_legal(names, s1['country'].to_pylist())
    (out / 'legal.json').write_text(json.dumps(legal))
    print(f'[s1] {split}: {len(legal)} legal/generic suffix tokens, e.g. {legal[:12]}')

    if split != 'train':
        return
    gt = pq.read_table(out / 'gt_pairs.parquet').to_pandas()
    gt = gt.sample(min(len(gt), config.DICT_SAMPLE_PAIRS), random_state=config.SEED)
    cols = ['business_name', 'business_address']
    a = pq.read_table(out / 's1_raw.parquet', columns=cols).take(gt.s1.to_numpy()).to_pandas()
    b = pq.read_table(out / 's23_raw.parquet', columns=cols).take(gt.s23.to_numpy()).to_pandas()

    abbr = mine_abbreviations(((canon(x), canon(y)) for x, y in zip(a.business_address, b.business_address)),
                              config.DICT_MIN_COUNT)
    indic = [(rb, canon(ra)) for ra, rb in zip(np.concatenate([a.business_name, a.business_address]),
                                               np.concatenate([b.business_name, b.business_address]))
             if has_indic(rb)]
    translit = mine_transliterations(indic, config.DICT_MIN_COUNT)
    (out / 'maps.json').write_text(json.dumps({'abbr': abbr, 'translit': translit}, indent=0))
    print(f'[s1] mined {len(abbr)} abbreviations (e.g. {list(abbr.items())[:6]}) and '
          f'{len(translit)} transliterations (e.g. {list(translit.items())[:6]})')


def load(split):
    """-> (abbr, translit, legal) for a split; the maps always come from train."""
    maps = json.loads((config.WORK_DIR / 'train' / 'maps.json').read_text())
    legal = json.loads((config.WORK_DIR / split / 'legal.json').read_text())
    return maps['abbr'], maps['translit'], set(legal)
