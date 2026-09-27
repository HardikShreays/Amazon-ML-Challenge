"""Stage 1b — dictionaries mined from the provided data (nothing hand-written, no external data).

1. abbreviation map (address tokens) — from TRAIN ground-truth pairs: a short token in one address
   that is a same-first-letter subsequence of an unmatched token (or joined bigram) in its true
   partner's address: dr→drive, rd→road, mh→maharashtra, nc→northcarolina.
2. transliteration map (name + address tokens) — from TRAIN ground-truth pairs whose satellite side
   is in an Indic script: romanised token → best fuzzy Latin token of the partner: praivet→private.
3. legal / generic suffix set, per country — from S1 names only (no labels): trailing tokens and
   trailing bigrams used by >= LEGAL_MIN_SHARE of a country's S1 names. Countries seen in train
   keep exactly the train set, so the model sees the features it was trained on. A country absent
   from train (France) gets the strict rule in mine_legal_strict: the loose rule also stripped its
   generic nouns (club, centre, ecole...), which collapsed sibling businesses such as 'Lille Club'
   and 'Lille Amicale' onto one core name (test audit: 78 multi-owner satellites per 1k S1s).

4. abbreviation map for a country absent from train (France) — mined WITHOUT labels from the split
   itself: S1/satellite pairs with an identical squashed name (>= 8 chars, unique among the
   country's S1s) and the same first address number are near-certain matches and stand in for the
   ground truth of (1): av→avenue, bd→boulevard, st→saint, imp→impasse, rte→route ... Joined-bigram
   artefacts ('rd'→'rued') are dropped by requiring the long form to be a real address token of
   that country. Countries seen in train keep exactly the train map.

Outputs: work/train/maps.json (1+2), work/<split>/legal.json (3, {country: [tokens]}),
         work/<split>/abbr_extra.json (4, {country: {short: long}}; test only).
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from . import config
from .s1_normalise import canon, has_indic

_INDIC_RUN = re.compile(r'[ऀ-ൿ]+')
_NUM = re.compile(r'\d+')


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
        # joined bigrams only from unmatched tokens: 'north carolina' can explain 'nc', 'nagel circle' can't
        longs1 = u1 | {x + y for x, y in zip(t1, t1[1:]) if x in u1 and y in u1}
        longs2 = u2 | {x + y for x, y in zip(t2, t2[1:]) if x in u2 and y in u2}
        for shorts, longs in ((u1, longs2), (u2, longs1)):
            for s in shorts:
                if 2 <= len(s) <= 4:
                    totals[s] += 1
                    for lg in longs:
                        if len(lg) >= len(s) + 2 and _is_subseq(s, lg):
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


def mine_legal_strict(names_canon, min_share=config.LEGAL_MIN_SHARE):
    """One country's unambiguous legal forms: frequent final tokens that are final in >= LEGAL_MIN_LAST
    of the names containing them and follow >= LEGAL_MIN_PREDECESSORS different words."""
    last, anywhere, pred = Counter(), Counter(), defaultdict(set)
    for toks in (n.split() for n in names_canon):
        if toks:
            last[toks[-1]] += 1
            anywhere.update(set(toks))
            if len(toks) >= 2:
                pred[toks[-1]].add(toks[-2])
    n = max(len(names_canon), 1)
    return sorted(t for t, v in last.items()
                  if v / n >= min_share and v / anywhere[t] >= config.LEGAL_MIN_LAST
                  and len(pred[t]) >= config.LEGAL_MIN_PREDECESSORS and not t.isdigit() and len(t) <= 12)


def mine_abbreviations_unlabelled(split, country):
    """Abbreviation map for one country from near-certain S1/satellite pairs (module docstring, 4)
    -> ({short: long}, number of pseudo pairs)."""
    out = config.split_dir(split)

    def frame(name):
        """Squashed name, first house number and canonical address of every record of `country` in one raw parquet."""
        t = pq.read_table(out / name, columns=['business_name', 'business_address', 'country'],
                          filters=[('country', '=', country)])
        addr = [canon(a) for a in t['business_address'].to_pylist()]
        first = [_NUM.search(a) for a in addr]
        return pd.DataFrame({'sq': [canon(n).replace(' ', '') for n in t['business_name'].to_pylist()],
                             'n0': [(m.group().lstrip('0') or '0') if m else '' for m in first], 'addr': addr})

    a, b = frame('s1_raw.parquet'), frame('s23_raw.parquet')
    a, b = a[(a.sq.str.len() >= 8) & (a.n0 != '')], b[(b.sq.str.len() >= 8) & (b.n0 != '')]
    a = a[~a.duplicated(['sq', 'n0'], keep=False)]
    pairs = a.merge(b, on=['sq', 'n0'], suffixes=('_a', '_b'))
    mined = mine_abbreviations(zip(pairs.addr_a, pairs.addr_b), config.DICT_MIN_COUNT)
    vocab = Counter(t for x in a.addr for t in set(x.split()))
    return {k: v for k, v in mined.items() if vocab[v] >= config.DICT_MIN_COUNT}, len(pairs)


def build_abbr_extra(split):
    """Mine and store the unlabelled abbreviation maps of every S1 country absent from train."""
    out = config.split_dir(split)
    seen = set(pq.read_table(config.WORK_DIR / 'train' / 's1_raw.parquet', columns=['country'])['country']
               .unique().to_pylist())
    countries = set(pq.read_table(out / 's1_raw.parquet', columns=['country'])['country'].unique().to_pylist())
    extra = {}
    for c in sorted(countries - seen):
        extra[c], n = mine_abbreviations_unlabelled(split, c)
        print(f'[s1] {split}/{c}: {len(extra[c])} abbreviations mined without labels from {n:,} near-certain '
              f'pairs, e.g. {list(extra[c].items())[:8]}')
    (out / 'abbr_extra.json').write_text(json.dumps(extra, indent=0))
    return extra


def _read_legal(path):
    """legal.json as {country: set}; the older flat-list format means one set for every country ('*')."""
    raw = json.loads(path.read_text())
    return {'*': set(raw)} if isinstance(raw, list) else {c: set(v) for c, v in raw.items()}


def legal_for(legal, country):
    """Suffix set of one country from a _read_legal dict."""
    return legal.get(country, legal.get('*', set()))


def build(split):
    """Mine the train-only maps (train split) and the per-split, per-country legal suffix sets."""
    out = config.split_dir(split)
    s1 = pq.read_table(out / 's1_raw.parquet', columns=['business_name', 'country'])
    names = [canon(n) for n in s1['business_name'].to_pylist()]
    countries = s1['country'].to_pylist()
    if split == 'train':
        union = mine_legal(names, countries)             # one set for all train countries, as trained on
        legal = {c: union for c in sorted(set(countries))}
    else:
        train = _read_legal(config.WORK_DIR / 'train' / 'legal.json')
        seen = set(pq.read_table(config.WORK_DIR / 'train' / 's1_raw.parquet', columns=['country'])['country']
                   .unique().to_pylist())
        legal = {}
        for c in sorted(set(countries)):
            legal[c] = (sorted(legal_for(train, c)) if c in seen else
                        mine_legal_strict([n for n, k in zip(names, countries) if k == c]))
    (out / 'legal.json').write_text(json.dumps(legal))
    for c, toks in legal.items():
        print(f'[s1] {split}/{c}: {len(toks)} legal/generic suffix tokens, e.g. {toks[:12]}')

    if split != 'train':
        build_abbr_extra(split)
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
    """-> (abbr, translit, {country: legal set}) for a split. The maps come from train; `abbr` is
    {country: map}: '*' is the train map, and a country unseen in train gets ONLY its own unlabelled
    map (mined here on first use if abbr_extra.json does not exist yet). The train map is US/India
    specific: applied to France it read the stopwords 'de la' as 'delaware louisiana'."""
    maps = json.loads((config.WORK_DIR / 'train' / 'maps.json').read_text())
    abbr = {'*': maps['abbr']}
    if split != 'train':
        path = config.WORK_DIR / split / 'abbr_extra.json'
        abbr.update(json.loads(path.read_text()) if path.exists() else build_abbr_extra(split))
    return abbr, maps['translit'], _read_legal(config.WORK_DIR / split / 'legal.json')
