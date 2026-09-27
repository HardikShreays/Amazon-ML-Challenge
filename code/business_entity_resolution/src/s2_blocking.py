"""Stage 2 — multi-key blocking: 10^13 possible pairs -> <= MAX_CANDIDATES per S1 entity.

Keys (all within one country; country is an open set of labels, never hardcoded):
    K1  every address number (first NUMS_PER_ADDR) x 3-char prefix of each of the 2 rarest address words
    K2  the 2 rarest address words (order-invariant: survives component reordering)
    K3  first 8 chars of the core-name squash (domain names, suffix drift, punctuation)
    K4  metaphone of the first two core-name tokens (typos)
    K5  char-trigram TF-IDF -> truncated SVD -> HNSW cosine top-ANN_K (safety net for everything fuzzy)
    K6  each of the first 2 core-name tokens x each of the first NUMS_PER_ADDR address numbers
    K7  the whole core-name squash (K3's 8-char prefix is too generic for common first words:
        'heritage...' blocks exceed MAX_BLOCK, 'heritagequalityhorse' does not)
A key value shared by more than MAX_BLOCK satellite rows is too generic and is ignored.

EDA (notebook §8): plain K1-K4 cover 92.2% of true pairs, the refined K1/K3/K4 used here 95.4%;
K5 targets the rest (non-Latin names with noisy addresses).

K6 + K7 were added after measuring an uncapped union on 8k held-out train entities: they catch ~40%
of the true pairs no K1-K5 key produced (0.953 union recall).

The union is scored with a cheap, non-learned pre-score and the top MAX_CANDIDATES per entity are
kept. That bounded set is exactly what the model scores, i.e. it IS candidate_pairs.tsv.

Output: work/<split>/candidates.parquet  (s1, s23, kbits, prescore, ann_sim)
        work/<split>/entities.npy        S1 indices that were blocked (train: the sampled ones)
"""
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config
from .s1_signatures import load_idf

K1, K2, K3, K4, K5, K6, K7 = 1, 2, 4, 8, 16, 32, 64
KEY_NAMES = {K1: 'K1 numbers', K2: 'K2 rare words', K3: 'K3 name squash', K4: 'K4 phonetic', K5: 'K5 ANN',
             K6: 'K6 name token x number', K7: 'K7 full squash'}
LOOKUP_BITS = (K1, K2, K3, K4, K6, K7)
COLS = ['idx', 'squash', 'phon', 'addr_tok', 'nums', 'name_c', 'name_core']
KEY_COLS = ('squash', 'phon', 'addr_tok', 'nums', 'name_core')
_W = {}   # per-worker IDF table, filled by _init


def _init(idf, default):
    """Worker initialiser: share the address-token IDF table used by the pre-score."""
    _W.update(idf=idf, default=default)


def rare_words(addr_tok, idf, default, k=2):
    """The k highest-IDF distinct address words (ties broken alphabetically for determinism)."""
    return sorted(set(addr_tok.split()), key=lambda t: (-idf.get(t, default), t))[:k]


def record_keys(frame, idf, default):
    """Blocking keys K1-K4, K6, K7 for every row of `frame` -> {bit: (row positions, uint64 key hashes)}."""
    out = {b: ([], []) for b in LOOKUP_BITS}

    def add(bit, i, key):
        """Record one (row, key value) emission for blocking key `bit`."""
        out[bit][0].append(i)
        out[bit][1].append(key)

    for i, (sq, ph, at, nm, core) in enumerate(zip(frame['squash'], frame['phon'], frame['addr_tok'], frame['nums'],
                                                   frame['name_core'])):
        r2 = rare_words(at, idf, default)
        lead = [t for t in core.split()[:2] if len(t) >= 3]
        for n in nm.split()[:config.NUMS_PER_ADDR]:
            for r in r2:
                add(K1, i, f'{n}|{r[:3]}')
            for t in lead:
                add(K6, i, f'{t}#{n}')
        if len(r2) == 2:
            add(K2, i, '|'.join(r2 if r2[0] < r2[1] else r2[::-1]))
        if len(sq) >= 4:
            add(K3, i, sq[:8])
        if ph:
            add(K4, i, ph)
        if len(sq) >= 6:
            add(K7, i, '=' + sq)
    return {b: (np.asarray(rows, dtype=np.int64), pd.util.hash_array(np.asarray(keys, dtype=object)))
            for b, (rows, keys) in out.items()}


def _keys_batch(cols):
    """Worker: record_keys for one slice of KEY_COLS."""
    return record_keys(dict(zip(KEY_COLS, cols)), _W['idf'], _W['default'])


def parallel_keys(pool, frame):
    """record_keys over all cores: slices are keyed in the workers, then re-offset and joined in order."""
    n = len(frame['squash'])
    step = max(1, math.ceil(n / (config.N_JOBS * 4)))
    starts = range(0, n, step)
    parts = pool.map(_keys_batch, [tuple(frame[c][i:i + step] for c in KEY_COLS) for i in starts])
    return {b: (np.concatenate([p[b][0] + i for p, i in zip(parts, starts)]),
                np.concatenate([p[b][1] for p in parts])) for b in LOOKUP_BITS}


class KeyIndex:
    """Satellite keys sorted by hash, so a batch of S1 keys is resolved with two searchsorted calls."""

    def __init__(self, keys):
        self.tables = {}
        for bit, (rows, hashes) in keys.items():
            order = np.argsort(hashes, kind='stable')
            self.tables[bit] = (hashes[order], rows[order])

    def lookup(self, bit, q_rows, q_hashes):
        """All (s1_row, s23_row) sharing a key value, skipping values with > MAX_BLOCK satellites."""
        hashes, rows = self.tables[bit]
        lo = np.searchsorted(hashes, q_hashes, 'left')
        hi = np.searchsorted(hashes, q_hashes, 'right')
        cnt = hi - lo
        ok = (cnt > 0) & (cnt <= config.MAX_BLOCK)
        q_rows, lo, cnt = q_rows[ok], lo[ok], cnt[ok]
        starts = np.repeat(lo - np.cumsum(cnt) + cnt, cnt)       # offset trick: expand ranges without a loop
        pos = starts + np.arange(cnt.sum())
        return np.repeat(q_rows, cnt), rows[pos]


class AnnIndex:
    """K5: char-trigram TF-IDF compressed with truncated SVD, searched with HNSW (cosine)."""

    def __init__(self, texts, pool):
        import faiss                    # hnswlib has no Windows wheels; faiss HNSW with the same M / ef settings
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        rng = np.random.default_rng(config.SEED)
        fit = [texts[i] for i in rng.permutation(len(texts))[:config.SVD_FIT_ROWS]]
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                                   dtype=np.float32).fit(fit)
        if hasattr(self.vec, 'stop_words_'):
            del self.vec.stop_words_    # every pruned trigram; only bloats the copy shipped to workers
        self.pool = pool
        x = self.vec.transform(fit)
        dim = max(2, min(config.SVD_DIM, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(dim, random_state=config.SEED).fit(x)
        faiss.omp_set_num_threads(config.N_JOBS)
        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)   # cosine = IP on unit vectors
        self.index.hnsw.efConstruction = 100
        for i in range(0, len(texts), config.CHUNK_ROWS):              # ids are insertion order = row index
            self.index.add(self.embed(texts[i:i + config.CHUNK_ROWS]))
        self.index.hnsw.efSearch = max(64, config.ANN_K * 2)
        self.n = len(texts)

    def embed(self, texts):
        """TF-IDF + SVD row by row, so slicing across workers gives exactly the single-process result."""
        step = max(1, math.ceil(len(texts) / config.N_JOBS))
        parts = self.pool.map(_embed_batch, [(self.vec, self.svd, texts[i:i + step])
                                             for i in range(0, len(texts), step)])
        x = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
        self.faiss.normalize_L2(x)
        return x

    def query(self, texts):
        """-> (query row, satellite row, cosine similarity) for the top-ANN_K neighbours of each text."""
        k = min(config.ANN_K, self.n)
        sim, labels = self.index.search(self.embed(texts), k)
        q, labels, sim = np.repeat(np.arange(len(texts)), k), labels.ravel().astype(np.int64), sim.ravel()
        ok = labels >= 0                                                  # faiss pads short result lists with -1
        return q[ok], labels[ok], sim[ok]


def _embed_batch(args):
    """Worker: TF-IDF -> SVD for one slice of texts."""
    vec, svd, texts = args
    return svd.transform(vec.transform(texts)).astype(np.float32)


def ann_text(frame):
    """Text embedded by K5: full canonical name + folded address + numbers."""
    return [f'{a} {b} {c}' for a, b, c in zip(frame['name_c'], frame['addr_tok'], frame['nums'])]


def prescore(a, b, kbits):
    """Cheap non-learned pair score used only to rank/cap candidates (all rapidfuzz C++, threaded).
    A pair where either address is empty is ranked on the name alone: under the name+address blend it
    could never make the cap (held-out train: recall at 40 candidates 0.926 -> 0.952 with this rule)."""
    name = cpdist(a['squash'], b['squash'], scorer=fuzz.ratio, workers=-1)
    addr = cpdist(a['addr_tok'], b['addr_tok'], scorer=fuzz.token_set_ratio, workers=-1)
    num = cpdist(a['nums'], b['nums'], scorer=fuzz.token_set_ratio, workers=-1)
    name_set = cpdist(a['name_c'], b['name_c'], scorer=fuzz.token_set_ratio, workers=-1)
    no_addr = np.fromiter((not x or not y for x, y in zip(a['addr_tok'], b['addr_tok'])), bool, len(name))
    nkeys = np.unpackbits(kbits[:, None], axis=1).sum(axis=1)
    blend = np.where(no_addr, 0.9 * np.maximum(name, name_set), 0.35 * name + 0.35 * addr + 0.2 * num)
    return blend / 100 + 0.02 * nkeys


def top_k_per_group(group, score, k):
    """Positions of the k highest-scoring rows within each group (vectorised)."""
    order = np.lexsort((-score, group))
    g = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    rank = np.arange(len(g)) - np.repeat(starts, sizes)
    return order[rank < k]


def _load(path, country):
    """Signature columns of one country as {column: python list / numpy array}."""
    t = pq.read_table(path, columns=COLS, filters=[('country', '=', country)])
    return {c: (t[c].to_numpy() if c == 'idx' else t[c].to_pylist()) for c in COLS}


def _take(frame, rows):
    """The columns the pre-score needs, for the given rows of a signature frame."""
    return {c: [frame[c][i] for i in rows] for c in ('squash', 'addr_tok', 'nums', 'name_c')}


def block_country(s1, s23, pool, writer):
    """Block every S1 row of one country against that country's satellites; append to writer."""
    index = KeyIndex(parallel_keys(pool, s23))
    ann = AnnIndex(ann_text(s23), pool) if len(s23['idx']) > 1 else None
    total = 0
    for start in range(0, len(s1['idx']), config.S1_CHUNK):
        sl = slice(start, start + config.S1_CHUNK)
        chunk = {c: s1[c][sl] for c in COLS}
        pa_, pb_, bits_, sims_ = [], [], [], []
        for bit, (rows, hashes) in parallel_keys(pool, chunk).items():
            a, b = index.lookup(bit, rows, hashes)
            pa_.append(a); pb_.append(b); bits_.append(np.full(len(a), bit, np.uint8)); sims_.append(np.zeros(len(a), np.float32))
        if ann is not None:
            a, b, sim = ann.query(ann_text(chunk))
            pa_.append(a); pb_.append(b); bits_.append(np.full(len(a), K5, np.uint8)); sims_.append(sim.astype(np.float32))
        a, b = np.concatenate(pa_), np.concatenate(pb_)
        if not len(a):
            continue
        # union of all keys: one row per pair, key bits OR-ed, best ANN similarity kept
        code, inv = np.unique((a << 32) | b, return_inverse=True)
        kbits = np.zeros(len(code), np.uint8); np.bitwise_or.at(kbits, inv, np.concatenate(bits_))
        sim = np.zeros(len(code), np.float32); np.maximum.at(sim, inv, np.concatenate(sims_))
        a, b = (code >> 32).astype(np.int64), (code & 0xFFFFFFFF).astype(np.int64)

        score = prescore(_take(chunk, a), _take(s23, b), kbits)
        keep = top_k_per_group(a, score, config.MAX_CANDIDATES)
        writer.write_table(pa.table({
            's1': chunk['idx'][a[keep]].astype(np.int32), 's23': s23['idx'][b[keep]].astype(np.int32),
            'kbits': kbits[keep], 'prescore': score[keep].astype(np.float32), 'ann_sim': sim[keep],
        }))
        total += len(keep)
    return total


def build(split):
    """Run blocking for every country present in this split's S1."""
    out = config.split_dir(split)
    idf, default = load_idf(split)
    s1_meta = pq.read_table(out / 's1_sig.parquet', columns=['idx', 'country']).to_pandas()
    entities = s1_meta['idx'].to_numpy()
    if split == 'train' and config.TRAIN_S1_SAMPLE and config.TRAIN_S1_SAMPLE < len(entities):
        rng = np.random.default_rng(config.SEED)
        entities = np.sort(rng.choice(entities, config.TRAIN_S1_SAMPLE, replace=False))
    np.save(out / 'entities.npy', entities)
    keep = np.zeros(len(s1_meta), bool); keep[entities] = True

    schema = pa.schema([('s1', pa.int32()), ('s23', pa.int32()), ('kbits', pa.uint8()),
                        ('prescore', pa.float32()), ('ann_sim', pa.float32())])
    with pq.ParquetWriter(out / 'candidates.parquet', schema) as writer, \
            config.worker_pool(_init, (idf, default)) as pool:
        for country in sorted(s1_meta['country'].unique()):
            s1 = _load(out / 's1_sig.parquet', country)
            sel = keep[s1['idx']]
            s1 = {c: (v[sel] if c == 'idx' else [x for x, k in zip(v, sel) if k]) for c, v in s1.items()}
            s23 = _load(out / 's23_sig.parquet', country)
            if not len(s1['idx']) or not len(s23['idx']):
                continue
            n = block_country(s1, s23, pool, writer)
            print(f'[s2] {split}/{country}: {len(s1["idx"]):,} S1 x {len(s23["idx"]):,} S2/S3 -> {n:,} candidates '
                  f'({n / len(s1["idx"]):.1f} per entity)')
