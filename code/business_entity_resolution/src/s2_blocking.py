"""Stage 2 — multi-key blocking: 10^13 possible pairs -> <= MAX_CANDIDATES per S1 entity.

Keys (all within one country; country is an open set of labels, never hardcoded):
    K1  every address number (first NUMS_PER_ADDR) x 3-char prefix of each of the 2 rarest address words
    K2  the 2 rarest address words (order-invariant: survives component reordering)
    K3  first 8 chars of the core-name squash (domain names, suffix drift, punctuation)
    K4  metaphone of the first two core-name tokens (typos)
    K5  char-trigram TF-IDF -> truncated SVD -> HNSW cosine top-ANN_K (safety net for everything fuzzy)
A key value shared by more than MAX_BLOCK satellite rows is too generic and is ignored.

EDA (notebook §8): plain K1-K4 cover 92.2% of true pairs, the refined K1/K3/K4 used here 95.4%;
K5 targets the rest (non-Latin names with noisy addresses).

The union is scored with a cheap, non-learned pre-score and the top MAX_CANDIDATES per entity are
kept. That bounded set is exactly what the model scores, i.e. it IS candidate_pairs.tsv.

Output: work/<split>/candidates.parquet  (s1, s23, kbits, prescore, ann_sim)
        work/<split>/entities.npy        S1 indices that were blocked (train: the sampled ones)
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config
from .s1_signatures import load_idf

K1, K2, K3, K4, K5 = 1, 2, 4, 8, 16
KEY_NAMES = {K1: 'K1 numbers', K2: 'K2 rare words', K3: 'K3 name squash', K4: 'K4 phonetic', K5: 'K5 ANN'}
COLS = ['idx', 'squash', 'phon', 'addr_tok', 'nums', 'name_c']


def rare_words(addr_tok, idf, default, k=2):
    """The k highest-IDF distinct address words (ties broken alphabetically for determinism)."""
    return sorted(set(addr_tok.split()), key=lambda t: (-idf.get(t, default), t))[:k]


def record_keys(frame, idf, default):
    """Blocking keys K1-K4 for every row of `frame` -> {bit: (row positions, uint64 key hashes)}."""
    out = {b: ([], []) for b in (K1, K2, K3, K4)}

    def add(bit, i, key):
        out[bit][0].append(i)
        out[bit][1].append(key)

    for i, (sq, ph, at, nm) in enumerate(zip(frame['squash'], frame['phon'], frame['addr_tok'], frame['nums'])):
        r2 = rare_words(at, idf, default)
        for n in nm.split()[:config.NUMS_PER_ADDR]:
            for r in r2:
                add(K1, i, f'{n}|{r[:3]}')
        if len(r2) == 2:
            add(K2, i, '|'.join(r2 if r2[0] < r2[1] else r2[::-1]))
        if len(sq) >= 4:
            add(K3, i, sq[:8])
        if ph:
            add(K4, i, ph)
    return {b: (np.asarray(rows, dtype=np.int64), pd.util.hash_array(np.asarray(keys, dtype=object)))
            for b, (rows, keys) in out.items()}


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

    def __init__(self, texts):
        import hnswlib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        rng = np.random.default_rng(config.SEED)
        fit = [texts[i] for i in rng.permutation(len(texts))[:config.SVD_FIT_ROWS]]
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                                   dtype=np.float32).fit(fit)
        x = self.vec.transform(fit)
        dim = max(2, min(config.SVD_DIM, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(dim, random_state=config.SEED).fit(x)
        self.index = hnswlib.Index('cosine', dim)
        self.index.init_index(max_elements=len(texts), ef_construction=100, M=16, random_seed=config.SEED)
        for i in range(0, len(texts), config.CHUNK_ROWS):
            part = texts[i:i + config.CHUNK_ROWS]
            self.index.add_items(self.embed(part), np.arange(i, i + len(part)), num_threads=config.N_JOBS)
        self.index.set_ef(max(64, config.ANN_K * 2))
        self.n = len(texts)

    def embed(self, texts):
        return self.svd.transform(self.vec.transform(texts)).astype(np.float32)

    def query(self, texts):
        """-> (query row, satellite row, cosine similarity) for the top-ANN_K neighbours of each text."""
        k = min(config.ANN_K, self.n)
        labels, dist = self.index.knn_query(self.embed(texts), k=k, num_threads=config.N_JOBS)
        return np.repeat(np.arange(len(texts)), k), labels.ravel().astype(np.int64), 1 - dist.ravel()


def ann_text(frame):
    """Text embedded by K5: full canonical name + folded address + numbers."""
    return [f'{a} {b} {c}' for a, b, c in zip(frame['name_c'], frame['addr_tok'], frame['nums'])]


def prescore(a, b, kbits):
    """Cheap non-learned pair score used only to rank/cap candidates (all rapidfuzz C++, threaded)."""
    name = cpdist(a['squash'], b['squash'], scorer=fuzz.ratio, workers=-1)
    addr = cpdist(a['addr_tok'], b['addr_tok'], scorer=fuzz.token_set_ratio, workers=-1)
    num = cpdist(a['nums'], b['nums'], scorer=fuzz.token_set_ratio, workers=-1)
    nkeys = np.unpackbits(kbits[:, None], axis=1).sum(axis=1)
    return (0.35 * name + 0.35 * addr + 0.2 * num) / 100 + 0.02 * nkeys


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
    return {c: [frame[c][i] for i in rows] for c in ('squash', 'addr_tok', 'nums')}


def block_country(s1, s23, idf, default, writer):
    """Block every S1 row of one country against that country's satellites; append to writer."""
    index = KeyIndex(record_keys(s23, idf, default))
    ann = AnnIndex(ann_text(s23)) if len(s23['idx']) > 1 else None
    total = 0
    for start in range(0, len(s1['idx']), config.S1_CHUNK):
        sl = slice(start, start + config.S1_CHUNK)
        chunk = {c: s1[c][sl] for c in COLS}
        pa_, pb_, bits_, sims_ = [], [], [], []
        for bit, (rows, hashes) in record_keys(chunk, idf, default).items():
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
    with pq.ParquetWriter(out / 'candidates.parquet', schema) as writer:
        for country in sorted(s1_meta['country'].unique()):
            s1 = _load(out / 's1_sig.parquet', country)
            sel = keep[s1['idx']]
            s1 = {c: (v[sel] if c == 'idx' else [x for x, k in zip(v, sel) if k]) for c, v in s1.items()}
            s23 = _load(out / 's23_sig.parquet', country)
            if not len(s1['idx']) or not len(s23['idx']):
                continue
            n = block_country(s1, s23, idf, default, writer)
            print(f'[s2] {split}/{country}: {len(s1["idx"]):,} S1 x {len(s23["idx"]):,} S2/S3 -> {n:,} candidates '
                  f'({n / len(s1["idx"]):.1f} per entity)')
