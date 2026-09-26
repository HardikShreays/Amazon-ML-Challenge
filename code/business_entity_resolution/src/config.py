"""Paths, constants and tunables shared by every pipeline stage.

Paths can be overridden with environment variables so the same code runs on the full data,
on the smoke-test subset, or from inside the submission zip:

    ER_DATA_DIR    folder containing train/ and test/ TSVs   (default: <repo>/student_resource/dataset)
    ER_WORK_DIR    intermediate parquet / models / shards      (default: <repo>/work)
    ER_OUTPUT_DIR  matching_results.tsv + candidate_pairs.tsv  (default: <repo>/output)
    ER_SMOKE=1     tiny settings for the end-to-end smoke test
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get('ER_DATA_DIR', REPO_ROOT / 'student_resource' / 'dataset'))
WORK_DIR = Path(os.environ.get('ER_WORK_DIR', REPO_ROOT / 'work'))
OUTPUT_DIR = Path(os.environ.get('ER_OUTPUT_DIR', REPO_ROOT / 'output'))
VALIDATOR = REPO_ROOT / 'student_resource' / 'utils' / 'validate_submission.py'
SMOKE = os.environ.get('ER_SMOKE') == '1'

SEED = 42
N_JOBS = os.cpu_count() or 4

# ---- Stage 0/1: ingest, normalisation, signatures -------------------------------------------
CHUNK_ROWS = 1_000_000          # rows per signature chunk (bounds peak RAM)
DICT_SAMPLE_PAIRS = 300_000     # ground-truth pairs used to mine abbreviation / transliteration maps
DICT_MIN_COUNT = 3 if SMOKE else 20
LEGAL_MIN_SHARE = 0.003         # a trailing name token used by >=0.3% of S1 names is a legal/generic suffix

# ---- Stage 2: blocking ----------------------------------------------------------------------
NUMS_PER_ADDR = 3               # K1 keys on each of the first 3 numbers (EDA: the lead number is noisy)
MAX_BLOCK = 200                 # a key value shared by more satellite rows than this is too generic
ANN_K = 30                      # K5: nearest neighbours per S1 entity
SVD_DIM = 16 if SMOKE else 64   # K5: char-trigram TF-IDF compressed to this many dims
SVD_FIT_ROWS = 300_000
MAX_CANDIDATES = 40             # final cap per S1 entity after the cheap pre-score (-> candidate_pairs.tsv)
S1_CHUNK = 50_000               # S1 rows blocked at a time

# ---- Stage 3/4: features and model ----------------------------------------------------------
# 8 GB RAM budget: training pairs come from a sample of S1 entities, but they are always blocked
# against the FULL satellite pool so the model sees the real haystack and real distractors.
TRAIN_S1_SAMPLE = None if SMOKE else 300_000
VALID_FRAC = 0.3                # of the sampled entities; half calibrates, half tunes the decision layer
NEG_PER_POS = 5                 # easy-negative downsampling ratio (hard negatives are always kept)
HARD_NEG_RANK = 5               # negatives ranked <= this by pre-score count as hard
FEATURE_CHUNK = 2_000_000       # candidate pairs per feature shard

LGB_PARAMS = dict(
    objective='binary', learning_rate=0.05, num_leaves=31 if SMOKE else 191,
    min_data_in_leaf=20 if SMOKE else 200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, metric='average_precision', verbose=-1, seed=SEED,
    num_threads=N_JOBS,
)
NUM_BOOST_ROUND = 60 if SMOKE else 3000
EARLY_STOPPING = 20 if SMOKE else 100

# ---- Stage 6: decision layer ----------------------------------------------------------------
MAX_MATCHES = 8                 # 99%+ of training clusters have <= 7 matches
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def worker_pool(initializer=None, initargs=()):
    """N_JOBS worker processes pinned to one BLAS thread. Workers only do Python / rapidfuzz work, and on
    Windows every spawned numpy commits ~1 GB for its OpenBLAS thread buffers (16 workers ~ 18 GB)."""
    from multiprocessing import Pool
    old = os.environ.get('OPENBLAS_NUM_THREADS')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'           # read by the children at numpy import
    try:
        return Pool(N_JOBS, initializer=initializer, initargs=initargs)
    finally:
        if old is None:
            os.environ.pop('OPENBLAS_NUM_THREADS')
        else:
            os.environ['OPENBLAS_NUM_THREADS'] = old


def split_dir(split):
    """Work folder for one split (train / test), created on demand."""
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d
