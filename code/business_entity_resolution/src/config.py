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
VALIDATOR = Path(__file__).resolve().parent / 'validate_submission.py'   # organisers' checker, vendored unchanged
SMOKE = os.environ.get('ER_SMOKE') == '1'

SEED = 42
N_JOBS = os.cpu_count() or 4

# ---- Stage 0/1: ingest, normalisation, signatures -------------------------------------------
CHUNK_ROWS = 1_000_000          # rows per signature chunk (bounds peak RAM)
DICT_SAMPLE_PAIRS = 300_000     # ground-truth pairs used to mine abbreviation / transliteration maps
DICT_MIN_COUNT = 3 if SMOKE else 20
LEGAL_MIN_SHARE = 0.003         # a trailing name token used by >=0.3% of S1 names is a legal/generic suffix
# Countries absent from train (France) get only unambiguous legal forms: tokens that end the name
# whenever they appear and follow many different words (sarl, sas, eurl ...), not generic nouns
# that also sit mid-name (club, centre, ecole) or saint-name parts ('saint jean').
LEGAL_MIN_LAST = 0.9
LEGAL_MIN_PREDECESSORS = 50

# ---- Stage 2: blocking ----------------------------------------------------------------------
NUMS_PER_ADDR = 3               # K1 keys on each of the first 3 numbers (EDA: the lead number is noisy)
MAX_BLOCK = 200                 # a key value shared by more satellite rows than this is too generic
ANN_K = 30                      # K5: nearest neighbours per S1 entity
SVD_DIM = 16 if SMOKE else 64   # K5: char-trigram TF-IDF compressed to this many dims
SVD_FIT_ROWS = 300_000
MAX_CANDIDATES = 40             # final cap per S1 entity after the cheap pre-score (-> candidate_pairs.tsv)
S1_CHUNK = 50_000               # S1 rows blocked at a time

# ---- Stage 3/4: features and model ----------------------------------------------------------
# Every train S1 is blocked (TRAIN_S1_SAMPLE = None), exactly as in test: with a sample, a satellite
# sees only a fraction of its real competitors, so the competition features and the one-owner
# partition were learned in a much sparser world than the one they are applied to. The model is then
# FIT on a sample of TRAIN_FIT_S1 entities; features are loaded by row mask so RAM stays bounded.
TRAIN_S1_SAMPLE = None
TRAIN_FIT_S1 = None if SMOKE else 600_000
VALID_FRAC = 0.3 if SMOKE else 0.1  # half calibrates, half tunes the decision layer (~110k entities each)
NEG_PER_POS = 5                 # easy-negative downsampling ratio (hard negatives are always kept)
HARD_NEG_RANK = 5               # negatives ranked <= this by pre-score count as hard
FEATURE_CHUNK = 2_000_000       # candidate pairs per feature shard
# Name-rarity features (v5): name-token IDF overlap and how many S1s / satellites share the exact name.
# ER_NAME_FEATS=0 reproduces the v4 feature set. Scoring always uses the feature list stored with the model.
NAME_FEATS = os.environ.get('ER_NAME_FEATS', '1') == '1'


def _xgb_device():
    """'cuda' when this XGBoost build has CUDA and a GPU answers, else 'cpu' (ER_DEVICE overrides)."""
    if os.environ.get('ER_DEVICE'):
        return os.environ['ER_DEVICE']
    try:
        import xgboost as xgb
        if not xgb.build_info().get('USE_CUDA'):
            return 'cpu'
        import numpy as np
        xgb.train({'device': 'cuda', 'tree_method': 'hist'}, xgb.DMatrix(np.zeros((2, 1)), label=[0, 1]), 1)
        return 'cuda'
    except Exception:
        return 'cpu'


# XGBoost on the GPU (a 4 GB RTX 3050 trains ~10x faster than LightGBM on 16 CPU threads here, and
# scoring ~100M test pairs through thousands of trees drops from hours to minutes).
XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='aucpr', tree_method='hist', device=_xgb_device(),
    learning_rate=0.3 if SMOKE else 0.06, grow_policy='lossguide', max_depth=0,
    max_leaves=31 if SMOKE else 255, min_child_weight=5 if SMOKE else 50, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=SEED, nthread=N_JOBS,
)
NUM_BOOST_ROUND = 60 if SMOKE else 4000
EARLY_STOPPING = 20 if SMOKE else 150

# Pseudo-labels for countries absent from train (s4_train.Pseudo): number of unseen-country test S1s whose
# confidently scored pairs (p >= PSEUDO_HI or <= PSEUDO_LO in a previous model's test scores) join the
# training rows. 0 disables it. Needs work/test/scores.parquet from an earlier full run.
PSEUDO_S1 = int(os.environ.get('ER_PSEUDO_S1', '0'))
PSEUDO_HI, PSEUDO_LO = 0.97, 0.03
REUSE_PASS1 = os.environ.get('ER_REUSE_PASS1') == '1'   # skip refitting pass 1 when its models exist (retries)

# ---- Stage 6: decision layer ----------------------------------------------------------------
MAX_MATCHES = 8                 # 99%+ of training clusters have <= 7 matches
TAU_GRID = [round(0.025 * i, 3) for i in range(2, 40)]


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
