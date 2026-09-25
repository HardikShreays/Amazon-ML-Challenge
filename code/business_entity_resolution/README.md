# Business Entity Resolution — reproducible pipeline

Blocking + two-pass LightGBM matcher + global decision layer. It links every Source-1 business
to its Source-2/3 duplicates. The design and its rationale are in
`../../Amazon_ML_2026_Entity_Resolution_Build_Plan.md`, and the evidence behind each decision is
in `../../notebooks/01_eda.ipynb`.

## 1. Environment

Python 3.13. Every dependency is MIT, BSD or Apache-2.0 licensed. No network access is needed at
any point.

```bash
uv venv .venv --python 3.13          # or: python3 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
```

## 2. Data location

By default the code reads `<repo>/student_resource/dataset/{train,test}/`. Point it somewhere
else with environment variables:

| variable | meaning | default |
|---|---|---|
| `ER_DATA_DIR` | folder containing `train/` and `test/` TSVs | `<repo>/student_resource/dataset` |
| `ER_WORK_DIR` | intermediate parquet, shards and models | `<repo>/work` |
| `ER_OUTPUT_DIR` | `matching_results.tsv` + `candidate_pairs.tsv` | `<repo>/output` |

## 3. Reproduce end to end (run from this folder)

```bash
python -m src.smoke_test             # ~30 s: tiny slice of the real data through every stage + validator
python -m src.run_all --split train  # ingest -> dictionaries -> signatures -> blocking -> diagnostics -> features -> train -> tune decision
python -m src.run_all --split test   # ingest -> dictionaries -> signatures -> blocking -> features -> score -> decide -> output -> validate
```

Each stage writes a checkpoint and is skipped when that checkpoint already exists.
Use `--from <stage>` to re-run a stage and everything after it, or `--force` to re-run everything.
The test run ends by calling `student_resource/utils/validate_submission.py` and fails loudly if
the files would be rejected.

## 4. Stages

| file | stage | output (in `ER_WORK_DIR/<split>/`) |
|---|---|---|
| `s0_ingest.py` | TSV → parquet, int32 row indices, ground truth → index pairs | `s1_raw`, `s23_raw`, `gt_pairs` parquet |
| `s1_normalise.py` | canonical text, romanisation of all 9 Indic scripts | — |
| `s1_dictionaries.py` | abbreviation + transliteration maps mined from train GT; legal/generic suffixes mined per split | `maps.json` (train), `legal.json` |
| `s1_signatures.py` | per-record keys, token sets, numbers, flags; address IDF | `s1_sig`, `s23_sig` parquet, `idf.pkl` |
| `s2_blocking.py` | K1–K5 union → cheap pre-score → top 40 per entity | `candidates.parquet`, `entities.npy` |
| `s2_diagnostics.py` | pair completeness, reduction ratio, per-key recall (train) | `blocking_diagnostics.json` |
| `s3_features.py` | ~50 pairwise features, parquet shards | `features/part-*.parquet` |
| `s4_train.py` | 2-fold OOF pass-1 model, pass-2 model with competition features, isotonic calibration | `work/models/*` |
| `s4_score.py` | streaming inference | `scores.parquet` |
| `s6_decide.py` | partition constraint + per-entity expected-F0.5; τ/mode tuned on held-out entities | `models/decision.json`, `matches.parquet` |
| `s7_output.py` | both submission TSVs + validator | `ER_OUTPUT_DIR/*.tsv` |
| `evaluate.py` | macro-F0.5 exactly as the challenge defines it | — |
| `run_all.py` | resumable entry point | — |
| `smoke_test.py` | end-to-end check on a tiny real-data slice | `<repo>/smoke/` |

The modules `s1_normalise`, `evaluate` and `s6_decide` each have a small self-check under
`__main__` (`python -m src.evaluate`), and the smoke test runs all three.

## 5. Memory plan (developed on an 8 GB laptop)

* Row indices are int32 and ids are encoded as int64. Strings are only touched at I/O, in the
  signature pass, and while features are computed.
* Blocking and features run **one country at a time**, and blocking also works in S1 chunks. Key
  lookups use a sorted hash index with `searchsorted`, so there are no pandas merges.
* Training pairs come from `TRAIN_S1_SAMPLE` = 300k sampled S1 entities. They are always blocked
  against the **full** satellite pool, so the model sees the real distractors.
* Test scoring streams feature shards. Only `(s1, s23, p1)` for the whole split is held in memory.

Tunables live in `src/config.py`.

## 6. Build the submission zip (from the repo root)

```bash
mkdir -p pkg/code && cp -r output pkg/ && cp -r code/business_entity_resolution pkg/code/ \
  && cp Documentation_template.md pkg/ && (cd pkg && zip -r ../TEAM_submission.zip . -x '*__pycache__*')
```

## 7. Known risks / next steps

* **Full-scale runtime is not yet measured.** So far only the smoke test has run. Watch K5
  (HNSW over ~6M US satellites) and feature extraction. Both have knobs in `config.py`
  (`SVD_DIM`, `ANN_K`, `FEATURE_CHUNK`).
* **France and generic words.** The suffix set mined per split contains generic nouns (`club`,
  `centre`, `groupe`) as well as legal forms. On French test data, names such as
  `La Teste-de-Buch Club` and `La Teste-de-Buch Centre` then share a core name, and only the
  address numbers separate them. Check the France slice of predictions after the first full run.
  If they over-merge, restrict the stripped set to tokens that actually *drift* between true pairs
  in the train ground truth.
* Stage 5 of the plan (cross-encoder rerank of the grey band) is not built. Add it only after a
  full submission exists.
