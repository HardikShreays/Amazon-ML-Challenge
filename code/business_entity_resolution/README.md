# Business Entity Resolution — reproducible pipeline

Blocking + two-pass XGBoost matcher (GPU) + cross-encoder re-scoring of the grey zone + global
decision layer. It links every Source-1 business to its Source-2/3 duplicates. The methodology is in
`Documentation_template.md` at the root of the submission zip.

The submitted `output/` files are **v6**: held-out macro F0.5 **0.9804** on 110k training entities
that were used for neither fitting nor calibration.

No external data, APIs or geocoding are used anywhere. Every dictionary is mined from the provided
TSVs. The only pretrained weights are `intfloat/multilingual-e5-base` (MIT, 278M parameters),
fine-tuned on the provided data.

## 1. Environment

Python 3.13. Every dependency is MIT, BSD or Apache-2.0 licensed. Stages 0–7 need no network
access. Stage 8 downloads the `intfloat/multilingual-e5-base` base weights from the Hugging Face
hub once; that is a model download, not a data lookup.

```bash
uv venv .venv --python 3.13          # or: python3 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
uv pip install -r requirements_gpu.txt   # stage 8 only (CUDA GPU)
```

## 2. Data location

`<root>` is two levels above this folder: the repo root, or the root of the unzipped submission.
By default the code reads `<root>/student_resource/dataset/{train,test}/` and writes
`<root>/output/`. Point it somewhere else with environment variables:

| variable | meaning | default |
|---|---|---|
| `ER_DATA_DIR` | folder containing `train/` and `test/` TSVs (organisers' file names) | `<root>/student_resource/dataset` |
| `ER_WORK_DIR` | intermediate parquet, shards and models | `<root>/work` |
| `ER_OUTPUT_DIR` | `matching_results.tsv` + `candidate_pairs.tsv` | `<root>/output` |

The organisers' validator is vendored unchanged as `src/validate_submission.py`, so this folder is
self-contained.

## 3. Reproduce end to end (run from this folder)

```bash
python -m src.smoke_test             # ~3 min: tiny slice of the real data through every stage + validator
python -m src.run_all --split train  # ingest -> dictionaries -> signatures -> blocking -> diagnostics -> features -> train -> tune decision
python -m src.run_all --split test   # ingest -> dictionaries -> signatures -> blocking -> features -> score -> decide -> output -> validate
```

Each stage writes a checkpoint and is skipped when that checkpoint already exists.
Use `--from <stage>` to re-run a stage and everything after it, or `--force` to re-run everything.
The test run ends by calling the vendored validator (`src/validate_submission.py`) and fails loudly
if the files would be rejected.

At this point `output/` holds the **v5** submission (GBDT only, tune macro F0.5 0.9765). Stage 8
turns it into the final **v6** submission.

Measured on the reference machine (section 5): train blocking ≈ 38 min, GPU training (both passes
+ calibration) ≈ 22 min, decision tuning ≈ 3 min, test scoring ≈ 12 min. Feature extraction is the
longest stage; it runs one country at a time and resumes from finished shards.

### Stage 8 — cross-encoder re-scoring of the grey zone (after `run_all --split test`) → final v6

```bash
python -m src.s8_ce_export
python src/s8_ce_gpu.py train --model intfloat/multilingual-e5-base --data $ER_WORK_DIR/ce/train.parquet        --out $ER_WORK_DIR/ce/e5_model --full --lr 3e-5 --bs 32 --max_len 96 --minutes 20
python src/s8_ce_gpu.py score --model intfloat/multilingual-e5-base --adapter $ER_WORK_DIR/ce/e5_model        --data $ER_WORK_DIR/ce/tune.parquet --out $ER_WORK_DIR/ce/tune_ce.parquet --max_len 96
python src/s8_ce_gpu.py score --model intfloat/multilingual-e5-base --adapter $ER_WORK_DIR/ce/e5_model        --data $ER_WORK_DIR/ce/test.parquet --out $ER_WORK_DIR/ce/test_ce.parquet --max_len 96
python -m src.s8_ce_merge
```

`s8_ce_gpu.py` runs on one GPU as plain `python`, or on several with `torchrun --nproc_per_node=N`.
Training is capped at 20 minutes (about 233k pairs seen on an RTX 3050). Scoring the 2.49M test
grey-zone pairs takes about 40 minutes. `s8_ce_merge` fits the stacker on TUNE, re-tunes τ, and
writes both TSVs and `ce_decision.json` to `ER_OUTPUT_DIR`. The final decision was: variant
`stack`, mode `partition`, τ = 0.725, τ_noaddr = 0.75.

`run_ce_aws.sh` runs the same stage 8b on a rented multi-GPU box with the Qwen3 + LoRA alternative.
It was not used for the final submission.

## 4. Stages

| file | stage | output (in `ER_WORK_DIR/<split>/`) |
|---|---|---|
| `s0_ingest.py` | TSV → parquet, int32 row indices, ground truth → index pairs | `s1_raw`, `s23_raw`, `gt_pairs` parquet |
| `s1_normalise.py` | canonical text, romanisation of all 9 Indic scripts | — |
| `s1_dictionaries.py` | abbreviation + transliteration maps mined from train GT; legal/generic suffixes mined per split | `maps.json` (train), `legal.json` |
| `s1_signatures.py` | per-record keys, token sets, numbers, flags; address IDF | `s1_sig`, `s23_sig` parquet, `idf.pkl` |
| `s2_blocking.py` | K1–K7 union → cheap pre-score (name-only when an address is empty) → top 40 per entity | `candidates.parquet`, `entities.npy` |
| `s2_diagnostics.py` | pair completeness, reduction ratio, per-key recall (train) | `blocking_diagnostics.json` |
| `s3_features.py` | 66 pairwise features incl. name rarity (`ER_NAME_FEATS=0` drops those 9), parquet shards; `augment()` adds the name columns to existing shards | `features/part-*.parquet`, `name_stats.pkl` |
| `s4_train.py` | XGBoost (CUDA if available): 2-fold OOF pass-1, pass-2 with competition features, isotonic calibration | `work/models/*` |
| `s4_score.py` | streaming inference, batched on the GPU | `scores.parquet` |
| `s6_decide.py` | partition constraint + per-entity expected-F0.5; τ/mode tuned on held-out entities | `models/decision.json`, `matches.parquet` |
| `s7_output.py` | both submission TSVs + validator | `ER_OUTPUT_DIR/*.tsv` |
| `s8_ce_export.py` | grey-zone pairs (GBDT p in [0.01, 0.99]) + fine-tuning pairs (TRAIN positives / hard negatives, CAL grey zone, France pseudo-labels) as raw text | `ce/{train,tune,test}.parquet` |
| `s8_ce_gpu.py` | cross-encoder: fine-tune `intfloat/multilingual-e5-base` (MIT; `--full`) or Qwen3 (Apache-2.0; LoRA), then score the grey zone. Standalone, needs `requirements_gpu.txt` | `ce/e5_model/`, `ce/{tune,test}_ce.parquet` |
| `s8_ce_merge.py` | stack GBDT + cross-encoder (logistic regression, fit on TUNE), re-tune τ, decide, write + validate | `ER_OUTPUT_DIR/*.tsv`, `ce_decision.json` |
| `evaluate.py` | macro-F0.5 exactly as the challenge defines it | — |
| `run_all.py` | resumable entry point | — |
| `smoke_test.py` | end-to-end check on a tiny real-data slice | `<root>/smoke/` |
| `fast_submit.py` | early fallback used for leaderboard v1/v3: pass-1 models truncated to N trees, no pass 2. Not part of the final path | `ER_OUTPUT_DIR/*.tsv` |
| `validate_submission.py` | the organisers' submission validator, vendored unchanged | — |

The modules `s1_normalise`, `evaluate` and `s6_decide` each have a small self-check under
`__main__` (`python -m src.evaluate`), and the smoke test runs all three.

## 5. Hardware and memory plan (Ryzen 7 4800H, 24 GB RAM, RTX 3050 4 GB)

* Row indices are int32 and ids are encoded as int64. Strings are only touched at I/O, in the
  signature pass, and while features are computed.
* Blocking and features run **one country at a time**, and blocking also works in S1 chunks. Key
  lookups use a sorted hash index with `searchsorted`, so there are no pandas merges.
* **Every** train S1 entity is blocked (`TRAIN_S1_SAMPLE = None`), exactly as in test, so each
  satellite meets all of its real competitors and the competition features / one-owner partition are
  learned under test conditions. The model is fit on `TRAIN_FIT_S1` = 600k sampled entities; feature
  shards are read with a row mask (`load_matrix(..., mask)`) so RAM stays below ~12 GB.
* XGBoost trains and predicts on the GPU (`device='cuda'`, predictions in 1M-row batches for a 4 GB
  card). Without CUDA it falls back to the CPU automatically; `ER_DEVICE=cpu` forces it.
* The test split uses the train split's mined dictionaries (`maps.json`) and models, so run
  `--split train` before `--split test`. Run one heavy stage at a time: two concurrent feature
  stages do not fit in 24 GB.

## 6. Build the submission zip (from the repo root)

```bash
mkdir -p pkg/code && cp -r output pkg/ && cp -r code/business_entity_resolution pkg/code/ \
  && cp Documentation_template.md pkg/ && (cd pkg && zip -r ../Algosheras_submission.zip . -x '*__pycache__*')
```

## 7. Known limitations

* **France has no labels.** Its suffix set keeps only unambiguous legal forms (`sarl`, `sas`,
  `eurl` ...), and its address abbreviations are mined without labels from near-certain test pairs.
  The held-out score is measured on US/India only, so the France score is an estimate.
* **Blocking caps recall at 0.972** of true training pairs (40 candidates per entity). The
  largest remaining loss is empty-address, name-only pairs whose name is shared by many businesses.
* **Stage 8 fine-tuning is time-capped (20 min on a 4 GB GPU).** The cross-encoder saw 233k of the
  749k exported pairs. A longer run or a larger Apache-2.0/MIT model (Qwen3 1.7B via
  `run_ce_aws.sh`) is the obvious next step.
