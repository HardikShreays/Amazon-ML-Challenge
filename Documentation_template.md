# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Algosheras
**Team Members:** Shreyas Golhani, Utkarsh Jain, Vaageesh Kumar Singh, Hardik Shreyas
**Submission Date:** 27 September 2026

> Final pipeline (v6), trained on the full training set. Validation numbers are macro F0.5 on
> 110,185 held-out training entities used for neither fitting nor calibration.
> **Final held-out macro F0.5: 0.9804.**

---

## 1. Executive Summary

We treat entity resolution as three linked problems, each tuned for F0.5:
1. **Recall-first blocking.** Seven complementary keys, led by the address, cut ~10¹³ possible pairs
   to at most 40 candidates per entity while keeping **97.2%** of true pairs.
2. **A calibrated two-pass XGBoost matcher (trained on the GPU)** over 66 language-agnostic
   similarity, name-rarity and competition features, plus 7 score-context features in pass 2.
3. **A fine-tuned multilingual cross-encoder** (`intfloat/multilingual-e5-base`, MIT, 278M
   parameters). It re-reads only the 3.6% of pairs where the GBDT is unsure and is stacked with it.
4. **A global decision layer.** It exploits a structural property we measured: no satellite record
   ever belongs to two entities. It also picks the number of matches per entity by maximising
   expected F0.5, instead of applying one global threshold.

Held-out macro F0.5 went from 0.9575 (first full pipeline) to **0.9804** (final).

---

## 2. Methodology

### 2.1 Problem Analysis

All findings below are measured in `notebooks/01_eda.ipynb`.

* **Scale.** 2.2M S1 entities and 10.3M S2/S3 records in train; 1.7M S1 and 10.0M S2/S3 in test.
* **Labels.** Only 5.58% of S1 entities are singletons. The mean is 3.46 matches (max 11), and
  77.6% of entities have 2–5 matches.
* **Partition property.** 7,638,365 links point to 7,638,365 distinct satellite ids. Zero
  satellite records have two owners.
* **Distractors.** 26.6% of S2 and 25.4% of S3 records match nothing.
* **Noise.**
  * 3.3% of satellite addresses are empty.
  * 11–19% of satellite names are in one of **9 Indic scripts**.
  * About 3% of names are domains, about 2% have junk prefixes, and 19% of S2 names are ALL-CAPS.
  * About 20% of addresses start with city/state rather than the street, so components are
    reordered.
  * Legal suffixes drift or move to the front of the name (`Pvt Spd Infosys Limited`).
  * Satellites inject extra numbers (`NO ##135 2804`).
* **Signal strength.** Numeric-token Jaccard is 0.70 on true pairs and 0.007 on random pairs.
  9% of true pairs have almost no name similarity (rebrands, transliterations), and 91% of those
  are recoverable through the address.
* **France is 15% of test and absent from train.** Nothing in the pipeline can be country-specific.
  French address abbreviations are therefore mined *without labels* from the test data itself (§2.2).

### 2.2 Solution Strategy

**Approach type:** Blocking + supervised pairwise GBDT + global constrained assignment.
**Core innovation:** a decision layer that enforces the measured one-owner-per-record partition
and cuts each entity's ranked candidate list at the size that maximises expected F0.5.
Predicting no match at all (k = 0) is one of the options, so singletons can be predicted.

Text canonicalisation is identical for all sources and splits:
* NFKD, then strip Latin accents only.
* One offset-keyed table romanises every Indic Unicode block. They share a layout, so one table
  covers Devanagari, Tamil, Kannada and the rest.
* Lowercase, map `&` to `and`, drop punctuation, join initials (`L.L.C.` → `llc`).

All dictionaries are **mined from the provided data**:
* **Abbreviations** from train ground-truth pairs: `dr→drive`, `mh→maharashtra`, `nc→northcarolina`.
* **Transliterations** from train ground-truth pairs: `praivet→private`, `dilli→delhi`.
* **Legal/generic suffixes** from the trailing-token frequencies of each split's S1 names. No
  labels are used, and this is how French `sarl`/`sas`/`eurl` appear on test.
* **Abbreviations for a country unseen in train (France)**, mined without labels. Test
  S1/satellite pairs with an identical squashed name (unique among the country's S1s) and the same
  house number are near-certain matches (367k pairs). They replace the ground truth in the same
  miner: `av→avenue`, `bd→boulevard`, `st→saint`, `imp→impasse`, `rte→route`, `ch→chemin`. The
  US/India map is *not* applied to France, because it read the stopwords `de la` as
  `delaware louisiana`.
* House numbers lose leading zeros (`0040` = `40`, `00722` = `722`).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all within country):
  * **K1:** every address number × prefix of the two rarest address words.
  * **K2:** the two rarest address words, which survive component reordering.
  * **K3:** 8-character prefix of the core-name "squash", which handles domains and suffix drift.
  * **K4:** metaphone of the first two core-name tokens.
  * **K5:** char-trigram TF-IDF, reduced by truncated SVD and searched with HNSW (top 30), as a
    fuzzy safety net.
  * **K6:** each of the first two core-name tokens × each of the first three house numbers.
  * **K7:** the whole core-name squash. K3's 8-character prefix is too generic for common first
    words: `heritage…` blocks exceed the size limit, but `heritagequalityhorse` does not.

  Key values shared by more than 200 satellite records are ignored as too generic. The union is
  ranked by a cheap non-learned pre-score and capped at **40 candidates per entity**. When either
  address is empty, the pair is ranked on name similarity alone, because under the name+address
  blend such pairs could never make the cap. That capped set is exactly what the model scores and
  what `candidate_pairs.tsv` contains.
- **How K6, K7 and the empty-address rule were found:** we rebuilt the *uncapped* key union for
  8,000 held-out entities.
  * The union already held 95.3% of true pairs, but the cap kept only 92.6%.
  * 92% of the lost pairs belonged to entities that hit the cap, and most had an empty satellite
    address.
  * The empty-address rule alone lifts recall at 40 candidates to 95.2%.
  * Among pairs no key produced, K6 catches 33% and K7 13% at the same block-size limit.
- **Evidence from EDA on 20k sampled clusters:**
  * The plan's plain keys K1–K4 reach 92.2% pair completeness.
  * Refining K1 to key on every number and stripping legal tokens anywhere in the name raises
    that to **95.4%**.
  * The remaining misses are dominated by non-Latin names combined with noisy addresses. K5 and
    romanisation target exactly that group.
- **Candidate pairs generated:**
  * train: 87,445,609 (2,206,821 S1 entities, 39.6 per entity)
  * test: 68,655,650 (1,732,544 S1 entities)
  * reduction ratio 0.999996
- **How we ensured true matches were not lost:** we measure pair completeness, reduction ratio
  and per-key "found only by this key" recall on held-out training entities, always against the
  **full** satellite pool (`s2_diagnostics.py`). Blocking is fixed before any model tuning.
  Pair completeness on the full training set is **0.972** (US 0.982, India 0.956), up from 0.929
  with K1–K5 and the plain pre-score. Share of true pairs found by only one key: K6 1.0%, K2 1.0%,
  K1 0.7%, K5 0.7%.

---

## 4. Matching Model

**Features used (66 in pass 1, plus 7 score-context features in pass 2; all float32, country-free):**
- **Name features:**
  * rapidfuzz ratio, partial, token-sort and token-set on canonical and core names
  * on the squash form: Jaro-Winkler, LCS, and the common-prefix length and fraction (catches
    domain-style names)
  * token Jaccard and containment, acronym match, legal-suffix compatibility, length deltas,
    domain and non-Latin flags
- **Address features:**
  * order-invariant token ratios
  * numeric-set Jaccard and containment, lead-number equality, near-miss (`157th` vs `157nd`)
  * IDF-weighted overlap in both directions, and the maximum shared IDF
  * empty-address flag, so the model learns a name-only regime
- **Name-rarity features:** a name-only match on a unique name is near-certain, but on a name shared
  by dozens of businesses it is a coin flip. The features are:
  * IDF (over the split's core names) of the shared name tokens and of the unmatched ones
  * how many S1s and satellites of the country carry exactly this squashed name

  Added in v5, they raise held-out macro F0.5 from 0.9713 to 0.9765.
- **Other (competition context):**
  * blocking key bits, pre-score, ANN similarity
  * rank within the entity, margin to the entity's best candidate
  * rank among the S1 entities competing for the same satellite
  * in pass 2, the same statistics on the pass-1 model score, plus the expected cluster size

**Model type:** XGBoost binary classifier (Apache-2.0; well under 8B parameters).
* Trained and run on an RTX 3050 (4 GB) with `device='cuda'`, with a CPU fallback.
* Leaf-wise trees with 255 leaves, learning rate 0.06, early stopping on validation average
  precision.
* **Every train S1 entity is blocked**, as in test. With a sample, each satellite saw only ~14% of
  its real competitors, so the competition features and the one-owner partition were learned in a
  far sparser world than test. After the fix, `p1_rev_rank` (the rank of this S1 among all S1s
  claiming the satellite) is the second most important feature.
* The model is fit on 600k sampled entities.
* **Negatives.** Our own blocker's hard negatives. All negatives ranked in the top 5 by pre-score
  are kept, and easy negatives are down-sampled to about 5:1.
* **Splits** are grouped by S1 entity: train / calibration / tune.
* **Pass 1** is trained 2-fold out-of-fold, so the competition features computed on its score are
  not over-confident.
* **Pass 2** adds those features.
* **Calibration:** isotonic regression on the calibration entities.

**Stage 2 matcher: cross-encoder on the grey zone (v6).**
* On TUNE, 98% of the GBDT's false negatives and 94% of its false positives have p in [0.01, 0.99].
  That band is only 3.6% of pairs (2.49M of 68.7M on test), so only it is re-read by a transformer.
* Model: `intfloat/multilingual-e5-base` (MIT, 278M parameters, multilingual, so it covers French
  and Indic scripts). It is fine-tuned as a binary pair classifier on the raw text
  `"<country> / A: <name> ; <address> / B: <name> ; <address>"`, with the 250k-token embedding
  table frozen, bf16, 20 minutes on an RTX 3050 (233k pairs seen).
* Fine-tuning pairs (749k): TRAIN-role positives with their top-3 pre-score negatives, every
  calibration-split grey-zone pair, and 120k **France pseudo-labels**. These are test pairs the
  GBDT is sure of (p > 0.995 and the satellite's top claimant, or a hard candidate with
  p < 0.003). They teach French surface forms without external data, and they lie outside the
  grey zone, so no pair is both trained on and re-scored.
* Stacking: logistic regression on [logit p_GBDT, cross-encoder logit, address-missing flag], fit
  on TUNE and cross-fitted by entity for the reported score. Grey-zone pair accuracy goes from
  0.909 (GBDT) to 0.930 (stack). The cross-encoder alone (0.898) is weaker than the GBDT; the
  gain comes from their errors being different.
* Code: `src/s8_ce_export.py`, `src/s8_ce_gpu.py` (also supports Qwen3 + LoRA on multi-GPU),
  `src/s8_ce_merge.py`.

**Threshold selection method:**
* The partition constraint gives each satellite to its highest-probability claimant only.
* Per entity, the cut k maximises E[F0.5(k)] ≈ 1.25·Σp₁..ₖ / (0.25·k + Σp), with
  E[F0.5(0)] = Π(1 − pᵢ).
* τ and the decision mode (plain threshold / + partition / + expected-F0.5) are grid-searched by
  direct macro-F0.5 on tune entities that were used for neither fitting nor calibration.
* At most 8 matches per entity. Scores within 1e-4 are ties, resolved towards `partition`.
* Chosen setting: partition, τ = 0.675, and τ = 0.70 when either address is empty (v5).
  After stacking (v6): τ = 0.725, and 0.75 when either address is empty.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro) on held-out tune entities:**

  | version | change | tune macro F0.5 | blocking ceiling |
  |---|---|---|---|
  | v2 | K1–K5, sampled train, LightGBM | 0.9575 | 0.9756 |
  | v4 | + K6/K7, empty-address pre-score, full-population train, GPU XGBoost | 0.9713 | 0.9908 |
  | v5 | + name-rarity features | 0.9765 | 0.9908 |
  | v6 | + cross-encoder (multilingual-e5-base) re-scoring the GBDT grey zone, stacked | **0.9804** | 0.9908 |

- **Where the remaining loss sits (v4 breakdown):**
  * Blocking misses 2.8% of true pairs.
  * About 45% of the matcher's missed pairs are name-only (an address is missing). That is the
    slice v5 targets.
- **Common false positives (wrong merges):**
  * *Expected from EDA and the smoke test:* businesses named after their town with a generic
    word (`La Teste-de-Buch Club` vs `La Teste-de-Buch Centre`). After stripping generic
    suffixes, their core names collide. Only address numbers separate them, so this needs
    checking on the France slice.
  * Same building, different unit.
- **Common false negatives (missed matches):**
  * *Expected from EDA:* rebranded names with truncated addresses.
  * Transliterated names whose satellite address lost its house number.
  * House-number noise that cannot be separated using the text alone: 1418 vs 1406-B is a true
    pair, 2723 vs 2724 is not.

---

## 6. Conclusion

Measuring *where* F0.5 was lost drove every improvement. The fixes, in order:
1. Blocking capped the score first: the candidate cap silently dropped empty-address satellites.
2. Then the train/test mismatch in the competition features.
3. Then name-only ambiguity.

Together they took held-out macro F0.5 from 0.9575 to 0.9765. A cross-encoder aimed only at the
GBDT's grey zone then added the last step, to **0.9804**. On its own it is no better than the GBDT
(0.9764), but its errors differ, so the stack gains. GPU training and scoring made
full-population training and full-model inference fit the challenge window: training went from
2.5 h to 23 min, and test scoring takes about 12 min.

**Compliance.** No external databases, APIs, geocoding or web data are used. Every dictionary is
mined from the provided TSVs, and France is handled without labels. All libraries are
MIT/BSD/Apache-2.0. The only pretrained model is `intfloat/multilingual-e5-base` (MIT, 278M
parameters, well under the 8B limit), fine-tuned only on the provided data. The GBDT is XGBoost
(Apache-2.0).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
* `src/`: stages `s0_ingest` → `s1_*` → `s2_blocking` → `s3_features` → `s4_train` / `s4_score`
  → `s6_decide` → `s7_output` → `s8_ce_export` → `s8_ce_gpu` → `s8_ce_merge`, plus `evaluate.py`
  (macro-F0.5), `smoke_test.py` and the organisers' `validate_submission.py` (vendored unchanged).
* `README.md`, `requirements.txt` and `requirements_gpu.txt` (stage 8), all pinned.

Entry points:
* `python -m src.run_all --split train`
* `python -m src.run_all --split test`, which writes the v5 output files and runs the validator.
* Stage 8 (`s8_ce_export` → `s8_ce_gpu.py train/score` → `s8_ce_merge`), which writes the final v6
  files. The exact commands are in the code README.

### B. Submission Version History

Every version was validated with the official validator (`PASS`). "Tune" is macro F0.5 on the
110k held-out training entities.

| version | what changed | tune macro F0.5 |
|---|---|---|
| v1 | fast fallback: pass-1 models cut to 100 trees, one global threshold | 0.9486 |
| v2 | full two-pass LightGBM + isotonic calibration + partition decision | 0.9575 |
| v3 | France suffix fix (legal forms only), empty-address threshold | 0.9556 |
| v4 | K6/K7 keys, empty-address pre-score, full-population training, GPU XGBoost | 0.9713 |
| v5 | + 9 name-rarity features | 0.9765 |
| **v6 (final)** | + multilingual-e5-base cross-encoder on the grey zone, stacked with the GBDT | **0.9804** |

Final decision (`ce_decision.json`): variant `stack`, mode `partition`, τ = 0.725 (0.75 when an
address is empty). The final output has 5,705,925 matches, 3.29 per S1 entity, and 105,381
predicted singletons out of 1,732,544 test entities.

### C. Additional Results

EDA figures (country mix, match-count distribution, similarity distributions, blocking-key pair
completeness) are in `notebooks/01_eda.ipynb` in the project repository.

Running the pipeline writes the full-run blocking diagnostics, feature importance and the τ sweep
to `ER_WORK_DIR/train/blocking_diagnostics.json`, `ER_WORK_DIR/models/importance.csv` and
`ER_WORK_DIR/models/tau_sweep.csv`.
