# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Draft status: methodology, blocking, features and model are complete and pass an end-to-end
> smoke test. Sections marked **[pending full run]** are filled in once the pipeline has been
> trained on the full training set.

---

## 1. Executive Summary

We treat entity resolution as three linked problems, each tuned for F0.5:
1. **Recall-first blocking.** Five complementary keys, led by the address, cut ~10¹³ possible pairs
   to at most 40 candidates per entity.
2. **A calibrated two-pass LightGBM matcher** over ~50 language-agnostic similarity and
   competition features.
3. **A global decision layer.** It exploits a structural property we measured: no satellite record
   ever belongs to two entities. It also picks the number of matches per entity by maximising
   expected F0.5, instead of applying one global threshold.

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

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all within country):
  * **K1:** every address number × prefix of the two rarest address words.
  * **K2:** the two rarest address words, which survive component reordering.
  * **K3:** 8-character prefix of the core-name "squash", which handles domains and suffix drift.
  * **K4:** metaphone of the first two core-name tokens.
  * **K5:** char-trigram TF-IDF, reduced by truncated SVD and searched with HNSW (top 30), as a
    fuzzy safety net.

  Key values shared by more than 200 satellite records are ignored as too generic. The union is
  ranked by a cheap non-learned pre-score and capped at **40 candidates per entity**. That capped
  set is exactly what the model scores and what `candidate_pairs.tsv` contains.
- **Evidence from EDA on 20k sampled clusters:**
  * The plan's plain keys K1–K4 reach 92.2% pair completeness.
  * Refining K1 to key on every number and stripping legal tokens anywhere in the name raises
    that to **95.4%**.
  * The remaining misses are dominated by non-Latin names combined with noisy addresses. K5 and
    romanisation target exactly that group.
- **Candidate pairs generated:** [pending full run]
- **How we ensured true matches were not lost:** we measure pair completeness, reduction ratio
  and per-key "found only by this key" recall on held-out training entities, always against the
  **full** satellite pool (`s2_diagnostics.py`). Blocking is fixed before any model tuning.
  Pair completeness: [pending full run].

---

## 4. Matching Model

**Features used (~50, all float32, country-free):**
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
- **Other (competition context):**
  * blocking key bits, pre-score, ANN similarity
  * rank within the entity, margin to the entity's best candidate
  * rank among the S1 entities competing for the same satellite
  * in pass 2, the same statistics on the pass-1 model score, plus the expected cluster size

**Model type:** LightGBM binary classifier (MIT licence; well under 8B parameters).
* **Negatives.** Our own blocker's hard negatives. All negatives ranked in the top 5 by pre-score
  are kept, and easy negatives are down-sampled to about 5:1.
* **Splits** are grouped by S1 entity: train / calibration / tune.
* **Pass 1** is trained 2-fold out-of-fold, so the competition features computed on its score are
  not over-confident.
* **Pass 2** adds those features.
* **Calibration:** isotonic regression on the calibration entities.

**Threshold selection method:**
* The partition constraint gives each satellite to its highest-probability claimant only.
* Per entity, the cut k maximises E[F0.5(k)] ≈ 1.25·Σp₁..ₖ / (0.25·k + Σp), with
  E[F0.5(0)] = Π(1 − pᵢ).
* τ and the decision mode (plain threshold / + partition / + expected-F0.5) are grid-searched by
  direct macro-F0.5 on tune entities that were used for neither fitting nor calibration.
* At most 8 matches per entity.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [pending full run, validation on held-out tune entities]
- **Blocking ceiling (F0.5 of a perfect matcher on our candidates):** [pending full run]
- **Common false positives (wrong merges):**
  * *Expected from EDA and the smoke test:* businesses named after their town with a generic
    word (`La Teste-de-Buch Club` vs `La Teste-de-Buch Centre`). After stripping generic
    suffixes, their core names collide. Only address numbers separate them, so this needs
    checking on the France slice.
  * Same building, different unit.
- **Common false negatives (missed matches):**
  * *Expected from EDA:* rebranded names with truncated addresses.
  * Transliterated names whose satellite address lost its house number.
  * [pending full-run error analysis]

---

## 6. Conclusion

[pending full run] The measured partition property and the per-entity expected-F0.5 cut are the
two elements we expect to separate this pipeline from threshold-based baselines. Blocking recall
is monitored first because it caps everything else.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
* `src/`: stages `s0_ingest` → `s1_*` → `s2_blocking` → `s3_features` → `s4_train` / `s4_score`
  → `s6_decide` → `s7_output`, plus `evaluate.py` (macro-F0.5) and `smoke_test.py`.
* `README.md` and `requirements.txt` (pinned).

Entry points:
* `python -m src.run_all --split train`
* `python -m src.run_all --split test`, which writes both output files and runs the official
  validator.

### B. Additional Results

EDA figures (country mix, match-count distribution, similarity distributions, blocking-key pair
completeness) are in `notebooks/01_eda.ipynb`.

Full-run blocking diagnostics, feature importance and the τ sweep will be exported from
`work/train/blocking_diagnostics.json`, `work/models/importance.csv` and
`work/models/tau_sweep.csv`. [pending full run]
