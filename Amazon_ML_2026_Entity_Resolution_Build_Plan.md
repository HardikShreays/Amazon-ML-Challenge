# Amazon ML Challenge 2026 — Business Entity Resolution
## Solution Design, Working & Process Flow

**Author:** Shreyash Golhani
**Date:** 26 September 2026
**Status:** Design document — nothing built yet. This is the blueprint to build from.

---

## 0. TL;DR — The Whole Plan in One Page

We must link 1.73M Source-1 business records to their duplicates among 10.0M Source-2/Source-3
records, scored by **macro F0.5** (precision weighted 2x over recall).

Brute force is 17.3 trillion pairs. The entire game is:

```
   reduce 17.3 trillion pairs  ->  ~100 million candidates  ->  ~6 million confident links
        (BLOCKING)                     (GBDT MATCHER)              (GLOBAL DECISION LAYER)
         recall lever                  ranking lever                 precision lever
```

Four findings from the data drive every design decision below:

| # | Finding (measured) | Consequence for design |
|---|---|---|
| 1 | **No S2/S3 record is ever claimed by two different S1 entities** — 7,638,365 links, 7,638,365 distinct IDs, zero collisions | The truth is a **partition**. We can enforce a global "one owner per record" constraint. This is the single biggest precision lever available and most teams will miss it. |
| 2 | **Address is a stronger key than name.** Real positives include `Veohalo` and `Koronyx Co` matching S1 entities with completely different names but byte-identical addresses | Address must be the primary blocking key. A name-first pipeline silently loses these. |
| 3 | **Only 5.58% of S1 entities are singletons; the average entity has 3.46 matches** | Predicting "no match" is almost always wrong. Bias the decision layer toward emitting 2-4 matches, but keep a dedicated high-precision singleton path (each correct singleton is a free 1.0). |
| 4 | **Test contains ~15% France, which never appears in training** | No country-specific hardcoding anywhere. All features must be language-agnostic and validated on a held-out country. |

---

## 1. Problem Restated

Given three independent sources of business records (`entity_id`, `business_name`,
`business_address`, `country`), find for every Source-1 entity the set of Source-2 and
Source-3 records referring to the same real-world business.

* **Source 1** is the clean, deduplicated reference source.
* **Source 2 / Source 3** are noisy satellites.
* Output: one row per S1 entity, comma-separated matched IDs (empty if none).
* Metric: F0.5 computed **per S1 entity**, then averaged over **all** entities (singletons included).

### Why F0.5 changes the strategy

With beta = 0.5, precision counts twice as much as recall. Concretely, for an entity with
3 true matches:

| Prediction | Precision | Recall | F0.5 |
|---|---|---|---|
| 3 correct | 1.00 | 1.00 | **1.000** |
| 2 correct, 0 wrong | 1.00 | 0.67 | **0.909** |
| 3 correct, 1 wrong | 0.75 | 1.00 | **0.789** |
| 2 correct, 2 wrong | 0.50 | 0.67 | **0.526** |

**Dropping a true match costs far less than adding a false one.** A missed link costs ~0.09;
a false merge costs ~0.21. Every threshold in this pipeline is tuned with that asymmetry in mind.

---

## 2. Data Reality — What the EDA Actually Shows

### 2.1 Scale

| File | Rows | Size |
|---|---|---|
| train_source1.tsv | 2,206,821 | 210 MB |
| train_source2.tsv | 5,034,616 | 489 MB |
| train_source3.tsv | 5,285,603 | 504 MB |
| train_ground_truth.tsv | 2,206,821 | 127 MB |
| test_source1.tsv | 1,732,544 | 175 MB |
| test_source2.tsv | 4,887,273 | 509 MB |
| test_source3.tsv | 5,082,316 | 506 MB |

This is a **systems problem as much as an ML problem**. Nothing here fits in a naive pandas
`merge`. Everything must be streamed, chunked, and stored in compact integer/sparse form.

### 2.2 Label structure (full scan of ground truth)

```
  0 matches:   123,247 entities ( 5.58%)   <- singletons
  1 match  :   119,157 entities ( 5.40%)
  2 matches:   375,212 entities (17.00%)
  3 matches:   530,841 entities (24.05%)   <- mode
  4 matches:   484,115 entities (21.94%)
  5 matches:   321,957 entities (14.59%)
  6 matches:   164,868 entities ( 7.47%)
  7 matches:    63,968 entities ( 2.90%)
  8+ matches:   23,456 entities ( 1.06%)   (max observed = 11)
  ------------------------------------------------
  TOTAL S1 = 2,206,821   TOTAL links = 7,638,365   avg = 3.46
```

Of 10,320,219 S2+S3 training records, 7,638,365 are linked — so roughly **26% of S2/S3 records
are pure distractors** that match nothing. They exist specifically to generate false positives.

### 2.3 The partition property (the key insight)

```
total links               : 7,638,365
distinct S2/S3 ids        : 7,638,365
ids claimed by >1 entity  : 0  (0.0000%)
```

Every satellite record has **at most one owner**. This is a hard structural constraint that we
exploit in Stage 6. It converts independent per-pair decisions into a global assignment problem,
killing an entire class of false positives (the "two similar S1 entities both grab the same
S2 record" failure).

### 2.4 Field quality (300k-row samples)

| Source | Empty address | Non-ASCII name | Domain-style name | Countries (test) |
|---|---|---|---|---|
| Train S1 | 0.0% | 0.0% | 0.0% | US 60 / India 40 |
| Train S2 | 3.3% | 15.1% | 4.0% | US 60 / India 40 |
| Train S3 | 3.3% | 11.6% | 4.0% | US 60 / India 40 |
| Test S1 | 0.0% | 2.3% | 0.0% | India 47 / US 38 / **France 15** |
| Test S2 | 2.7% | 19.0% | 3.2% | India 47 / US 38 / **France 15** |
| Test S3 | 2.7% | 14.5% | 3.3% | India 47 / US 38 / **France 15** |

Source 1 is clean by construction. All noise lives in S2/S3. ~3% of satellite records have
**no address at all** — those must fall back to a name-only path.

### 2.5 The noise taxonomy (real examples pulled from matched clusters)

**Cluster S1-840021930** — `Segura Classic Armour Corp | 14243 157th Place, Renton, WA`
```
S2: seguraclassicarmour.com   | 14243 157TH PL, RENTON, WA
S3: Segura Classic [Armour]   | 14243 157nd Place, Renton, Washington
S3: Segura Classic Armour     | 14243 157nd Pl, Washington, Renton
```
-> name replaced by a **domain**; `Place`->`PL`; `157th`->`157nd` (typo); `WA`->`Washington`;
city/state **reordered**; stray `[ ]` brackets.

**Cluster S1-396556632** — `Choice Pioneer Defense | 314 Nagel Circle, Hendersonville, NC`
```
S2: choicepioneerdefense.com   | 314. NAGEL CIRCLE, HENDERSONVILLE, NC
S3: Veohalo                    | 314 Nagel Circle, Hendersonville, North Carolina
S3: Choice Pioneer Defense Inc.| 314 Nagel Cir, Hendersonville, North Carolina
```
-> **`Veohalo` is a true match with zero name overlap.** Address carries the entire signal.

**Cluster S1-950521576** — `Pediatric Partners LLC | 10816 Weston Drive, Carmel, IN`
```
S2: Pediatric  Partners LLC   | 10816 WESTON DR, CARMEL, IN
S3: Pediatric Parlters LLC    | #10816 Weston Drive, Carmel, Indiana
S3: PEDIATRIC PARTNERS LLC    | #10816 Weston Dr, Carmel, Indiana
S3: Pediatric                 | 10816 Weston Drive, Carmel, IN
```
-> character typo (`Parlters`), double spaces, `#` prefix, **truncated name**.

**Cluster S1-606489920** — `Express Advertising Associates | Saint Louis, Fl 0, MO, 9502 Port Drive`
```
S2: Express Advertising Associates Ltd | SAINT LOUIS, MO, 9502 PROT DR
S3: Express Advertising                | Saint Louis, 9502-9506 Port Drive, Missouri
S3: Koronyx Co                         | 9502 Port Drive, Fl 0, Saint Louis, MO
```
-> `PROT` for `PORT` (transposition), **address range** `9502-9506`, another zero-overlap name.

**Cluster S1-587557495** — `Crescent Electronics Private Limited | Tc 12/689/16 ... Thrissur, Kerala`
```
S2: CRESCENT [ELECTRONICS]              | TC 12/689/16 ... THRISSUR, Kerala
S2: CRESCENT ÉLECTRONICS-PRIVATE LIMITED| Kerala, TC 12/689/16 ... THRISSUR   (reordered)
S2: Crescent Electronics Private Limited| TC 12/689/16 ... THRISSUR, Kerala
```
-> spurious **accent** (`ÉLECTRONICS`), hyphenation, component reordering.

**Consolidated noise taxonomy:**

| Class | Examples seen |
|---|---|
| Legal suffix drift | Corp / Corporation / Inc / Ltd / LLC / Pvt / Private Limited / SARL / SAS / EURL / SASU / SCI |
| Name replaced | domain (`veohalo`-style rebrand), truncation (`Pediatric`), DBA |
| Character noise | typos, transpositions (`PROT`), spurious accents (`ÉLECTRONICS`, `Àmicale`), brackets, `&` vs `and`, doubled spaces |
| Script change | Devanagari (`राम मार्केटिंग प्राइवेट लिमिटेड`), Kannada (`ಕರ್ನಾಟಕ`) |
| Address abbreviation | Rd/Road, St/Street, Dr/Drive, Cir/Circle, Pl/Place, Ave/Avenue, R./Rue, Av/Avenue |
| State/region form | NC/North Carolina, IN/Indiana, WA/Washington, MH/Maharashtra, TN/Tamilnadu, UP |
| Component reorder | `City, State, Street` vs `Street, City, State` — extremely common |
| Missing components | no PIN/ZIP, no state, empty address (~3%) |
| Indian address idioms | `H.No`, `Door No`, `Plot No`, `KH NO`, `S/o`, `C/o`, `Near Metro Pillar No: 234`, `Opp.Rta Office` |
| French address idioms | `R.`/`Rue`, `Bd`/`Boulevard`, `Impasse`, `Allée`, `Chemin`, `bis`, département vs région |

### 2.6 The France problem

France is 15% of test and **0% of train**. Anything learned as "US pattern" or "India pattern"
will not transfer. Mitigations:

* No country-conditioned model branches; `country` enters only as an equality feature.
* Abbreviation dictionaries are **mined from the data itself**, not hand-written per country.
* Validation includes a **leave-one-country-out** fold (train on US, validate on India, and
  vice-versa) to measure transfer honestly before trusting the France slice.

---

## 3. Solution Architecture

```
                          ┌─────────────────────────────────────────┐
                          │  STAGE 0 — INGEST & NORMALISE           │
                          │  TSV -> parquet, int ids, canonical text│
                          └────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 1 — SIGNATURE EXTRACTION         │
                          │  numeric keys, token sets, char n-grams │
                          └────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 2 — MULTI-KEY BLOCKING           │
                          │  inverted indexes + TF-IDF top-K        │
                          │  17.3e12 -> ~1e8 pairs                  │
                          │  >>> emits candidate_pairs.tsv <<<      │
                          └────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 3 — PAIRWISE FEATURES (~60)      │
                          │  name / address / context / competition │
                          └────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 4 — GBDT MATCHER (LightGBM)      │
                          │  P(match | pair), hard-negative trained │
                          └────────────────┬────────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │  STAGE 5 — (OPTIONAL) CROSS-ENCODER RERANK    │
                   │  small multilingual model on the grey band    │
                   └───────────────────────┬───────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 6 — GLOBAL DECISION LAYER        │
                          │  partition constraint + expected-F0.5   │
                          │  >>> emits matching_results.tsv <<<     │
                          └────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────────────┐
                          │  STAGE 7 — VALIDATE & PACKAGE           │
                          └─────────────────────────────────────────┘
```

**Approach type:** Blocking + supervised pairwise classifier + global constrained assignment.
**Core innovation:** exploiting the measured partition property in a per-entity
expected-F0.5 decision layer, instead of a single global probability threshold.

---

## 4. Stage-by-Stage Working

### Stage 0 — Ingest & Normalise

**Goal:** turn 2.4 GB of noisy TSV into compact, canonical columnar data.

1. Read with an explicit tab separator (`sep="\t"`, `quoting=csv.QUOTE_NONE`, `dtype=str`,
   `keep_default_na=False`). Reading without `sep="\t"` silently yields one column — a known trap.
2. Map `entity_id` -> `int32` row index; keep the string form only for output. Saves ~8x memory.
3. Persist as **parquet** partitioned by `country`. Every later stage reads parquet, not TSV.

**Text canonicalisation (identical function for all three sources):**

```
Unicode NFKD  ->  strip combining marks (ÉLECTRONICS -> ELECTRONICS, Àmicale -> Amicale)
lowercase
&  ->  " and " ;  strip [ ] ( ) { } " ' . , # - / \
collapse whitespace
```

Two derived name forms, both retained:

* `name_tokens` — canonical token list, legal suffixes stripped to a separate field.
* `name_squash` — all non-alphanumerics removed: `"Segura Classic Armour Corp"` -> `seguraclassicarmourcorp`
  and `"seguraclassicarmour.com"` -> `seguraclassicarmourcom`.

> The squash form is what makes **domain-style names recoverable without any external
> word-segmentation**: the two strings above share a 19-character prefix. This single trick
> handles the ~4% domain-name records for free.

**Legal-suffix handling:** build the suffix list **empirically** — take the most frequent
trailing 1-3 token n-grams across all sources, per country. This auto-discovers `inc`, `llc`,
`corp`, `ltd`, `pvt`, `private limited`, and crucially the unseen French `sarl`, `sas`, `eurl`,
`sasu`, `sci` without hardcoding anything. Suffixes are stripped for the *core-name*
similarity features but kept as a separate categorical "suffix compatible?" feature.

**Script normalisation:** for Devanagari/Kannada/other non-Latin names, apply a deterministic
ISO-15919-style romanisation table (pure code, no external data — allowed) to produce a Latin
approximation. Additionally, **mine a transliteration dictionary from the training ground
truth itself**: within each cluster, align non-Latin names against Latin ones to learn that
`प्राइवेट लिमिटेड` == `private limited`, `प्रॉपर्टीज` == `properties`. This is learned from
provided data only — fully compliant with the fair-play rules.

---

### Stage 1 — Signature Extraction

For each record we precompute, once:

| Signature | Purpose |
|---|---|
| `num_tokens` — all numeric runs in the address (`14243`, `157`, `12`, `689`, `16`) | Primary blocking key; house numbers are near-invariant |
| `lead_num` — the first/longest numeric token | Highest-precision single key |
| `addr_tokens` — alphabetic address tokens after abbreviation folding | Secondary key + Jaccard features |
| `addr_rare` — the 3 highest-IDF address tokens | Distinctive blocking key (street/locality names) |
| `name_tokens`, `name_core`, `name_squash` | Name keys and features |
| `name_3gram`, `addr_3gram` — character trigram sets | TF-IDF cosine retrieval, typo-robust |
| `phonetic` — Double Metaphone of core name tokens | Catches `Parlters`/`Partners`, `PROT`/`PORT` |

**Abbreviation folding is learned, not hardcoded.** Procedure: for every token pair (short,
long) co-occurring in the same ground-truth cluster in the same positional slot, count
co-occurrence; keep pairs with high mutual information and where short is a prefix/subsequence
of long. This yields `dr->drive`, `pl->place`, `cir->circle`, `nc->north carolina`,
`mh->maharashtra`, and — because the mechanism is generic — will produce nothing harmful for
French, where we instead rely on trigram similarity (`r.`/`rue` is already a prefix match).

---

### Stage 2 — Multi-Key Blocking (candidate generation)

**This stage sets the recall ceiling. Everything downstream can only reduce recall.**
Target: **>= 98% of true links retained**, at **<= 60 candidates per S1 entity**.

We use a **union of complementary keys**, because each alone has a blind spot:

| # | Key | Formula | Catches | Blind spot |
|---|---|---|---|---|
| **K1** | Numeric-address anchor | `(country, lead_num, first_3_chars_of_top_IDF_addr_token)` | The dominant case: `14243` + `157`, `10816` + `wes` | empty addresses, number typos |
| **K2** | Rare-token address pair | `(country, sorted pair of top-2 IDF addr tokens)` | reordered addresses, missing house number | generic street names |
| **K3** | Name squash prefix | `(country, name_squash[:8])` | domain names, suffix drift | rebrands, truncation, leading typo |
| **K4** | Phonetic name | `(country, metaphone(name_core[:2 tokens]))` | typos, transliteration | zero-overlap names |
| **K5** | TF-IDF top-K retrieval | char-trigram cosine over `name + address`, top 30 per S1 | fuzzy everything, the safety net | cost |

**How K5 is computed at this scale.** Build a sparse char-trigram TF-IDF matrix per country
(scipy CSR, `float32`, ~10M rows x ~60k features). Then either

* sparse matrix product `S1_chunk @ S23.T` in chunks of ~20k S1 rows, keeping top-K per row
  (`sparse_dot_topn`), or
* an ANN index (`hnswlib`/`faiss`, both permissively licensed) over L2-normalised
  truncated-SVD embeddings for a ~10x speedup.

Chunked sparse top-K is the safer default; ANN is the optimisation if wall-clock becomes tight.

**Pruning.** Take the union of K1-K5, then **cheap pre-score** every candidate with a
non-learned scalar (IDF-weighted token overlap on address + trigram Jaccard on name) and keep
the **top 40 per S1**. This bounded set is exactly what gets written to `candidate_pairs.tsv`
and fed to the model — which is what the rules require ("the *last* blocking stage, whatever
your model actually runs inference over").

**Blocking diagnostics we must track on the validation split:**

```
Pair Completeness (recall ceiling) = true pairs kept / all true pairs      target >= 0.98
Reduction Ratio                    = 1 - (candidates / all possible pairs) target >= 0.999999
Candidates per S1 entity           = mean / p95 / max                      target ~40 / 40 / 40
```

If K1-K5 leave recall below 98%, **fix blocking before touching the model.** No classifier can
recover a pair it never sees.

---

### Stage 3 — Pairwise Feature Engineering (~60 features)

Computed for every surviving candidate pair. All features are **symmetric, language-agnostic,
and country-free** (country enters only as equality).

**A. Name similarity (~18)**
- Token Jaccard and containment on `name_core` (both directions — handles truncation `Pediatric`)
- TF-IDF cosine on word tokens; TF-IDF cosine on char 3-grams
- Normalised Levenshtein and Jaro-Winkler on `name_squash`
- **Longest common prefix length of `name_squash`** (the domain-name killer feature)
- Longest common substring ratio
- Acronym match (`IBM` vs `International Business Machines`)
- Double-Metaphone token overlap
- Token-sort ratio and token-set ratio (order-invariant)
- Legal-suffix compatibility (same / one-missing / conflicting)
- Flags: is-domain-style, is-non-Latin-script, name length delta, token count delta

**B. Address similarity (~22)**
- **Numeric token exact-set Jaccard** and **lead-number equality** — strongest single features
- Numeric near-match (edit distance 1 on digits: `157th`/`157nd`; range containment `9502` in `9502-9506`)
- IDF-weighted address token overlap (order-invariant — handles reordering)
- Address token Jaccard and containment
- TF-IDF char-trigram cosine on address
- Street-suffix agreement after abbreviation folding
- City / state-slot agreement after expansion, plus "state present in both?" flags
- Count of shared high-IDF tokens; max IDF among shared tokens
- Address length delta, token count delta
- **Empty-address flags** (one / both) — routes ~3% of records to the name-only regime

**C. Context & competition (~12)** — these turn independent pairs into a *ranked list*
- Candidate's rank within this S1 entity by pre-score, and by model score in a 2nd pass
- **Score margin vs. this entity's best candidate** and vs. its 2nd best
- Number of candidates this S1 entity has
- **Reverse rank**: this S1's rank among all S1 entities competing for this S2/S3 record
- **Mutual-best flag**: is this pair each other's top choice?
- Source indicator (S2 vs S3) and per-source count already selected
- Country equality

> Group C is what most teams omit. Entity resolution is not i.i.d. pair classification — a
> candidate scoring 0.7 when the runner-up scores 0.2 is very different from one scoring 0.7
> when three others also score 0.7. These features encode exactly that.

**Implementation note:** all of Stage 3 runs as a **vectorised, chunked, multiprocess** job
over candidate blocks stored as parquet. Never materialise 100M x 60 float64 in RAM — write
`float32` feature shards to disk (~24 GB total, less if cheap features are quantised).

---

### Stage 4 — GBDT Matcher

**Model: LightGBM binary classifier** (MIT licence, well under 8B params — fully compliant).

* **Why GBDT, not a neural net:** the signal is heterogeneous tabular similarity scores with
  sharp thresholds ("house numbers equal" is a step function). Trees model this natively,
  train on 100M rows in ~20 min on CPU, and are trivially interpretable for the error analysis
  the write-up requires.
* **Training data:** run Stage 2 blocking on the *training* split.
  * Positives = candidate pairs present in `train_ground_truth.tsv` (~7.5M after blocking losses)
  * Negatives = candidate pairs **not** in ground truth (~80M) — these are **hard negatives by
    construction**, since blocking already judged them plausible. Random negatives would be
    useless here.
  * Subsample negatives to roughly 1:5 positive:negative, keeping *all* high-pre-score
    negatives (the confusable ones) and downsampling the easy tail. Apply `scale_pos_weight`
    to keep probabilities calibrated.
* **Key hyperparameters:** `objective=binary`, `num_leaves` 128-256, `learning_rate` 0.05,
  `min_data_in_leaf` 200, `feature_fraction` 0.8, `bagging_fraction` 0.8, early stopping on
  validation AUC-PR (not AUC — the classes are imbalanced).
* **Calibration:** isotonic regression on a held-out slice, so the probabilities the Stage-6
  decision layer consumes are genuinely probabilities. **This matters** — expected-F0.5
  optimisation is only correct with calibrated inputs.
* **Grouping discipline:** all splits are **grouped by S1 entity**. A cluster must never
  straddle train and validation, or scores are optimistic.

**Two-pass scoring.** Run the model once to get raw scores, compute the Group-C competition
features from those scores, then run a second model that includes them. This is a cheap
stacking step worth a meaningful F0.5 gain.

---

### Stage 5 — Optional Cross-Encoder Rerank (only if time allows)

Reserve for the **grey band** (`0.3 < p < 0.7`), typically 5-10% of candidates:

* A small multilingual cross-encoder — e.g. `multilingual-e5-small` (MIT) or `LaBSE` (Apache 2.0),
  both far below 8B parameters and licence-compliant — fine-tuned on
  `"{name} [SEP] {address}"` pairs as binary classification.
* Chief value: **transliterated names** (Devanagari/Kannada) and **French**, where character
  features are weakest and a multilingual encoder generalises across scripts natively.
* Its output becomes one extra feature into Stage 6 (or a blended score), not a replacement.

**Build this only after Stages 0-6 are working end to end and submitted once.** The GBDT gets
us to a competitive score; this is incremental.

---

### Stage 6 — Global Decision Layer (where F0.5 is actually won)

Most teams stop at "threshold the probability at 0.5". That leaves a lot of score on the table.
Three mechanisms, applied in order:

**6a. Partition constraint (from Finding #1).**
Since no S2/S3 record legitimately belongs to two S1 entities, resolve every contested record
in favour of its highest-scoring S1 and drop the rest. Implement as a greedy pass over pairs
sorted by descending probability with a "claimed" bitset over S2/S3 ids — O(n log n), one pass.

```
sort all pairs by p desc
claimed = bitset(10M)
for (s1, s23, p) in pairs:
    if p >= tau and not claimed[s23]:
        accept(s1, s23); claimed[s23] = True
```

This is a direct, data-proven precision gain: every dropped duplicate claim was, with certainty,
a false positive.

**6b. Per-entity expected-F0.5 maximisation.**
Because the metric is macro-averaged *per entity*, the optimal cut is per entity, not global.
For each S1 entity, sort its surviving candidates by calibrated probability `p1 >= p2 >= ...`
and evaluate every prefix `k = 0, 1, 2, ..., K`:

```
E[TP(k)]   = sum(p_1..p_k)
E[|pred|]  = k
E[|truth|] = sum(p_1..p_K)                  # expected true cluster size
E[F0.5(k)] ~ 1.25 * E[TP] / (0.25 * E[|pred|] + E[|truth|])
choose k* = argmax_k E[F0.5(k)]
```

`k = 0` (predict singleton) is a legitimate candidate in that argmax — which is exactly how the
5.58% singletons earn their free 1.0 without us hand-tuning a separate rule. Because F0.5
penalises the denominator's `0.25 * |pred|` term, this naturally stops adding candidates once
probabilities drop, entity by entity.

**6c. Global threshold `tau` sweep.**
Tune the single remaining scalar by direct macro-F0.5 grid search on the validation split
(the metric is cheap to compute). Expect the optimum to land **well above 0.5** — likely
0.6-0.75 — precisely because of the precision weighting.

**Sanity guardrails before writing output:**
- Cap matches per entity at ~8 (99% of training clusters are <= 7; beyond that we are almost
  certainly merging distinct businesses).
- Require `country` equality unless the model is very confident.
- Every emitted ID must appear in that entity's candidate list (the validator warns otherwise).

---

### Stage 7 — Validation, Output & Packaging

**Validation protocol.** Hold out 200k S1 entities from training, **grouped** — but keep the
*entire* S2/S3 pool in the blocking index, so the held-out entities face the same 10M-record
haystack and the same distractors as the real test set. A naive split that also shrinks the
satellite pool makes blocking look far better than it is.

Additionally run a **leave-one-country-out** fold (train US -> validate India, and reverse) as
the only honest proxy we have for France transfer.

**Scoring.** Implement macro-F0.5 exactly as specified — per entity, averaged over **all**
entities including singletons, where a correctly predicted empty list scores 1.0 and any
prediction on a true singleton scores 0.0.

**Outputs** (tab-separated, UTF-8, `df.to_csv(path, sep="\t", index=False)`):
- `output/matching_results.tsv` — one row per test S1 entity (all 1,732,544), the leaderboard file
- `output/candidate_pairs.tsv` — the Stage-2 final candidate set

**Always run the provided validator before uploading:**
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Submissions are capped at 5/day — a format rejection is a wasted submission.

**Submission package:**
```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in)
```

---

## 5. Process Flow — End to End

```
 dataset/train/*.tsv                                    dataset/test/*.tsv
        │                                                       │
        ▼                                                       ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ S0  normalise -> parquet (shared code path, train and test)         │
 └─────────────────────────────────────────────────────────────────────┘
        │                                                       │
        ▼                                                       │
 ┌──────────────────────────┐                                   │
 │ S1  signatures + learned │  abbreviation map, suffix list,    │
 │     dictionaries         │  transliteration map, IDF tables   │
 └──────────┬───────────────┘        (fit on TRAIN only) ────────┤
            │                                                   │
            ▼                                                   ▼
 ┌──────────────────────────┐                     ┌──────────────────────────┐
 │ S2  blocking (train)     │                     │ S2  blocking (test)      │
 │  -> ~110M candidates     │                     │  -> ~70M candidates      │
 │  measure recall ceiling  │                     │  -> candidate_pairs.tsv  │
 └──────────┬───────────────┘                     └──────────┬───────────────┘
            ▼                                                ▼
 ┌──────────────────────────┐                     ┌──────────────────────────┐
 │ S3  features + labels    │                     │ S3  features             │
 └──────────┬───────────────┘                     └──────────┬───────────────┘
            ▼                                                │
 ┌──────────────────────────┐                                │
 │ S4  train LightGBM       │───────── model.txt ───────────►│
 │     + isotonic calibrate │                                ▼
 └──────────┬───────────────┘                     ┌──────────────────────────┐
            │                                     │ S4' score test pairs     │
            ▼                                     └──────────┬───────────────┘
 ┌──────────────────────────┐                                │
 │ S6  tune tau + decision  │──── tau, caps ────────────────►│
 │     on held-out 200k     │                                ▼
 └──────────────────────────┘                     ┌──────────────────────────┐
                                                  │ S6' partition constraint │
                                                  │     + expected-F0.5 cut  │
                                                  └──────────┬───────────────┘
                                                             ▼
                                                  ┌──────────────────────────┐
                                                  │ S7  validate + submit    │
                                                  └──────────────────────────┘
```

**Leakage discipline:** every dictionary, IDF table, and threshold is **fit on training data
only** and then applied unchanged to test. IDF tables are the easy mistake — computing them
over train+test combined is a subtle leak; compute on train, apply to test.

---

## 6. Compute Plan

Target: full test inference on a single laptop/workstation in a few hours.

| Stage | Technique | Est. time | Peak RAM |
|---|---|---|---|
| S0 normalise | streamed chunks, pyarrow | ~15 min | 4 GB |
| S1 signatures | multiprocess map over chunks | ~25 min | 6 GB |
| S2 blocking | inverted index + chunked sparse top-K | 1.5-3 h | 12-16 GB |
| S3 features | vectorised, multiprocess, float32 shards | ~1 h | 8 GB |
| S4 train | LightGBM, ~90M rows subsampled | ~25 min | 12 GB |
| S4' score | batched predict | ~20 min | 4 GB |
| S6 decision | sort + greedy bitset | ~10 min | 6 GB |

**Engineering rules that keep this tractable:**
1. `int32` entity indices everywhere; strings only at I/O boundaries.
2. Parquet + `float32` shards; never a single monolithic DataFrame.
3. Process **per country** — it partitions the work naturally and bounds every index.
4. Checkpoint after every stage; the pipeline must be resumable (3-day competition window).
5. `multiprocessing.Pool` over chunk files; the work is embarrassingly parallel.

**Stack:** Python 3.11, pandas, pyarrow, numpy, scipy, scikit-learn, LightGBM, rapidfuzz
(fast Levenshtein/Jaro-Winkler), jellyfish (Metaphone), optionally hnswlib/faiss and
sentence-transformers. All permissively licensed.

---

## 7. Build Order (3-day window)

| Priority | Deliverable | Why this order |
|---|---|---|
| **P0** | S0 + S1 + a K1/K3-only blocker + 12 core features + LightGBM + global threshold; **submit** | Gets a real leaderboard number on day 1. An end-to-end baseline beats a perfect half-pipeline. |
| **P1** | Full K1-K5 blocking; measure and fix pair completeness to >= 98% | Raises the recall ceiling — every later gain is capped by this. |
| **P2** | Full ~60-feature set + two-pass competition features | Biggest single model-quality jump. |
| **P3** | **Stage 6 decision layer** (partition constraint + expected-F0.5) | Cheapest large F0.5 gain in the whole plan. Do not skip. |
| **P4** | Learned abbreviation/transliteration dictionaries; leave-one-country-out check | Protects the 15% France slice and the non-Latin names. |
| **P5** | Cross-encoder rerank on the grey band | Incremental; only with time to spare. |

**Submit early and often** — 5 uploads/day, and public-vs-private leaderboard divergence is
best detected by keeping the decision layer from overfitting the public slice.

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Blocking recall ceiling too low | Hard cap on final score | Measure pair completeness *first*; 5 complementary keys; K5 TF-IDF as safety net |
| France generalisation failure | 15% of test degraded | No country branches; learned (not hardcoded) dictionaries; leave-one-country-out validation |
| Transliterated names (Devanagari/Kannada) | ~15% of S2 names | Romanisation + dictionary mined from ground truth; address carries these pairs anyway |
| Empty addresses (~3%) | Address features go blind | Explicit empty flags; name-only regime the model learns separately |
| Over-merging near-identical businesses (same plaza, different unit) | Direct F0.5 loss, 2x weighted | Partition constraint; per-entity expected-F0.5; cap at 8 matches |
| Memory blow-up in blocking | Pipeline won't finish | Per-country processing; hard cap 40 candidates/entity; disk-backed shards |
| Format rejection | Wasted submission | Run `validate_submission.py` every time; `sep="\t"`, UTF-8, all 1,732,544 rows present |
| Public/private leaderboard gap | Final ranking drop | Tune on our own held-out split, not the public leaderboard |

---

## 9. Fair-Play Compliance

The rules **strictly prohibit** external data lookup — no entity-resolution APIs, no business
registries, no geocoding services, no internet augmentation. This design is compliant by
construction:

* Every dictionary (abbreviations, legal suffixes, transliterations, IDF) is **mined from the
  provided training files**.
* Romanisation uses a deterministic in-code character table, not a lookup service.
* Any pretrained encoder used in Stage 5 is a general-purpose multilingual language model
  (MIT/Apache 2.0, < 8B params), not an entity database — and is fine-tuned solely on the
  provided training data.
* No network calls anywhere in the pipeline.

---

## 10. Expected Outcome

| Checkpoint | Expected macro F0.5 |
|---|---|
| P0 baseline (simple blocking + few features + global threshold) | ~0.55 - 0.65 |
| P2 (full blocking + full features) | ~0.72 - 0.80 |
| P3 (+ decision layer) | ~0.80 - 0.86 |
| P5 (+ cross-encoder rerank) | marginal, +0.01 - 0.02 |

These are ranges, not promises — they exist to set expectations for whether a stage is
behaving. The decision layer (P3) is where this design expects to separate from a conventional
threshold-based pipeline, and it costs the least to build.

---

## 11. Repository Layout to Build

```
code/business_entity_resolution/
├── README.md                  # exact end-to-end reproduction steps
├── requirements.txt           # pinned versions
└── src/
    ├── config.py              # paths, constants, tunables
    ├── s0_ingest.py           # TSV -> parquet, id mapping
    ├── s1_normalise.py        # canonicalisation, romanisation
    ├── s1_dictionaries.py     # mine abbreviations/suffixes/translit from train
    ├── s1_signatures.py       # numeric keys, token sets, n-grams, phonetics
    ├── s2_blocking.py         # K1-K5 + pre-score prune -> candidate_pairs
    ├── s2_diagnostics.py      # pair completeness, reduction ratio
    ├── s3_features.py         # the ~60 pairwise features
    ├── s4_train.py            # LightGBM + isotonic calibration
    ├── s4_score.py            # batched inference
    ├── s6_decide.py           # partition constraint + expected-F0.5 + tau
    ├── s7_output.py           # write both TSVs
    └── evaluate.py            # macro-F0.5 scorer for our own splits
```

Entry point: `python -m src.run_all --split test`, resumable stage by stage.

---

## 12. Summary

The winning formula for this challenge is not an exotic model — it is:

1. **Blocking that does not lose true pairs** (measure pair completeness before anything else).
2. **Address-first features**, because names are sometimes replaced entirely.
3. **A calibrated GBDT** over ~60 similarity + competition features.
4. **A global decision layer** that exploits the measured partition property and optimises
   expected F0.5 per entity rather than thresholding globally.

Items 1 and 4 are where this plan differs most from a textbook entity-resolution pipeline,
and they are where the score will come from.
