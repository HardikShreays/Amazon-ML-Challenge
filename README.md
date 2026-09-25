# Amazon ML Challenge 2026 — Business Entity Resolution

Link every Source-1 business record to its duplicates in Source 2 and Source 3. The score is
macro F0.5 per S1 entity, which weights precision twice as heavily as recall.

| path | what |
|---|---|
| `Amazon_ML_2026_Entity_Resolution_Build_Plan.md` | solution design (blocking → GBDT → global decision layer) |
| `notebooks/01_eda.ipynb` | executed EDA. Every section ends with a takeaway; §11 maps each finding to a design decision |
| `code/business_entity_resolution/` | the pipeline: `src/`, `README.md` (exact run steps), `requirements.txt` |
| `Documentation_template.md` | methodology write-up for the submission (results filled in after the full run) |
| `student_resource/` | organisers' starter kit (validator, template); the dataset is gitignored |

Quick start:

```bash
uv venv .venv --python 3.13 && uv pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution
../../.venv/bin/python -m src.smoke_test          # everything end to end on a tiny slice (~30 s)
../../.venv/bin/python -m src.run_all --split train
../../.venv/bin/python -m src.run_all --split test # -> output/matching_results.tsv (validated)
```
