"""Compare submission versions: matches per S1 and multi-owner satellites by country, plus France
pairs dropped / added between two versions (with names and addresses, for eyeballing).

    python analysis/compare_versions.py v1_fast100_threshold v2_full_partition v3_francefix_600
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'student_resource' / 'dataset' / 'test'
N = 12


def pairs(version):
    """(s1_id, s23_id) matched pairs of one submission."""
    m = pd.read_csv(ROOT / 'submissions' / version / 'matching_results.tsv', sep='\t', quoting=3, dtype=str,
                    keep_default_na=False)
    m = m.assign(s23_id=m.matched_entity_ids.str.split(',')).explode('s23_id')
    return m[m.s23_id != ''][['source1_entity_id', 's23_id']].rename(columns={'source1_entity_id': 's1_id'})


versions = sys.argv[1:]
s1 = pq.read_table(ROOT / 'work' / 'test' / 's1_raw.parquet').to_pandas().rename(
    columns={'entity_id': 's1_id', 'business_name': 's_name', 'business_address': 's_addr'})
country = s1.set_index('s1_id').country
n_s1 = s1.country.value_counts()
got = {v: pairs(v) for v in versions}

rows = []
for v, p in got.items():
    p = p.assign(country=country.reindex(p.s1_id).to_numpy())
    multi = p.groupby('s23_id').s1_id.transform('size') >= 2
    for c in sorted(n_s1.index):
        g, mg = p[p.country == c], multi[p.country == c]
        rows.append({'version': v, 'country': c, 'matches/S1': round(len(g) / n_s1[c], 3),
                     'empty S1s': int(n_s1[c] - g.s1_id.nunique()), 'multi-owner pairs': int(mg.sum())})
print(pd.DataFrame(rows).to_string(index=False))

if len(versions) >= 2:
    a, b = versions[0], versions[-1]
    key = lambda d: set(zip(d.s1_id, d.s23_id))  # noqa: E731
    ka, kb = key(got[a]), key(got[b])
    s23 = pq.read_table(ROOT / 'work' / 'test' / 's23_raw.parquet', columns=['entity_id', 'business_name', 'business_address']) \
        .to_pandas().rename(columns={'entity_id': 's23_id', 'business_name': 'r_name', 'business_address': 'r_addr'})
    for title, diff in ((f'in {a} but NOT in {b}', ka - kb), (f'in {b} but NOT in {a}', kb - ka)):
        d = pd.DataFrame(sorted(diff), columns=['s1_id', 's23_id'])
        d['country'] = country.reindex(d.s1_id).to_numpy()
        print(f'\n### pairs {title}: {len(d):,} | by country {d.country.value_counts().to_dict()}')
        fr = d[d.country == 'France']
        ex = fr.sample(min(N, len(fr)), random_state=1).merge(s1, on='s1_id').merge(s23, on='s23_id')
        with pd.option_context('display.max_colwidth', 46, 'display.width', 240):
            print(ex[['s_name', 's_addr', 'r_name', 'r_addr']].to_string(index=False))
