"""Per-country rates of the suspicious test slices, normalised by S1 count, and what the
'partition' decision mode (one owner per satellite) would remove from the fast submission."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'business_entity_resolution'))
from src import config  # noqa: E402
from src.s6_decide import decide  # noqa: E402

T = config.WORK_DIR / 'test'
country = pq.read_table(T / 's1_raw.parquet', columns=['country'])['country'].to_numpy(zero_copy_only=False)
pairs = pq.read_table(T / 'audit_pairs.parquet', columns=['s1', 's23', 'p', 'sfx_compat']).to_pandas()
s1_core = pq.read_table(T / 's1_sig.parquet', columns=['name_core'])['name_core'].to_numpy(zero_copy_only=False)
s23_core = pq.read_table(T / 's23_sig.parquet', columns=['name_core'])['name_core'].to_numpy(zero_copy_only=False)

thr = decide(pairs.s1, pairs.s23, pairs.p, 0.65, 'threshold')
par = decide(pairs.s1, pairs.s23, pairs.p, 0.65, 'partition')
m = pairs[thr].copy()
m['country'] = country[m.s1.to_numpy()]
m['multi'] = m.groupby('s23').s1.transform('size') >= 2
m['g'] = (s1_core[m.s1.to_numpy()] == s23_core[m.s23.to_numpy()]) & (m.sfx_compat == 3)
m['dropped_by_partition'] = ~par[thr]
n_s1 = pd.Series(country).value_counts()

rows = []
for c, g in m.groupby('country'):
    rows.append({'country': c, 'S1s': n_s1[c], 'matches/S1': len(g) / n_s1[c],
                 'H pairs per 1k S1': 1000 * g.multi.sum() / n_s1[c],
                 'S1s touched by H': g[g.multi].s1.nunique(),
                 'G pairs per 1k S1': 1000 * g.g.sum() / n_s1[c],
                 'partition drops': int(g.dropped_by_partition.sum()),
                 'S1s left empty by partition': int(len(set(g.s1) - set(g.s1[~g.dropped_by_partition])))})
print(pd.DataFrame(rows).round(2).to_string(index=False))

# words that make the France sibling pairs different: tokens in one name_c but not the other
sig1 = pq.read_table(T / 's1_sig.parquet', columns=['name_c', 'sfx']).to_pandas()
sig23 = pq.read_table(T / 's23_sig.parquet', columns=['name_c', 'sfx']).to_pandas()
fr = m[(m.country == 'France') & m.multi]
diff = [tuple(sorted(set(a.split()) ^ set(b.split())))
        for a, b in zip(sig1.name_c.to_numpy()[fr.s1], sig23.name_c.to_numpy()[fr.s23])]
print('\nFrance H pairs: most common name-token differences (symmetric difference of canonical names):')
print(pd.Series(diff).value_counts().head(15).to_string())
sfx = json.loads((T / 'legal.json').read_text())
sfx = sfx if isinstance(sfx, list) else sfx.get('legal', sfx)
print(f'\ntest legal/generic suffix set ({len(sfx)} tokens):', sorted(sfx)[:80])
