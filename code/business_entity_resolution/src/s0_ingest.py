"""Stage 0 — ingest the raw TSVs into compact parquet.

Outputs (in work/<split>/):
    s1_raw.parquet   entity_id, business_name, business_address, country      (row number = S1 index)
    s23_raw.parquet  same columns + src (2/3); S2 rows first, then S3          (row number = S2/S3 index)
    gt_pairs.parquet s1, s23 int32 row indices of every true link (train only)

Everything downstream works with int32 row indices; entity_id strings are only used again when
writing the submission files.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import config

COLS = ['entity_id', 'business_name', 'business_address', 'country']


def read_tsv(path):
    """Read a challenge TSV with pyarrow: explicit tab separator, quoting OFF (the data contains
    stray `"`), every column as string and empty fields kept as '' (never null)."""
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter='\t', quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in COLS},
                                             strings_can_be_null=False),
    )


def id_code(ids):
    """Encode 'S2-123' / 'S3-123' as one int64 (source * 1e10 + number) so ids can be matched with
    numpy searchsorted instead of a 10M-entry Python dict. Every id in the data is S{n}-<digits>."""
    ids = pa.array(ids) if not isinstance(ids, (pa.Array, pa.ChunkedArray)) else ids
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64()).to_numpy()
    return src * 10_000_000_000 + num


def ingest(split):
    """Convert one split's TSVs to parquet (and the ground truth to index pairs for train)."""
    out = config.split_dir(split)
    src_dir = config.DATA_DIR / split

    s1 = read_tsv(src_dir / f'{split}_source1.tsv')
    pq.write_table(s1, out / 's1_raw.parquet')

    parts = []
    for n in (2, 3):
        t = read_tsv(src_dir / f'{split}_source{n}.tsv')
        parts.append(t.append_column('src', pa.array(np.full(t.num_rows, n, dtype=np.int8))))
    s23 = pa.concat_tables(parts)
    pq.write_table(s23, out / 's23_raw.parquet')
    print(f'[s0] {split}: {s1.num_rows:,} S1 rows, {s23.num_rows:,} S2/S3 rows')

    gt_path = src_dir / f'{split}_ground_truth.tsv'
    if gt_path.exists():
        write_gt_pairs(gt_path, s1, s23, out / 'gt_pairs.parquet')


def write_gt_pairs(gt_path, s1, s23, out_path):
    """Explode 'S1 -> comma-separated ids' ground truth into (s1, s23) int32 index pairs."""
    gt = pacsv.read_csv(gt_path, parse_options=pacsv.ParseOptions(delimiter='\t', quote_char=False),
                        convert_options=pacsv.ConvertOptions(strings_can_be_null=False,
                                                             column_types={'source1_entity_id': pa.string(),
                                                                           'matched_entity_ids': pa.string()}))
    lists = pc.split_pattern(gt['matched_entity_ids'], ',')
    lengths = pc.list_value_length(lists).to_numpy()
    s1_ids = np.repeat(gt['source1_entity_id'].to_numpy(zero_copy_only=False), lengths)
    s23_ids = pc.list_flatten(lists)
    keep = pc.not_equal(s23_ids, '').to_numpy(zero_copy_only=False)  # empty list -> [''] after split
    s1_ids, s23_ids = s1_ids[keep], s23_ids.filter(pa.array(keep))

    s1_idx = lookup(id_code(s1['entity_id']), id_code(s1_ids))
    s23_idx = lookup(id_code(s23['entity_id']), id_code(s23_ids))
    pq.write_table(pa.table({'s1': s1_idx, 's23': s23_idx}), out_path)
    print(f'[s0] ground truth: {len(s1_idx):,} true links')


def lookup(table_codes, query_codes):
    """Row index of each query id code inside table_codes (all queries must exist)."""
    order = np.argsort(table_codes)
    pos = np.searchsorted(table_codes, query_codes, sorter=order)
    idx = order[np.minimum(pos, len(order) - 1)]
    assert (table_codes[idx] == query_codes).all(), 'ground truth references an unknown entity id'
    return idx.astype(np.int32)
