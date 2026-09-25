"""Entry point — runs the pipeline stage by stage, resumable.

    python -m src.run_all --split train          # S0 -> S1 -> S2 (+diagnostics) -> S3 -> S4 train -> S6 tune
    python -m src.run_all --split test           # S0 -> S1 -> S2 -> S3 -> S4 score -> S6 apply -> S7 write + validate

A stage is skipped when its output already exists (checkpointing across the 3-day window).
--from STAGE re-runs that stage and everything after it; --force re-runs everything.
"""
import argparse
import time

from . import (config, s0_ingest, s1_dictionaries, s1_signatures, s2_blocking, s2_diagnostics, s3_features,
               s4_score, s4_train, s6_decide, s7_output)


def stages(split):
    """(name, function, output that marks the stage as done) in execution order."""
    w = config.WORK_DIR / split
    models = config.WORK_DIR / 'models'
    common = [
        ('ingest', lambda: s0_ingest.ingest(split), w / 's23_raw.parquet'),
        ('dictionaries', lambda: s1_dictionaries.build(split), w / 'legal.json'),
        ('signatures', lambda: s1_signatures.build(split), w / 'idf.pkl'),
        ('blocking', lambda: s2_blocking.build(split), w / 'candidates.parquet'),
    ]
    if split == 'train':
        return common + [
            ('diagnostics', lambda: s2_diagnostics.report('train'), w / 'blocking_diagnostics.json'),
            ('features', lambda: s3_features.build('train'), w / 'features'),
            ('train', s4_train.build, models / 'meta.json'),
            ('decide', s6_decide.tune, models / 'decision.json'),
        ]
    return common + [
        ('features', lambda: s3_features.build(split), w / 'features'),
        ('score', lambda: s4_score.build(split), w / 'scores.parquet'),
        ('decide', lambda: s6_decide.apply(split), w / 'matches.parquet'),
        ('output', lambda: s7_output.build(split), config.OUTPUT_DIR / 'matching_results.tsv'),
        ('validate', s7_output.validate, None),                               # always runs
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--from', dest='start', help='re-run from this stage onwards')
    ap.add_argument('--force', action='store_true', help='re-run every stage')
    args = ap.parse_args()

    plan = stages(args.split)
    names = [n for n, _, _ in plan]
    if args.start and args.start not in names:
        ap.error(f'--from must be one of {names}')
    rerun = args.force
    for name, fn, done in plan:
        rerun = rerun or name == args.start
        if not rerun and done is not None and done.exists():
            print(f'[run] {args.split}/{name}: done, skipping')
            continue
        t0 = time.time()
        print(f'[run] {args.split}/{name} ...')
        fn()
        print(f'[run] {args.split}/{name} finished in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
