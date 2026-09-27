"""Stage 8a — export record-pair texts for the LLM cross-encoder (runs locally, ~5 min).

The GBDT is nearly always right when it is confident: 98% of its false negatives and 94% of its
false positives on TUNE have p in [0.01, 0.99], which is only ~3.6% of pairs. The cross-encoder
therefore re-reads just that grey zone.

    work/ce/train.parquet   a, b, c, label   fine-tuning pairs:
                              * CAL grey-zone pairs (honest hard cases, p from the calibrated GBDT)
                              * TRAIN-role entities: every positive + the top pre-score negatives
                              * France pseudo-labels from test (GBDT p > 0.995 and the satellite's
                                top claimant -> 1; hard candidates with p < 0.003 -> 0). France has no
                                labels; these teach the encoder French address / legal forms. They lie
                                outside the grey zone, so no pair is both trained on and re-scored.
    work/ce/tune.parquet    s1, s23, p, label, noaddr, a, b, c   TUNE grey zone (stacker + tau search)
    work/ce/test.parquet    s1, s23, p, noaddr, a, b, c          test grey zone (scored on the GPU box)

Run: ER_WORK_DIR=<work_v5> python -m src.s8_ce_export
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .s4_train import CAL, TRAIN, TUNE, entity_roles
from .s6_decide import _owner_mask, noaddr_mask

ZONE = (0.01, 0.99)
TRAIN_ENTITIES = 80_000         # TRAIN-role S1s sampled for positives + hard negatives
HARD_NEG = 3                    # negatives per sampled S1, by pre-score rank
FRANCE_PER_CLASS = 60_000
OUT = config.WORK_DIR / 'ce'


def record_text(split):
    """(s1 texts, s23 texts, s1 country) as numpy object arrays indexed like the pipeline's int ids."""
    def txt(path):
        """Raw "<name> ; <address>" text and country of every record in one parquet."""
        t = pq.read_table(path, columns=['business_name', 'business_address', 'country']).to_pandas()
        return (t.business_name.fillna('') + ' ; ' + t.business_address.fillna('')).to_numpy(object), \
            t.country.fillna('').to_numpy(object)
    w = config.WORK_DIR / split
    a, c = txt(w / 's1_raw.parquet')
    b, _ = txt(w / 's23_raw.parquet')
    return a, b, c


def attach(df, texts):
    """Add the raw text columns a (S1), b (satellite) and c (country) to a pair frame."""
    a, b, c = texts
    return df.assign(a=a[df.s1.to_numpy()], b=b[df.s23.to_numpy()], c=c[df.s1.to_numpy()])


def grey(df):
    """Pairs whose GBDT probability lies in the grey zone ZONE."""
    return df[(df.p >= ZONE[0]) & (df.p <= ZONE[1])].reset_index(drop=True)


def main():
    """Export the fine-tuning pairs and the TUNE / test grey-zone pairs as raw text to work/ce/."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    # ---- train split: CAL grey zone, TUNE grey zone, TRAIN-role positives + hard negatives ----
    tr = record_text('train')
    vs = pq.read_table(config.WORK_DIR / 'train' / 'valid_scores.parquet').to_pandas()
    vs['noaddr'] = noaddr_mask('train', vs.s1, vs.s23)
    cal = grey(vs[vs.role == CAL])
    tune = grey(vs[vs.role == TUNE])
    attach(tune, tr).to_parquet(OUT / 'tune.parquet')
    print(f'[s8] tune grey zone: {len(tune):,} pairs ({tune.label.mean():.3f} positive)', flush=True)

    ents = np.load(config.WORK_DIR / 'train' / 'entities.npy')
    role, _ = entity_roles(ents)
    pick = rng.choice(ents[role[ents] == TRAIN], TRAIN_ENTITIES, replace=False)
    cand = pq.read_table(config.WORK_DIR / 'train' / 'candidates.parquet', columns=['s1', 's23', 'prescore'])
    cand = cand.filter(pa.compute.is_in(cand['s1'], pa.array(pick))).to_pandas()
    gt = pq.read_table(config.WORK_DIR / 'train' / 'gt_pairs.parquet').to_pandas()
    gt = gt[np.isin(gt.s1, pick)]
    key = lambda d: d.s1.to_numpy(np.int64) << 32 | d.s23.to_numpy(np.int64)
    cand['label'] = np.isin(key(cand), key(gt))
    neg = cand[~cand.label].sort_values(['s1', 'prescore'], ascending=[True, False])
    neg = neg[neg.groupby('s1').cumcount() < HARD_NEG]
    fit = pd.concat([cand[cand.label], neg])
    fit = pd.concat([fit[['s1', 's23', 'label']], cal[['s1', 's23', 'label']]])
    fit = attach(fit, tr)
    del tr

    # ---- test split: grey zone to score + France pseudo-labels ----
    te = record_text('test')
    sc = pq.read_table(config.WORK_DIR / 'test' / 'scores.parquet').to_pandas()
    sc['noaddr'] = noaddr_mask('test', sc.s1, sc.s23)
    test = grey(sc)
    attach(test, te).to_parquet(OUT / 'test.parquet')
    print(f'[s8] test grey zone: {len(test):,} pairs', flush=True)

    fr = te[2][sc.s1.to_numpy()] == 'France'
    owner = _owner_mask(sc.s23.to_numpy(), sc.p.to_numpy())
    pos = np.flatnonzero(fr & owner & (sc.p.to_numpy() > 0.995))
    cand_t = pq.read_table(config.WORK_DIR / 'test' / 'candidates.parquet', columns=['s1', 's23', 'prescore']).to_pandas()
    assert (cand_t.s1.to_numpy() == sc.s1.to_numpy()).all() and (cand_t.s23.to_numpy() == sc.s23.to_numpy()).all()
    rank = cand_t.assign(i=np.arange(len(cand_t))).sort_values(['s1', 'prescore'], ascending=[True, False])
    hard = np.zeros(len(sc), bool)
    hard[rank.i.to_numpy()[rank.groupby('s1').cumcount().to_numpy() < 8]] = True
    neg = np.flatnonzero(fr & hard & (sc.p.to_numpy() < 0.003))
    pos = rng.choice(pos, min(FRANCE_PER_CLASS, len(pos)), replace=False)
    neg = rng.choice(neg, min(FRANCE_PER_CLASS, len(neg)), replace=False)
    pseudo = sc.iloc[np.r_[pos, neg]][['s1', 's23']].assign(label=np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    fit = pd.concat([fit, attach(pseudo, te)]).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    fit[['a', 'b', 'c', 'label']].to_parquet(OUT / 'train.parquet')
    print(f'[s8] train: {len(fit):,} pairs ({fit.label.mean():.3f} positive; CAL grey {len(cal):,}, '
          f'France pseudo {len(pseudo):,})', flush=True)


if __name__ == '__main__':
    main()
