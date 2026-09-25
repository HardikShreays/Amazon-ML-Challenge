"""Macro-F0.5 exactly as the challenge scores it.

Per S1 entity:  P = |pred ∩ truth| / |pred|,  R = |pred ∩ truth| / |truth|,
                F0.5 = 1.25·P·R / (0.25·P + R)
  * true singleton (truth empty): 1.0 if pred is empty, else 0.0
  * non-singleton with empty pred: 0.0
Averaged over ALL evaluated S1 entities (singletons and entities without candidates included).
"""
import numpy as np
import pandas as pd


def f05(pred, truth):
    """F0.5 of one entity from two sets of ids."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(set(pred) & set(truth))
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_s1, pred_s23, true_s1, true_s23, entities):
    """Vectorised macro-F0.5 over `entities` from (s1, s23) index pairs of predictions and truth."""
    entities = np.asarray(entities)
    pred = pd.DataFrame({'s1': pred_s1, 's23': pred_s23})
    truth = pd.DataFrame({'s1': true_s1, 's23': true_s23})
    truth = truth[truth.s1.isin(entities)]
    n_pred = pred.groupby('s1').size().reindex(entities, fill_value=0).to_numpy(float)
    n_true = truth.groupby('s1').size().reindex(entities, fill_value=0).to_numpy(float)
    tp = pred.merge(truth, on=['s1', 's23']).groupby('s1').size().reindex(entities, fill_value=0).to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
    return float(f.mean())


if __name__ == '__main__':
    # values from the problem statement and the build plan's F0.5 table
    assert round(f05({'a', 'b', 'c'}, {'a', 'c'}), 3) == 0.714
    assert round(f05({'a', 'b'}, {'a', 'b', 'c'}), 3) == 0.909
    assert round(f05({'a', 'b', 'c', 'x'}, {'a', 'b', 'c'}), 3) == 0.789
    assert round(f05({'a', 'b', 'x', 'y'}, {'a', 'b', 'c'}), 3) == 0.526
    assert f05(set(), set()) == 1.0 and f05({'a'}, set()) == 0.0 and f05(set(), {'a'}) == 0.0
    # vectorised version == per-entity loop: entity 0 (0.714), 1 true singleton predicted empty (1.0),
    # 2 singleton with a false merge (0.0), 3 missed entirely (0.0)
    m = macro_f05([0, 0, 0, 2], [1, 2, 3, 9], [0, 0, 3], [1, 3, 5], [0, 1, 2, 3])
    assert abs(m - (0.714285 + 1 + 0 + 0) / 4) < 1e-4, m
    print('evaluate self-check OK')
