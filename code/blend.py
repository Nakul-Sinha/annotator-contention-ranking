"""Metric-greedy blending of many weak, decorrelated ranking signals.

Follows the rank-1 recipe: sign-correct each candidate by its own OOF score, seed with the
best, then repeatedly test adding one more candidate to an equal-weight average of ranks,
accepting only when the *actual competition metric* improves by more than `min_gain`.
Repeated selection quantises the weights.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr
from metric import score


def rank_z(v):
    v = np.asarray(v, dtype=np.float64)
    v = np.nan_to_num(v, nan=np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0,
                      posinf=0.0, neginf=0.0)
    r = rankdata(v)
    r = (r - r.mean()) / (r.std() + 1e-12)
    return r


def greedy_blend(cands, y, rounds=14, min_gain=0.0015, verbose=True, seed_pool=None):
    """cands: dict name -> 1d array of OOF values. Returns (weights, oof_score, blend)."""
    names = list(cands)
    Z, sign, s0 = {}, {}, {}
    for n in names:
        z = rank_z(cands[n])
        sp = score(z, y)
        sn = score(-z, y)
        sign[n] = 1.0 if sp >= sn else -1.0
        Z[n] = z * sign[n]
        s0[n] = max(sp, sn)
    order = sorted(names, key=lambda n: -s0[n])
    if verbose:
        for n in order:
            print(f"    cand {n:16s} sign{sign[n]:+.0f}  solo {s0[n]:.4f}", flush=True)
    pool = order if seed_pool is None else [n for n in order if n in seed_pool]
    best = pool[0]
    sel = [best]
    cur = Z[best].copy()
    cur_s = s0[best]
    for _ in range(rounds - 1):
        gain = None
        for n in names:
            trial = (cur * len(sel) + Z[n]) / (len(sel) + 1)
            s = score(trial, y)
            if gain is None or s > gain[0]:
                gain = (s, n, trial)
        if gain[0] > cur_s + min_gain:
            cur_s, cur = gain[0], gain[2]
            sel.append(gain[1])
            if verbose:
                print(f"    + {gain[1]:16s} -> {cur_s:.4f}", flush=True)
        else:
            break
    w = {n: sign[n] * sel.count(n) / len(sel) for n in set(sel)}
    return w, cur_s, cur


def apply_blend(cands, w):
    out = None
    for n, v in w.items():
        z = rank_z(cands[n]) * v
        out = z if out is None else out + z
    return out


def corr_table(cands, names=None):
    names = names or list(cands)
    Z = np.stack([rank_z(cands[n]) for n in names])
    return names, np.corrcoef(Z)
