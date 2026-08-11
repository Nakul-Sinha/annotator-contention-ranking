"""Nested validation of the *blending procedure* itself.

Greedy selection over dozens of candidates on one OOF vector will overfit that vector. Here the
greedy blend is refitted inside subject-grouped outer folds and scored on the held-out rows, so
the reported number estimates what the procedure earns on unseen subjects rather than how well
it can fit the rows it selected on. Also sweeps min_gain, which controls how eagerly candidates
are accepted.
"""
import argparse, glob, os, sys
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from blend import greedy_blend, rank_z

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--include", default="")
ap.add_argument("--gains", default="0.0015,0.003,0.006,0.010")
ap.add_argument("--rounds", type=int, default=18)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.out, "oof_*.npz")))
if a.include:
    keep = set(a.include.split(","))
    files = [f for f in files if os.path.basename(f)[4:-4] in keep]
cands, y, groups = {}, None, None
for f in files:
    tag = os.path.basename(f)[4:-4]
    z = np.load(f, allow_pickle=True)
    y = z["y"] if y is None else y
    groups = z["groups"] if groups is None and "groups" in z.files else groups
    for k in z.files:
        if k.startswith("oof_") and np.isfinite(z[k]).all() and np.std(z[k]) > 0:
            cands[f"{tag}:{k[4:]}"] = z[k]
print(f"{len(files)} runs -> {len(cands)} candidates, n={len(y)}", flush=True)

outer = list(GroupKFold(n_splits=5).split(np.arange(len(y)), y, groups))
for g in [float(x) for x in a.gains.split(",")]:
    w_all, s_in = greedy_blend(cands, y, rounds=a.rounds, min_gain=g, verbose=False)
    held = []
    sizes = []
    for tr_i, va_i in outer:
        sub = {k: v[tr_i] for k, v in cands.items()}
        w, _ = greedy_blend(sub, y[tr_i], rounds=a.rounds, min_gain=g, verbose=False)
        p = None
        for n, c in w.items():
            z = rank_z(cands[n][va_i]) * c
            p = z if p is None else p + z
        held.append(score(p, y[va_i]))
        sizes.append(len(w))
    print(f"  min_gain {g:.4f}: in-sample {s_in:.4f} | nested held-out "
          f"{np.mean(held):.4f} +/- {np.std(held)/np.sqrt(len(held)):.4f} "
          f"| picks {np.mean(sizes):.1f} | full-fit picks {len(w_all)}", flush=True)
