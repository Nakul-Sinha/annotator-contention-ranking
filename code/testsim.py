"""Simulate the TEST evaluation condition.

Our OOF is scored over 982 faces drawn from ~141 animals, so it also demands ranking faces
*within* an animal -- differences that are dominated by per-frame sampling noise. The real test
set serves ONE face per animal across 173 disjoint animals, so every comparison the grader makes
is a between-animal comparison, where the predictable signal lives. This script re-scores the
same predictions under the test's own sampling scheme.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from blend import greedy_blend, apply_blend, rank_z

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--exclude", default="ep8,ep56,smoke")
ap.add_argument("--reps", type=int, default=400)
a = ap.parse_args()

ex = set(filter(None, a.exclude.split(",")))
files = [f for f in sorted(glob.glob(os.path.join(a.out, "oof_*.npz")))
         if os.path.basename(f)[4:-4] not in ex]
oof, y, groups = {}, None, None
for f in files:
    tag = os.path.basename(f)[4:-4]
    z = np.load(f, allow_pickle=True)
    y = z["y"] if y is None else y
    groups = z["groups"] if groups is None and "groups" in z.files else groups
    for k in z.files:
        if k.startswith("oof_") and np.isfinite(z[k]).all() and np.std(z[k]) > 0:
            oof[f"{tag}:{k[4:]}"] = z[k]

w, s_full, bl = greedy_blend(oof, y, verbose=False)
print(f"blend over {len(oof)} candidates: FULL-OOF score {s_full:.4f} (n={len(y)})", flush=True)

uq = np.unique(groups)
rng = np.random.default_rng(0)
scores, Cs, Rs = [], [], []
for _ in range(a.reps):
    pick = np.array([rng.choice(np.flatnonzero(groups == g)) for g in uq])
    s, C, R, b = score(bl[pick], y[pick], True)
    scores.append(s); Cs.append(C); Rs.append(R)
scores = np.array(scores)
print(f"ONE-FACE-PER-ANIMAL resampling (n={len(uq)} per draw, {a.reps} draws):")
print(f"  score mean {scores.mean():.4f}  sd {scores.std():.4f}  "
      f"p10 {np.percentile(scores,10):.4f}  p50 {np.percentile(scores,50):.4f}  "
      f"p90 {np.percentile(scores,90):.4f}")
print(f"  C mean {np.mean(Cs):.4f}   R mean {np.mean(Rs):.4f}")

# control: random subsets of the same size, ignoring animal structure
ctrl = []
for _ in range(a.reps):
    pick = rng.choice(len(y), len(uq), replace=False)
    ctrl.append(score(bl[pick], y[pick]))
print(f"CONTROL random subset of the same size: mean {np.mean(ctrl):.4f} sd {np.std(ctrl):.4f}")
print(f"=> structural lift from one-face-per-animal: {scores.mean()-np.mean(ctrl):+.4f}")
