"""Day-0 gate: profile labels, identify the published legibility baseline, anchor the metric."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score, capture_C
from feats import feature_matrix

DATA = sys.argv[1] if len(sys.argv) > 1 else r"G:\Datacurve\Latest_Chals\The Split Verdict Forecasting Asse\dataset"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
img = lambda i: os.path.join(DATA, "images", i + ".jpg")

print("=== label profile ===")
y = tr.contention.values
print(f"n={len(y)} mean={y.mean():.4f} sd={y.std():.4f} min={y.min()} max={y.max()}")
print(f"zeros={np.mean(y==0):.4f}  skew={float(pd.Series(y).skew()):.3f}")
for q in [.1,.25,.5,.75,.9,.95,.99]:
    print(f"  q{q:.2f} {np.quantile(y,q):.4f}")

print("\n=== computing image features ===", flush=True)
Xtr, names = feature_matrix([img(i) for i in tr.image_id])
Xte, _ = feature_matrix([img(i) for i in te.image_id])
np.save(os.path.join(OUT, "Xtr.npy"), Xtr); np.save(os.path.join(OUT, "Xte.npy"), Xte)
json.dump(names, open(os.path.join(OUT, "featnames.json"), "w"))
print("features:", len(names))

print("\n=== which feature explains the sample_submission baseline priority? ===")
base = ss.set_index("image_id").loc[te.image_id, "priority"].values
rows = []
for j, n in enumerate(names):
    r = spearmanr(Xte[:, j], base).statistic
    rows.append((abs(r), r, n, j))
rows.sort(reverse=True)
for a, r, n, j in rows[:12]:
    print(f"  {n:18s} spearman_vs_baseline {r:+.4f}")

print("\n=== each feature as a standalone priority, scored on TRAIN ===")
res = []
for j, n in enumerate(names):
    s_pos, C, R, b = score(Xtr[:, j], y, True)
    s_neg = score(-Xtr[:, j], y, True)
    if s_neg[0] > s_pos:
        res.append((s_neg[0], -1, n, s_neg[1], s_neg[2]))
    else:
        res.append((s_pos, +1, n, C, R))
res.sort(reverse=True)
for s, sg, n, C, R in res[:20]:
    print(f"  {n:18s} sign{sg:+d}  score {s:.4f}  C {C:+.4f}  R {R:+.4f}")

print("\n=== baseline anchor ===")
# The published baseline is a legibility priority; reproduce its best proxy on train.
bestj = rows[0][3]; bestn = rows[0][2]
sgn = np.sign(rows[0][1])
s, C, R, b = score(sgn * Xtr[:, bestj], y, True)
print(f"baseline proxy = {sgn:+.0f}*{bestn}: train score {s:.4f} (C {C:.4f} R {R:.4f})")
print(f"\nnoise floor: label is a small-sample statistic (3-5 assessors).")
