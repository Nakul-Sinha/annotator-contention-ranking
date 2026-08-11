"""Quantify subject leakage and validate the recovered subject grouping."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from subjects import cluster_subjects, descriptors

DATA = sys.argv[1] if len(sys.argv) > 1 else r"G:\Datacurve\Latest_Chals\The Split Verdict Forecasting Asse\dataset"
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
img = lambda i: os.path.join(DATA, "images", i + ".jpg")
y = tr.contention.values

print("=== nearest-neighbour appearance leakage on TRAIN ===", flush=True)
D = descriptors([img(i) for i in tr.image_id])
Dn = (D - D.mean(0)) / (D.std(0) + 1e-6)
Dn /= np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-8
S = Dn @ Dn.T
np.fill_diagonal(S, -9)
nn = S.argmax(1)
print(f"|y_i - y_nn| mean {np.mean(np.abs(y - y[nn])):.4f}")
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
print(f"|y_i - y_rand| mean {np.mean(np.abs(y - y[perm])):.4f}")
print(f"spearman(y, y_nn) = {spearmanr(y, y[nn]).statistic:+.4f}   <- appearance-NN target copy")
print(f"nn cosine-sim: mean {S.max(1).mean():.4f} p10 {np.percentile(S.max(1),10):.4f}")

print("\n=== 1-NN 'copy the neighbour's contention' as a priority (leak probe) ===")
s, C, R, b = score(y[nn], y, True)
print(f"  score {s:.4f} C {C:.4f} R {R:.4f}  (high => strong subject/appearance leakage)")

print("\n=== recovered subject clusters ===", flush=True)
for k in (141,):
    lab, _ = cluster_subjects([img(i) for i in tr.image_id], n_clusters=k)
    sizes = pd.Series(lab).value_counts()
    print(f"k={k}: sizes mean {sizes.mean():.2f} median {sizes.median():.0f} max {sizes.max()} "
          f"n_singleton {(sizes==1).sum()}")
    # within-cluster target homogeneity vs random clusters of the same sizes
    within = np.mean([np.var(y[lab == c]) for c in np.unique(lab) if (lab == c).sum() > 1])
    rnd = lab.copy(); rng.shuffle(rnd)
    within_r = np.mean([np.var(y[rnd == c]) for c in np.unique(rnd) if (rnd == c).sum() > 1])
    print(f"   within-cluster var {within:.5f} vs shuffled {within_r:.5f} "
          f"(total {np.var(y):.5f}) -> ICC-ish {1 - within/within_r:+.3f}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "groups141.npy"), lab)

print("\n=== how similar are TEST faces to TRAIN faces? (domain check) ===")
Dte = descriptors([img(i) for i in te.image_id])
Dte = (Dte - D.mean(0)) / (D.std(0) + 1e-6)
Dte /= np.linalg.norm(Dte, axis=1, keepdims=True) + 1e-8
Ste = Dte @ Dn.T
print(f"test->train max cosine: mean {Ste.max(1).mean():.4f} p90 {np.percentile(Ste.max(1),90):.4f}")
print(f"train->train max cosine: mean {S.max(1).mean():.4f} p90 {np.percentile(S.max(1),90):.4f}")
