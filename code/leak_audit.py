"""Leakage audit.

The test set serves one face per animal and its animals are disjoint from train, so any signal
that works by finding a *same-animal* neighbour in the training set will not transfer. This
script re-scores the neighbour-based and tabular candidates under progressively harsher
grouping regimes; a signal whose OOF score survives the harshest regime is carrying real,
transferable image information.

Regimes
  ward141   -- the grouping used for training folds (approx. the stated 141 animals)
  ward70    -- deliberate over-merging (each fold excludes ~twice as much look-alike material)
  ward35    -- extreme over-merging
  thresh_XX -- ward141 folds PLUS removal from each training fold of every image whose embedding
               cosine to any validation image exceeds XX (a hard near-duplicate firewall)
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
a = ap.parse_args()

tr = pd.read_csv(os.path.join(a.data, "train.csv"))
y = tr.contention.values.astype(np.float64)
E = np.load(os.path.join(a.out, "emb.npy"))
Ete = np.load(os.path.join(a.out, "emb_test.npy"))
z = np.load(os.path.join(a.out, "handfeat.npz"), allow_pickle=True)
Xtr, Xte = z["Xtr"], z["Xte"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Z = (Xtr - mu) / sd
yr = QuantileTransformer(output_distribution="normal", n_quantiles=256,
                         random_state=0).fit_transform(y.reshape(-1, 1)).ravel()

P = PCA(n_components=64, random_state=0).fit_transform(E)
GRP = {f"ward{k}": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(P)
       for k in (141, 70, 35)}
S = E @ E.T

print("test-vs-train neighbour geometry (does the test set even have close neighbours?)")
Ste = Ete @ E.T
Str = S.copy(); np.fill_diagonal(Str, -9)
for q in (50, 90, 99):
    print(f"  max-cos p{q}: test->train {np.percentile(Ste.max(1), q):.4f}   "
          f"train->train {np.percentile(Str.max(1), q):.4f}")


def knn_pred(k, tr_i, va_i, mask=None):
    Sub = S[np.ix_(va_i, tr_i)]
    if mask is not None:
        Sub = np.where(mask, -9.0, Sub)
    idx = np.argsort(-Sub, axis=1)[:, :k]
    w = np.take_along_axis(Sub, idx, 1)
    w = np.exp((w - w.max(1, keepdims=True)) * 12.0)
    return (w * y[tr_i][idx]).sum(1) / w.sum(1)


def evaluate(name, groups, thresh=None):
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(y)), y, groups))
    out = {}
    for k in (16, 48):
        p = np.zeros(len(y))
        for tr_i, va_i in folds:
            m = (S[np.ix_(va_i, tr_i)] > thresh) if thresh else None
            p[va_i] = knn_pred(k, tr_i, va_i, m)
        out[f"knn{k}"] = max(score(p, y), score(-p, y))
    for nm, mk in (("ridge", lambda: Ridge(alpha=20.0)),
                   ("gbm_rk", lambda: HistGradientBoostingRegressor(
                       max_depth=3, max_iter=250, learning_rate=.05,
                       l2_regularization=1.0, random_state=0))):
        p = np.zeros(len(y))
        tgt = y if nm == "ridge" else yr
        for tr_i, va_i in folds:
            keep = tr_i
            if thresh:
                bad = (S[np.ix_(va_i, tr_i)] > thresh).any(0)
                keep = tr_i[~bad]
            p[va_i] = mk().fit(Z[keep], tgt[keep]).predict(Z[va_i])
        out[nm] = max(score(p, y), score(-p, y))
    print(f"  {name:14s} " + "  ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
    return out


print("\ncandidate scores under harsher and harsher subject separation:")
for nm, g in GRP.items():
    evaluate(nm, g)
for th in (0.95, 0.90, 0.85):
    evaluate(f"thresh_{th}", GRP["ward141"], th)

print("\nfraction of TRAIN pairs above each cosine (look-alike density):")
iu = np.triu_indices(len(S), 1)
for th in (0.95, 0.90, 0.85):
    print(f"  train-train > {th}: {np.mean(S[iu] > th):.5f}   "
          f"test->train > {th}: {np.mean(Ste.max(1) > th):.4f} of test faces")
