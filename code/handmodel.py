"""Non-convolutional candidate signals, produced on the same subject-grouped folds.

These are deliberately weak and mutually decorrelated; they exist to be blended with the
convolutional read-outs, never to stand alone. Everything here is fit only on train.csv.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from feats import feature_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()

tr = pd.read_csv(os.path.join(a.data, "train.csv"))
te = pd.read_csv(os.path.join(a.data, "test.csv"))
img = lambda i: os.path.join(a.data, "images", i + ".jpg")
y = tr.contention.values.astype(np.float64)
groups = np.load(os.path.join(a.out, "groups.npy"))

fc = os.path.join(a.out, "handfeat.npz")
if os.path.exists(fc):
    z = np.load(fc, allow_pickle=True); Xtr, Xte = z["Xtr"], z["Xte"]
else:
    Xtr, fn = feature_matrix([img(i) for i in tr.image_id])
    Xte, _ = feature_matrix([img(i) for i in te.image_id])
    np.savez(fc, Xtr=Xtr, Xte=Xte, fnames=np.array(fn, dtype=object))

E = np.load(os.path.join(a.out, "emb.npy"))          # train embeddings (identity/appearance)
# test embeddings
ep = os.path.join(a.out, "emb_test.npy")
if os.path.exists(ep):
    Ete = np.load(ep)
else:
    import torch, timm
    from train_lib import load_images, to_tensor
    Ite = load_images([img(i) for i in te.image_id])
    m = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg").eval()
    with torch.no_grad():
        Ete = np.concatenate([m(to_tensor(Ite[i:i + 128])).numpy() for i in range(0, len(Ite), 128)])
    Ete /= np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-8
    np.save(ep, Ete)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
yr = QuantileTransformer(output_distribution="normal", n_quantiles=256,
                         random_state=0).fit_transform(y.reshape(-1, 1)).ravel()
thr = np.quantile(y, 0.75)

CANDS = {}
TE = {}
gkf = GroupKFold(n_splits=a.folds)
folds = list(gkf.split(np.arange(len(y)), y, groups))


def run(name, fit_predict):
    oof = np.zeros(len(y))
    for tr_i, va_i in folds:
        oof[va_i] = fit_predict(tr_i, va_i)
    CANDS[name] = oof
    TE[name] = fit_predict(np.arange(len(y)), None)
    print(f"  {name:14s} OOF {max(score(oof,y), score(-oof,y)):.4f}", flush=True)


def _ridge(alpha, target):
    def f(tr_i, va_i):
        m = Ridge(alpha=alpha).fit(Ztr[tr_i], target[tr_i])
        return m.predict(Ztr[va_i] if va_i is not None else Zte)
    return f


def _gbm(target, **kw):
    def f(tr_i, va_i):
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0, **kw)
        m.fit(Ztr[tr_i], target[tr_i])
        return m.predict(Ztr[va_i] if va_i is not None else Zte)
    return f


def _gbm_top(tr_i, va_i):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=0)
    m.fit(Ztr[tr_i], (y[tr_i] > thr).astype(int))
    return m.predict_proba(Ztr[va_i] if va_i is not None else Zte)[:, 1]


def _knn(k, use_emb):
    F = E if use_emb else (Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-8))
    G = Ete if use_emb else (Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-8))

    def f(tr_i, va_i):
        Q = F[va_i] if va_i is not None else G
        S = Q @ F[tr_i].T
        idx = np.argsort(-S, axis=1)[:, :k]
        w = np.take_along_axis(S, idx, 1)
        w = np.exp((w - w.max(1, keepdims=True)) * 12.0)
        return (w * y[tr_i][idx]).sum(1) / w.sum(1)
    return f


print("=== non-convolutional candidates (subject-grouped OOF) ===", flush=True)
run("h_ridge", _ridge(20.0, y))
run("h_ridge_rk", _ridge(20.0, yr))
run("h_ridge_a2", _ridge(200.0, yr))
run("h_gbm", _gbm(y))
run("h_gbm_rk", _gbm(yr))
run("h_gbmtop", _gbm_top)
run("h_knn16", _knn(16, True))
run("h_knn48", _knn(48, True))
run("h_knnf32", _knn(32, False))

np.savez(os.path.join(a.out, "oof_hand.npz"),
         **{f"oof_{k}": v for k, v in CANDS.items()},
         **{f"te_{k}": v for k, v in TE.items()}, y=y, groups=groups)
print("saved oof_hand.npz", flush=True)
