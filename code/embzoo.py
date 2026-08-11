"""Frozen-embedding zoo.

We only ever tested ResNet-18 features at the native 112 px. Generic backbones are trained at
224 px and stronger ones may carry far more of the pose / occlusion / capture-condition
information that drives panel disagreement -- and pure inference is cheap on CPU (one pass over
1155 tiny images), so several backbones can be afforded. Each backbone contributes ridge / kNN /
GBM read-outs on the same subject-grouped folds as blend candidates.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from train_lib import load_images

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--sizes", default="224")
ap.add_argument("--models", default="resnet50,convnext_tiny,efficientnet_b0,vit_small_patch16_224,"
                                    "resnet18,tf_efficientnet_b3")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
try:
    import cv2; cv2.setNumThreads(1)
except Exception:
    pass

tr = pd.read_csv(os.path.join(a.data, "train.csv"))
te = pd.read_csv(os.path.join(a.data, "test.csv"))
img = lambda i: os.path.join(a.data, "images", i + ".jpg")
y = tr.contention.values.astype(np.float64)
groups = np.load(os.path.join(a.out, "groups.npy"))
folds = list(GroupKFold(n_splits=a.folds).split(np.arange(len(y)), y, groups))
yr = QuantileTransformer(output_distribution="normal", n_quantiles=256,
                         random_state=0).fit_transform(y.reshape(-1, 1)).ravel()
Itr = load_images([img(i) for i in tr.image_id])
Ite = load_images([img(i) for i in te.image_id])

import timm
from timm.data import resolve_data_config


def embed(name, size, imgs, model, cfg):
    mean = np.array(cfg["mean"], np.float32) * 255.0
    std = np.array(cfg["std"], np.float32) * 255.0
    outs = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            ch = imgs[i:i + 64].astype(np.float32)
            if size != ch.shape[1]:
                ch = np.stack([cv2.resize(z, (size, size), interpolation=cv2.INTER_CUBIC)
                               for z in ch])
            x = (ch - mean) / std
            outs.append(model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).float().numpy())
    E = np.concatenate(outs)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


CAND, TE = {}, {}


def add(name, fp):
    try:
        oof = np.zeros(len(y))
        for tr_i, va_i in folds:
            oof[va_i] = fp(tr_i, va_i)
        CAND[name], TE[name] = oof, fp(np.arange(len(y)), None)
        return max(score(oof, y), score(-oof, y))
    except Exception as e:
        print(f"    {name} failed: {type(e).__name__}: {e}", flush=True)
        return float("nan")


for name in a.models.split(","):
    for size in [int(s) for s in a.sizes.split(",")]:
        t0 = time.time()
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg").eval()
            cfg = resolve_data_config({}, model=m)
        except Exception as e:
            print(f"{name}: unavailable ({type(e).__name__}: {e})", flush=True)
            continue
        Etr = embed(name, size, Itr, m, cfg)
        Ete = embed(name, size, Ite, m, cfg)
        tag = f"{name.split('_patch')[0][:14]}{size}"
        del m

        def ridge(alpha, tgt, E=Etr, G=Ete):
            return lambda p, q: Ridge(alpha=alpha).fit(E[p], tgt[p]).predict(
                E[q] if q is not None else G)

        def knn(k, E=Etr, G=Ete):
            def f(p, q):
                Q = E[q] if q is not None else G
                S = Q @ E[p].T
                idx = np.argsort(-S, axis=1)[:, :k]
                w = np.take_along_axis(S, idx, 1)
                w = np.exp((w - w.max(1, keepdims=True)) * 12.0)
                return (w * y[p][idx]).sum(1) / w.sum(1)
            return f

        def nov(k, E=Etr, G=Ete):
            def f(p, q):
                Q = E[q] if q is not None else G
                return -np.sort(Q @ E[p].T, axis=1)[:, -k:].mean(1)
            return f

        r = {
            "ridge": add(f"z_{tag}_ridge", ridge(30.0, y)),
            "ridgerk": add(f"z_{tag}_ridgerk", ridge(30.0, yr)),
            "ridgeA": add(f"z_{tag}_ridgeA", ridge(300.0, yr)),
            "knn16": add(f"z_{tag}_knn16", knn(16)),
            "knn48": add(f"z_{tag}_knn48", knn(48)),
            "nov16": add(f"z_{tag}_nov16", nov(16)),
        }
        print(f"{tag:22s} dim {Etr.shape[1]:5d} {time.time()-t0:5.0f}s  " +
              "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)

np.savez(os.path.join(a.out, "oof_zoo.npz"),
         **{f"oof_{k}": v for k, v in CAND.items()},
         **{f"te_{k}": v for k, v in TE.items()}, y=y, groups=groups)
print(f"saved oof_zoo.npz with {len(CAND)} candidates", flush=True)
