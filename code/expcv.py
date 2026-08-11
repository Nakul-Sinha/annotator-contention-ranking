"""Flexible CV driver: run one config, save subject-grouped OOF + test read-outs to npz.

All runs share one fixed group assignment and one deterministic GroupKFold split, so read-outs
from different runs live on the same OOF rows and can be blended against each other later.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from feats import feature_matrix
from blend import greedy_blend
from runner import run_cv, DEFAULT_CFG
from train_lib import load_images, READOUT_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--tag", required=True)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drop", type=float, default=0.30)
ap.add_argument("--lr", type=float, default=2.5e-3)
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--size", type=int, default=112)
ap.add_argument("--noaux", action="store_true")
ap.add_argument("--nopre", action="store_true")
ap.add_argument("--tta", type=int, default=4)
ap.add_argument("--photo", type=float, default=0.06)
ap.add_argument("--lrtm", type=float, default=0.25)
ap.add_argument("--hidden", type=int, default=192)
ap.add_argument("--bins", type=int, default=12)
ap.add_argument("--w", default="")           # e.g. "w_rank=1.5,w_reg=0.0"
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
try:
    import cv2; cv2.setNumThreads(1)
except Exception:
    pass

tr = pd.read_csv(os.path.join(a.data, "train.csv"))
te = pd.read_csv(os.path.join(a.data, "test.csv"))
img = lambda i: os.path.join(a.data, "images", i + ".jpg")
y = tr.contention.values.astype(np.float64)
t0 = time.time()

Itr = load_images([img(i) for i in tr.image_id])
Ite = load_images([img(i) for i in te.image_id])
if a.size != Itr.shape[1]:
    import cv2 as _cv
    rs = lambda A: np.stack([_cv.resize(z, (a.size, a.size), interpolation=_cv.INTER_CUBIC)
                             for z in A])
    Itr, Ite = rs(Itr), rs(Ite)

fcache = os.path.join(a.out, "handfeat.npz")
if os.path.exists(fcache):
    z = np.load(fcache, allow_pickle=True)
    Xtr, Xte, fnames = z["Xtr"], z["Xte"], list(z["fnames"])
else:
    Xtr, fnames = feature_matrix([img(i) for i in tr.image_id])
    Xte, _ = feature_matrix([img(i) for i in te.image_id])
    np.savez(fcache, Xtr=Xtr, Xte=Xte, fnames=np.array(fnames, dtype=object))
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Atr = ((Xtr - mu) / sd).astype(np.float32)
Ate = ((Xte - mu) / sd).astype(np.float32)

groups = np.load(os.path.join(a.out, "groups.npy"))

cfg = dict(DEFAULT_CFG)
cfg.update(epochs=a.epochs, backbone=a.backbone, seed=a.seed, drop=a.drop, lr=a.lr,
           bs=a.bs, tta=a.tta, photo=a.photo, aux=not a.noaux, pretrained=not a.nopre,
           lr_trunk_mult=a.lrtm, hidden=a.hidden, n_bins=a.bins)
for kv in filter(None, a.w.split(",")):
    k, v = kv.split("="); cfg[k] = float(v)
print(f"[{a.tag}] cfg={cfg}", flush=True)

oof, tep, nf = run_cv(Itr, y, Atr, groups, Ite, Ate, cfg, n_splits=a.folds)

print(f"\n[{a.tag}] per-readout OOF:", flush=True)
solo = {}
for k in READOUT_KEYS:
    sp, sn = score(oof[k], y), score(-oof[k], y)
    solo[k] = max(sp, sn)
for k in sorted(solo, key=lambda k: -solo[k]):
    print(f"  {k:14s} {solo[k]:.4f}", flush=True)
w, s, _ = greedy_blend({k: oof[k] for k in READOUT_KEYS}, y, verbose=False)
print(f"[{a.tag}] within-run blend {s:.4f}  {w}", flush=True)

np.savez(os.path.join(a.out, f"oof_{a.tag}.npz"),
         **{f"oof_{k}": oof[k] for k in READOUT_KEYS},
         **{f"te_{k}": tep[k] for k in READOUT_KEYS}, y=y, groups=groups)
json.dump(dict(tag=a.tag, cfg=cfg, solo=solo, blend=w, blend_score=s, folds=nf,
               secs=time.time() - t0), open(os.path.join(a.out, f"res_{a.tag}.json"), "w"),
          indent=1, default=float)
print(f"[{a.tag}] done in {time.time()-t0:.0f}s", flush=True)
