"""Experiment 1: subject grouping quality, CNN OOF, candidate table, greedy blend."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from feats import feature_matrix
from blend import greedy_blend, rank_z, corr_table
from runner import run_cv, DEFAULT_CFG
from train_lib import load_images, READOUT_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=26)
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--tag", default="e1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--groupmode", default="embed", choices=["embed", "random", "none"])
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

print("loading images...", flush=True)
Itr = load_images([img(i) for i in tr.image_id])
Ite = load_images([img(i) for i in te.image_id])
print(f"  {Itr.shape} {Ite.shape} in {time.time()-t0:.0f}s", flush=True)

print("hand features...", flush=True)
Xtr, fnames = feature_matrix([img(i) for i in tr.image_id])
Xte, _ = feature_matrix([img(i) for i in te.image_id])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Atr = ((Xtr - mu) / sd).astype(np.float32)
Ate = ((Xte - mu) / sd).astype(np.float32)
print(f"  {Atr.shape}", flush=True)

# ---- subject groups ---------------------------------------------------------
def embed_groups(k=141):
    """Cluster the training faces by generic appearance to approximate animal identity."""
    import timm, torch.nn.functional as F
    from sklearn.cluster import AgglomerativeClustering
    from train_lib import to_tensor
    m = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg").eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(Itr), 128):
            embs.append(m(to_tensor(Itr[i:i + 128])).numpy())
    E = np.concatenate(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                  linkage="average").fit_predict(E)
    return lab, E

if a.groupmode == "embed":
    gpath = os.path.join(a.out, "groups.npy")
    if os.path.exists(gpath):
        groups = np.load(gpath)
    else:
        groups, E = embed_groups()
        np.save(gpath, groups)
        np.save(os.path.join(a.out, "emb.npy"), E)
elif a.groupmode == "random":
    groups = np.random.default_rng(0).integers(0, 141, len(y))
else:
    groups = np.arange(len(y))

sz = pd.Series(groups).value_counts()
print(f"groups: n={sz.size} mean {sz.mean():.2f} median {sz.median():.0f} max {sz.max()} "
      f"singletons {(sz==1).sum()}", flush=True)
wv = np.mean([np.var(y[groups == c]) for c in np.unique(groups) if (groups == c).sum() > 1])
rg = np.random.default_rng(1).permutation(groups)
wr = np.mean([np.var(y[rg == c]) for c in np.unique(rg) if (rg == c).sum() > 1])
print(f"  within-group target var {wv:.5f} vs shuffled {wr:.5f} -> ICC {1-wv/wr:+.3f}", flush=True)

# ---- CNN CV -----------------------------------------------------------------
cfg = dict(DEFAULT_CFG); cfg.update(epochs=a.epochs, backbone=a.backbone, seed=a.seed)
print(f"\ncfg: {cfg}\n", flush=True)
oof, tepred, nf = run_cv(Itr, y, Atr, groups, Ite, Ate, cfg, n_splits=a.folds)

print("\n=== per-readout OOF ===", flush=True)
cands = {k: oof[k] for k in READOUT_KEYS}
for j, n in enumerate(fnames):
    cands[f"h_{n}"] = Xtr[:, j]
for k in sorted(cands, key=lambda k: -max(score(cands[k], y), score(-cands[k], y))):
    sp, sn = score(cands[k], y), score(-cands[k], y)
    if max(sp, sn) > 0.03:
        print(f"  {k:20s} {max(sp,sn):.4f} ({'+' if sp>=sn else '-'})", flush=True)

print("\n=== greedy blend (CNN readouts only) ===", flush=True)
w1, s1, _ = greedy_blend({k: oof[k] for k in READOUT_KEYS}, y)
print(f"  -> {w1}  OOF {s1:.4f}", flush=True)

print("\n=== greedy blend (CNN + hand features) ===", flush=True)
w2, s2, bl = greedy_blend(cands, y, seed_pool=READOUT_KEYS)
print(f"  -> {w2}  OOF {s2:.4f}", flush=True)
s, C, R, b = score(bl, y, True)
print(f"  final OOF score {s:.4f}  C {C:.4f}  R {R:.4f}  b {b:.4f}", flush=True)

np.savez(os.path.join(a.out, f"oof_{a.tag}.npz"),
         **{f"oof_{k}": oof[k] for k in READOUT_KEYS},
         **{f"te_{k}": tepred[k] for k in READOUT_KEYS},
         y=y, groups=groups, Xtr=Xtr, Xte=Xte)
json.dump(dict(cfg=cfg, blend_cnn=w1, s_cnn=s1, blend_all={k: v for k, v in w2.items()},
               s_all=s2, folds=nf, secs=time.time() - t0),
          open(os.path.join(a.out, f"res_{a.tag}.json"), "w"), indent=1, default=float)
print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
