"""Measure the epoch response: full subject-grouped OOF evaluated at several epoch counts
from a single training run per fold (checkpoints, not restarts)."""
import argparse, json, os, sys, time, math
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from feats import feature_matrix
from blend import greedy_blend
from train_lib import (ContentionNet, augment, compute_loss, load_images, make_bins,
                       readouts, soft_bin_targets, READOUT_KEYS)
from runner import DEFAULT_CFG

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--checks", default="8,16,26,40,56")
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--tag", default="ep")
ap.add_argument("--nopre", action="store_true")
ap.add_argument("--lr", type=float, default=2.5e-3)
ap.add_argument("--drop", type=float, default=0.30)
a = ap.parse_args()
CHECKS = [int(x) for x in a.checks.split(",")]
MAXEP = max(CHECKS)
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
try:
    import cv2; cv2.setNumThreads(1)
except Exception:
    pass

tr = pd.read_csv(os.path.join(a.data, "train.csv"))
te = pd.read_csv(os.path.join(a.data, "test.csv"))
img = lambda i: os.path.join(a.data, "images", i + ".jpg")
y = tr.contention.values.astype(np.float64)
Itr = load_images([img(i) for i in tr.image_id]); Ite = load_images([img(i) for i in te.image_id])
z = np.load(os.path.join(a.out, "handfeat.npz"), allow_pickle=True) \
    if os.path.exists(os.path.join(a.out, "handfeat.npz")) else None
if z is None:
    Xtr, fn = feature_matrix([img(i) for i in tr.image_id]); Xte, _ = feature_matrix([img(i) for i in te.image_id])
    np.savez(os.path.join(a.out, "handfeat.npz"), Xtr=Xtr, Xte=Xte, fnames=np.array(fn, dtype=object))
else:
    Xtr, Xte = z["Xtr"], z["Xte"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Atr, Ate = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
groups = np.load(os.path.join(a.out, "groups.npy"))

cfg = dict(DEFAULT_CFG); cfg.update(backbone=a.backbone, epochs=MAXEP, pretrained=not a.nopre,
                                    lr=a.lr, drop=a.drop)
OOF = {c: {k: np.zeros(len(y)) for k in READOUT_KEYS} for c in CHECKS}
TEP = {c: {k: np.zeros(len(Ite)) for k in READOUT_KEYS} for c in CHECKS}
t00 = time.time()

for fi, (tr_i, va_i) in enumerate(GroupKFold(n_splits=a.folds).split(np.arange(len(y)), y, groups)):
    t0 = time.time()
    seed = 100 + fi
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ytr = y[tr_i]
    qs, c1, c2, nb = make_bins(ytr, cfg["n_bins"])
    soft = soft_bin_targets(ytr, qs, c1)
    yt = (np.sqrt(np.maximum(ytr, 0)) / math.sqrt(ytr.max() + 1e-9)).astype(np.float32)
    top = (ytr > np.quantile(ytr, .75)).astype(np.float32)
    model = ContentionNet(cfg["backbone"], cfg["pretrained"], nb, Atr.shape[1], cfg["drop"],
                          cfg["hidden"])
    tp = list(model.trunk.parameters()); tid = {id(p) for p in tp}
    hp = [p for p in model.parameters() if id(p) not in tid]
    opt = torch.optim.AdamW([{"params": tp, "lr": cfg["lr"] * cfg["lr_trunk_mult"]},
                             {"params": hp, "lr": cfg["lr"]}], weight_decay=cfg["wd"])
    spe = max(1, len(tr_i) // cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg["lr"] * cfg["lr_trunk_mult"], cfg["lr"]],
        total_steps=spe * MAXEP, pct_start=0.25)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    c1_t = torch.from_numpy(c1.astype(np.float32))
    Xf = Itr[tr_i]
    for ep in range(MAXEP):
        model.train()
        perm = rng.permutation(len(tr_i))
        for b in range(spe):
            sel = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            if len(sel) < 4: continue
            from train_lib import to_tensor
            x = to_tensor(augment(Xf[sel], rng, photo=cfg["photo"]))
            tgt = dict(soft=torch.from_numpy(soft[sel]), yt=torch.from_numpy(yt[sel]),
                       y=torch.from_numpy(ytr[sel].astype(np.float32)),
                       top=torch.from_numpy(top[sel]), c1=c1_t)
            out = model(x, torch.from_numpy(Atr[tr_i][sel]).float())
            loss, _ = compute_loss(out, tgt, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            d = cfg["ema"] if ep > MAXEP * .3 else 0.9
            with torch.no_grad():
                sdm = model.state_dict()
                for k in ema:
                    if sdm[k].dtype.is_floating_point: ema[k].mul_(d).add_(sdm[k].float(), alpha=1 - d)
                    else: ema[k] = sdm[k].float()
        if (ep + 1) in CHECKS:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(live[k].dtype) for k, v in ema.items()})
            va = readouts(model, Itr[va_i], Atr[va_i], c1, c2, tta=cfg["tta"], seed=seed)
            tt = readouts(model, Ite, Ate, c1, c2, tta=cfg["tta"], seed=seed)
            for k in READOUT_KEYS:
                OOF[ep + 1][k][va_i] = va[k]; TEP[ep + 1][k] += tt[k]
            model.load_state_dict(live)
            print(f"  fold {fi} @ep{ep+1}: exp {score(va['cnn_exp'], y[va_i]):.4f} "
                  f"tail9 {score(va['cnn_tail9'], y[va_i]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f" fold {fi} total {time.time()-t0:.0f}s", flush=True)

print("\n=== epoch response (full subject-grouped OOF) ===", flush=True)
best = None
for c in CHECKS:
    for k in READOUT_KEYS: TEP[c][k] /= a.folds
    solo = {k: max(score(OOF[c][k], y), score(-OOF[c][k], y)) for k in READOUT_KEYS}
    w, s, _ = greedy_blend(OOF[c], y, verbose=False)
    top3 = sorted(solo, key=lambda k: -solo[k])[:3]
    print(f"  ep{c:3d}: blend {s:.4f} | " + " ".join(f"{k} {solo[k]:.4f}" for k in top3), flush=True)
    np.savez(os.path.join(a.out, f"oof_{a.tag}{c}.npz"),
             **{f"oof_{k}": OOF[c][k] for k in READOUT_KEYS},
             **{f"te_{k}": TEP[c][k] for k in READOUT_KEYS}, y=y, groups=groups)
    if best is None or s > best[1]: best = (c, s)
print(f"best epochs {best[0]} blend {best[1]:.4f}   total {time.time()-t00:.0f}s", flush=True)
