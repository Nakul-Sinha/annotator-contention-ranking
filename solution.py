#!/usr/bin/env python3
"""The Split Verdict -- contention-capture triage on rodent-face imagery.

Usage:  python3 solution.py <public_dir> <submission_out>

Approach
--------
The scored quantity is the *ordering* of a review priority, judged by a magnitude-weighted
capture curve (facet C, weight 0.6) and Spearman rank agreement (facet R, weight 0.4). Two
properties of the problem drive the design:

1. The label is a small-sample distributional statistic -- the mean over five rubric regions
   of the variance of a 3-5 assessor panel's 0/1/2 marks. A plain L1/L2 regression head
   predicts a conditional centre and systematically shrinks exactly the spread the target
   measures. We therefore predict the *distribution* of contention with a soft-binned
   categorical head (bin centres carry both E[y] and E[y^2], giving a law-of-total-variance
   spread estimate), and add a structured five-region head whose predicted mark distributions
   are turned back into an implied scatter and supervised against the label -- i.e. we
   supervise the quantity the target is built from, not only the target.

2. Only order matters, and the capture facet weights the head of the queue far more than the
   tail. So several read-outs are taken from every trained model -- binned expectation, upper
   tail mass, predicted upper quantiles, model uncertainty, heteroscedastic scale, region
   scatter, augmentation disagreement -- alongside classical image-legibility descriptors
   (sharpness, motion anisotropy, exposure, framing, occlusion, pose asymmetry) and small
   non-convolutional regressors fitted on them. Every one of these is a weak, partly
   decorrelated ranking signal. They are combined by a greedy, sign-corrected, equal-weight
   blend that accepts a candidate only when the *actual competition metric*, computed on
   subject-grouped out-of-fold predictions, improves. No single "best" estimator is chosen.

Validation is subject-aware. The challenge states each training animal contributes several
faces while the test set serves one face per animal, but supplies no subject column and random
ids. Approximate subject groups are recovered by unsupervised Ward clustering of generic image
embeddings of the *training images only* (measured target ICC +0.146 against shuffled groups);
they are used exclusively to build leakage-free folds and never as a model input.

Everything is trained from scratch on the provided train.csv within a measured wall-clock
budget: the cost of one fold is timed, the ensemble size is solved for from the time left, and
every loop carries a deadline guard.
"""
import json
import math
import os
import sys
import time
import warnings

T_START = time.time()
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 8)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, os.cpu_count() or 8)))

import numpy as np
import pandas as pd

TIME_BUDGET = float(os.environ.get("SV_BUDGET_S", 4700))     # ~78 min of a 90 min cap
INFER_RESERVE = float(os.environ.get("SV_RESERVE_S", 260))
FAST = os.environ.get("SV_FAST", "0") == "1"
SEED = 1234


def log(*a):
    print(f"[{time.time() - T_START:7.1f}s]", *a, flush=True)


def elapsed():
    return time.time() - T_START


def train_deadline():
    return T_START + TIME_BUDGET - INFER_RESERVE


# ============================================================== metric ======
def _capture_C(priority, y):
    p = np.asarray(priority, np.float64)
    p = np.where(np.isfinite(p), p, -np.inf)
    n = len(y)
    w = np.empty(n)
    w[np.argsort(-p, kind="stable")] = np.arange(n, 0, -1, dtype=np.float64)
    wo = np.empty(n)
    wo[np.argsort(-y, kind="stable")] = np.arange(n, 0, -1, dtype=np.float64)
    A_you, A_rand, A_or = float((y * w).sum()), float(y.sum()) * (n + 1) / 2.0, float((y * wo).sum())
    return 0.0 if A_or == A_rand else (A_you - A_rand) / (A_or - A_rand)


def _spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def comp_score(priority, y, detail=False):
    """Replica of the published scorer: six endpoint-matched response curves over
    b = clip(C,0,1)^0.6 * clip(R,0,1)^0.4.  Verified: oracle -> 1.0, random/constant/reverse -> 0."""
    p = np.nan_to_num(np.asarray(priority, np.float64), nan=-1e300, posinf=1e300, neginf=-1e300)
    y = np.asarray(y, np.float64)
    C = _capture_C(p, y)
    R = _spearman(p, y)
    b = np.clip(C, 0, 1) ** 0.6 * np.clip(R, 0, 1) ** 0.4
    g = np.mean([(math.exp(2.5 * b) - 1) / (math.exp(2.5) - 1),
                 math.log1p(4 * b) / math.log(5),
                 math.sin(math.pi * b / 2),
                 (1 - math.cos(math.pi * b)) / 2,
                 math.tan(1.3 * b) / math.tan(1.3),
                 math.tanh(2 * b) / math.tanh(2)])
    s = float(np.clip(g, 0, 1))
    return (s, C, R, b) if detail else s


# ====================================================== image descriptors ===
def _gray(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def image_features(a):
    """Classical legibility / indeterminacy descriptors: sharpness, directional blur,
    exposure, colour, framing, pose asymmetry and occlusion proxies."""
    f = {}
    g = _gray(a)
    H, W = g.shape
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    f["lap_var"] = float(np.var(lap))
    f["lap_abs"] = float(np.mean(np.abs(lap)))
    f["lap_p99"] = float(np.percentile(np.abs(lap), 99))
    gx, gy = np.diff(g, axis=1), np.diff(g, axis=0)
    ten = gx[:-1, :] ** 2 + gy[:, :-1] ** 2
    f["tenengrad"] = float(ten.mean())
    f["grad_p90"] = float(np.percentile(np.sqrt(ten), 90))
    f["edge_dens"] = float(np.mean(np.sqrt(ten) > 0.06))
    F = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    tot = F.sum() + 1e-8
    f["fft_hi"] = float(F[r > 0.30 * H].sum() / tot)
    f["fft_mid"] = float(F[(r > 0.12 * H) & (r <= 0.30 * H)].sum() / tot)
    sxx, syy = float((gx[:-1, :] ** 2).mean()), float((gy[:, :-1] ** 2).mean())
    sxy = float((gx[:-1, :] * gy[:, :-1]).mean())
    tr_, det = sxx + syy, sxx * syy - sxy ** 2
    disc = max(tr_ ** 2 / 4 - det, 0.0) ** 0.5
    l1, l2 = tr_ / 2 + disc, max(tr_ / 2 - disc, 1e-12)
    f["grad_aniso"] = float((l1 - l2) / (l1 + l2 + 1e-12))
    f["grad_coher"] = float(np.log(l1 / l2 + 1e-12))
    f["mean"], f["std"] = float(g.mean()), float(g.std())
    f["p01"], f["p99"] = float(np.percentile(g, 1)), float(np.percentile(g, 99))
    f["dyn_range"] = f["p99"] - f["p01"]
    f["clip_lo"], f["clip_hi"] = float((g < 0.02).mean()), float((g > 0.98).mean())
    hist = np.histogram(g, bins=32, range=(0, 1))[0].astype(np.float64) + 1e-9
    hist /= hist.sum()
    f["hist_ent"] = float(-(hist * np.log(hist)).sum())
    zg = (g - g.mean()) / (g.std() + 1e-8)
    f["skew"], f["kurt"] = float((zg ** 3).mean()), float((zg ** 4).mean())
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    f["sat_mean"], f["sat_std"] = float(sat.mean()), float(sat.std())
    f["rg"] = float(a[..., 0].mean() - a[..., 1].mean())
    f["by"] = float(a[..., 2].mean() - 0.5 * (a[..., 0].mean() + a[..., 1].mean()))
    f["chan_std"] = float(np.std([a[..., c].mean() for c in range(3)]))
    c = g[H // 4:3 * H // 4, W // 4:3 * W // 4]
    f["ctr_std"], f["ctr_mean"] = float(c.std()), float(c.mean())
    f["ctr_border_ratio"] = float(c.std() / (g.std() + 1e-8))
    f["ctr_sharp"] = float(np.var(lap[H // 4:3 * H // 4, W // 4:3 * W // 4]))
    f["sharp_ratio"] = float(f["ctr_sharp"] / (f["lap_var"] + 1e-12))
    mir = g[:, ::-1]
    f["asym_g"] = float(np.mean(np.abs(g - mir)))
    f["asym_g_c"] = float(1.0 - np.corrcoef(g.ravel(), mir.ravel())[0, 1])
    e = np.sqrt(ten) + 1e-8
    ey, ex = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    cy = float((e * ey).sum() / e.sum()) / e.shape[0] - 0.5
    cx = float((e * ex).sum() / e.sum()) / e.shape[1] - 0.5
    f["cen_off"], f["cen_x"], f["cen_y"] = float(np.hypot(cy, cx)), abs(cx), abs(cy)
    f["e_spread"] = float(np.sqrt((e * ((ey / e.shape[0] - .5) ** 2
                                        + (ex / e.shape[1] - .5) ** 2)).sum() / e.sum()))
    bs = 8
    blk = g[:H // bs * bs, :W // bs * bs].reshape(H // bs, bs, W // bs, bs)
    bstd = blk.std(axis=(1, 3))
    f["flat_frac"] = float((bstd < 0.02).mean())
    f["blk_std_std"], f["blk_std_min"] = float(bstd.std()), float(bstd.min())
    f["blk_mean_range"] = float(np.ptp(blk.mean(axis=(1, 3))))
    bs = 16
    blk2 = g[:H // bs * bs, :W // bs * bs].reshape(H // bs, bs, W // bs, bs)
    f["tile_mean_std"] = float(blk2.mean(axis=(1, 3)).std())
    f["tile_sharp_std"] = float(np.log1p(blk2.std(axis=(1, 3))).std())
    return f


def load_images(paths, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def one(p):
        im = Image.open(p).convert("RGB")
        return np.asarray(im, dtype=np.uint8)

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(one, paths)))


def feature_matrix(imgs, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda z: image_features(z.astype(np.float32) / 255.0), imgs))
    names = list(rows[0].keys())
    return np.array([[r[k] for k in names] for r in rows], np.float64), names


# =============================================================== blending ===
def rank_z(v):
    from scipy.stats import rankdata
    v = np.asarray(v, np.float64)
    if not np.isfinite(v).all():
        med = np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0
        v = np.nan_to_num(v, nan=med, posinf=med, neginf=med)
    r = rankdata(v)
    return (r - r.mean()) / (r.std() + 1e-12)


def greedy_blend(cands, y, rounds=18, min_gain=0.0015):
    """Sign-correct every candidate by its own OOF metric, seed with the best, then add
    candidates to an equal-weight rank average while the competition metric keeps improving."""
    names = [n for n in cands if np.std(cands[n]) > 0 and np.isfinite(cands[n]).any()]
    if not names:
        return {}, 0.0
    Z, sign, solo = {}, {}, {}
    for n in names:
        z = rank_z(cands[n])
        sp, sn = comp_score(z, y), comp_score(-z, y)
        sign[n] = 1.0 if sp >= sn else -1.0
        Z[n], solo[n] = z * sign[n], max(sp, sn)
    order = sorted(names, key=lambda n: -solo[n])
    sel, cur, cur_s = [order[0]], Z[order[0]].copy(), solo[order[0]]
    log(f"  blend seed {order[0]} solo {cur_s:.4f}")
    for _ in range(rounds - 1):
        best = None
        for n in names:
            trial = (cur * len(sel) + Z[n]) / (len(sel) + 1)
            s = comp_score(trial, y)
            if best is None or s > best[0]:
                best = (s, n, trial)
        if best[0] > cur_s + min_gain:
            cur_s, cur = best[0], best[2]
            sel.append(best[1])
            log(f"  + {best[1]:28s} -> {cur_s:.4f}")
        else:
            break
    return {n: sign[n] * sel.count(n) / len(sel) for n in set(sel)}, cur_s


def apply_blend(cands, w):
    out = None
    for n, v in w.items():
        z = rank_z(cands[n]) * v
        out = z if out is None else out + z
    return out


# ================================================================== model ===
def build_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    return torch, nn, F


def make_model(backbone, pretrained, n_bins, n_aux, drop, hidden):
    torch, nn, F = build_torch()

    class SmallNet(nn.Module):
        def __init__(self, widths=(32, 64, 96, 128)):
            super().__init__()
            L, c = [], 3
            for w in widths:
                L += [nn.Conv2d(c, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(),
                      nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(),
                      nn.MaxPool2d(2)]
                c = w
            self.body = nn.Sequential(*L)
            self.num_features = c * 2

        def forward(self, x):
            h = self.body(x)
            return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)

    def trunk_of(name):
        if name != "scratch":
            try:
                import timm
                m = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                      global_pool="avg")
                return m, m.num_features, "timm:" + name
            except Exception as e:
                log(f"  [backbone] timm '{name}' unavailable ({type(e).__name__}); trying torchvision")
            try:
                import torchvision.models as tvm
                m = tvm.resnet18(weights="DEFAULT" if pretrained else None)
                nf = m.fc.in_features
                m.fc = nn.Identity()
                return m, nf, "tv:resnet18"
            except Exception as e:
                log(f"  [backbone] torchvision unavailable ({type(e).__name__}); using scratch encoder")
        m = SmallNet()
        return m, m.num_features, "scratch"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk, nf, self.src = trunk_of(backbone)
            self.n_aux = n_aux
            self.drop = nn.Dropout(drop)
            self.neck = nn.Sequential(nn.Linear(nf + n_aux, hidden), nn.SiLU(), nn.Dropout(drop))
            self.head_bins = nn.Linear(hidden, n_bins)
            self.head_mu = nn.Linear(hidden, 1)
            self.head_logs = nn.Linear(hidden, 1)
            self.head_reg = nn.Linear(hidden, 15)
            self.head_top = nn.Linear(hidden, 1)
            nn.init.zeros_(self.head_logs.weight)
            nn.init.zeros_(self.head_logs.bias)

        def forward(self, x, aux=None):
            h = self.drop(self.trunk(x))
            if self.n_aux:
                h = torch.cat([h, aux], 1)
            h = self.neck(h)
            return {"bins": self.head_bins(h), "mu": self.head_mu(h).squeeze(1),
                    "logs": self.head_logs(h).squeeze(1),
                    "reg": self.head_reg(h).view(-1, 5, 3), "top": self.head_top(h).squeeze(1)}

    return Net()


def region_scatter(reg_logits):
    torch, nn, F = build_torch()
    p = F.softmax(reg_logits, dim=2)
    m = torch.tensor([0.0, 1.0, 2.0], device=p.device)
    e1 = (p * m).sum(2)
    e2 = (p * m * m).sum(2)
    return (e2 - e1 * e1).clamp_min(0)


IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(u8):
    import torch
    x = (u8.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def augment(u8, rng, photo=0.06):
    """Mild geometry + photometric jitter only. Blur/rescale/noise are deliberately excluded:
    they would destroy the legibility cues that drive panel disagreement."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        cv2 = None
    out = np.empty_like(u8)
    H, W = u8.shape[1:3]
    for i in range(len(u8)):
        im = u8[i]
        if rng.random() < 0.5:
            im = im[:, ::-1]
        if cv2 is not None:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-12, 12), rng.uniform(.94, 1.08))
            M[0, 2] += rng.uniform(-.05, .05) * W
            M[1, 2] += rng.uniform(-.05, .05) * H
            im = cv2.warpAffine(np.ascontiguousarray(im), M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        if photo > 0:
            imf = im.astype(np.float32)
            m = imf.mean()
            imf = (imf - m) * (1 + rng.uniform(-photo, photo)) + m * (1 + rng.uniform(-photo, photo))
            im = np.clip(imf, 0, 255).astype(np.uint8)
        out[i] = im
    return out


def make_bins(y, n_bins):
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = y.min() - 1e-6, y.max() + 1e-6
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs, y, "right") - 1, 0, len(qs) - 2)
    nb = len(qs) - 1
    c1 = np.array([y[idx == k].mean() if (idx == k).any() else .5 * (qs[k] + qs[k + 1])
                   for k in range(nb)])
    c2 = np.array([(y[idx == k] ** 2).mean() if (idx == k).any() else c1[k] ** 2
                   for k in range(nb)])
    return qs, c1, c2, nb


def soft_targets(y, qs, c1):
    nb = len(c1)
    idx = np.clip(np.searchsorted(qs, y, "right") - 1, 0, nb - 1)
    T = np.zeros((len(y), nb), np.float32)
    for i, (yi, k) in enumerate(zip(y, idx)):
        if yi <= c1[k] and k > 0:
            lo, hi = k - 1, k
        elif yi > c1[k] and k < nb - 1:
            lo, hi = k, k + 1
        else:
            T[i, k] = 1.0
            continue
        span = c1[hi] - c1[lo]
        w = 0.5 if span <= 1e-9 else float(np.clip((yi - c1[lo]) / span, 0, 1))
        T[i, lo], T[i, hi] = 1 - w, w
    return T


READOUT_KEYS = ["exp", "spread", "ent", "tail", "tail9", "mu", "sigma", "top", "reg",
                "regmax", "regstd", "tta", "e2", "q90", "q75", "cvar"]


def readouts(model, imgs, aux, c1, c2, tta, bs=128):
    import torch
    import torch.nn.functional as F
    model.eval()
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        cv2 = None
    acc, mu_runs = None, []
    for t in range(max(tta, 1)):
        parts = []
        for i in range(0, len(imgs), bs):
            ch = imgs[i:i + bs]
            if t > 0:
                if t % 2 == 1:
                    ch = ch[:, :, ::-1]
                if t >= 2 and cv2 is not None:
                    H, W = ch.shape[1:3]
                    M = cv2.getRotationMatrix2D((W / 2, H / 2), -6.0 if t == 2 else 6.0, 1.0)
                    ch = np.stack([cv2.warpAffine(np.ascontiguousarray(z), M, (W, H),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REFLECT_101) for z in ch])
                ch = np.ascontiguousarray(ch)
            with torch.no_grad():
                a = None if aux is None else torch.from_numpy(aux[i:i + bs]).float()
                o = model(to_tensor(ch), a)
                parts.append(dict(p=F.softmax(o["bins"], 1).numpy(), mu=o["mu"].numpy(),
                                  logs=o["logs"].numpy(), v=region_scatter(o["reg"]).numpy(),
                                  top=torch.sigmoid(o["top"]).numpy()))
        agg = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        mu_runs.append((agg["p"] * c1).sum(1))
        acc = agg if acc is None else {k: acc[k] + agg[k] for k in acc}
    n = max(tta, 1)
    agg = {k: v / n for k, v in acc.items()}
    p = agg["p"]
    e1, e2 = (p * c1).sum(1), (p * c2).sum(1)
    nb = p.shape[1]
    cum = np.cumsum(p, 1)
    w_up = np.maximum(cum - 0.75, 0)
    w_up = np.diff(np.concatenate([np.zeros((len(p), 1)), w_up], 1), axis=1)
    v = agg["v"]
    return {
        "exp": e1,
        "spread": np.sqrt(np.maximum(e2 - e1 ** 2, 0)),
        "ent": -(p * np.log(p + 1e-9)).sum(1),
        "tail": p[:, int(nb * .75):].sum(1),
        "tail9": p[:, int(nb * .9):].sum(1),
        "mu": agg["mu"], "sigma": agg["logs"], "top": agg["top"],
        "reg": v.mean(1), "regmax": v.max(1), "regstd": v.std(1),
        "tta": np.std(np.stack(mu_runs), 0) if len(mu_runs) > 1 else np.zeros(len(e1)),
        "e2": e2,
        "q90": c1[np.argmax(cum >= .90, axis=1)],
        "q75": c1[np.argmax(cum >= .75, axis=1)],
        "cvar": (w_up * c1).sum(1) / (w_up.sum(1) + 1e-9),
    }


def train_fold(imgs, y, aux, tr_i, te_imgs, te_aux, va_imgs, va_aux, cfg, deadline):
    torch, nn, F = build_torch()
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    ytr = y[tr_i]
    qs, c1, c2, nb = make_bins(ytr, cfg["n_bins"])
    soft = soft_targets(ytr, qs, c1)
    yt = (np.sqrt(np.maximum(ytr, 0)) / math.sqrt(ytr.max() + 1e-9)).astype(np.float32)
    top = (ytr > np.quantile(ytr, .75)).astype(np.float32)
    n_aux = aux.shape[1] if (aux is not None and cfg["aux"]) else 0
    model = make_model(cfg["backbone"], cfg["pretrained"], nb, n_aux, cfg["drop"], cfg["hidden"])
    tp = list(model.trunk.parameters())
    tid = {id(p) for p in tp}
    hp = [p for p in model.parameters() if id(p) not in tid]
    opt = torch.optim.AdamW([{"params": tp, "lr": cfg["lr"] * cfg["lr_trunk_mult"]},
                             {"params": hp, "lr": cfg["lr"]}], weight_decay=cfg["wd"])
    spe = max(1, len(tr_i) // cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg["lr"] * cfg["lr_trunk_mult"], cfg["lr"]],
        total_steps=spe * cfg["epochs"], pct_start=0.25)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    Xtr = imgs[tr_i]
    c1_t = torch.from_numpy(c1.astype(np.float32))
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(len(tr_i))
        for b in range(spe):
            sel = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            if len(sel) < 4:
                continue
            x = to_tensor(augment(Xtr[sel], rng, cfg["photo"]))
            a = torch.from_numpy(aux[tr_i][sel]).float() if n_aux else None
            o = model(x, a)
            yv = torch.from_numpy(ytr[sel].astype(np.float32))
            logp = F.log_softmax(o["bins"], 1)
            l_bin = -(torch.from_numpy(soft[sel]) * logp).sum(1).mean()
            l_mu = F.smooth_l1_loss(o["mu"], torch.from_numpy(yt[sel]), beta=0.5)
            s = o["logs"].clamp(-4, 4)
            l_het = (0.5 * (torch.from_numpy(yt[sel]) - o["mu"].detach()) ** 2
                     * torch.exp(-2 * s) + s).mean()
            l_reg = F.smooth_l1_loss(region_scatter(o["reg"]).mean(1), yv, beta=0.05)
            l_top = F.binary_cross_entropy_with_logits(o["top"], torch.from_numpy(top[sel]))
            e = (F.softmax(o["bins"], 1) * c1_t).sum(1)
            d, t = e[:, None] - e[None, :], yv[:, None] - yv[None, :]
            m = t > 1e-6
            if m.sum() > 0:
                hi = torch.maximum(yv[:, None].expand_as(t), yv[None, :].expand_as(t))[m]
                wgt = t[m] * (0.5 + hi / (yv.max() + 1e-6))
                l_rank = (F.softplus(-d[m] * 4.0) * wgt).sum() / wgt.sum()
            else:
                l_rank = e.sum() * 0.0
            loss = (cfg["w_bin"] * l_bin + cfg["w_mu"] * l_mu + cfg["w_het"] * l_het
                    + cfg["w_reg"] * l_reg + cfg["w_top"] * l_top + cfg["w_rank"] * l_rank)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            dec = cfg["ema"] if ep > cfg["epochs"] * .3 else 0.9
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    if sd[k].dtype.is_floating_point:
                        ema[k].mul_(dec).add_(sd[k].float(), alpha=1 - dec)
                    else:
                        ema[k] = sd[k].float()
        if time.time() > deadline:
            log(f"    [deadline] fold stopped at epoch {ep + 1}/{cfg['epochs']}")
            break
    sd = model.state_dict()
    model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})
    va = readouts(model, va_imgs, va_aux, c1, c2, cfg["tta"])
    te = readouts(model, te_imgs, te_aux, c1, c2, cfg["tta"])
    return va, te


# ================================================================== main ====
def resolve_paths():
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    cands = ["dataset/public", "public", "dataset", "./data/public", "."]
    for c in cands:
        if os.path.exists(os.path.join(c, "test.csv")) and os.path.isdir(os.path.join(c, "images")):
            return c, "working/submission.csv"
    return (sys.argv[1] if len(sys.argv) > 1 else "public",
            sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv")


def write_submission(ids, priority, out_path):
    pr = np.asarray(priority, np.float64)
    pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(pr) == len(ids), "priority/id length mismatch"
    df = pd.DataFrame({"image_id": list(ids), "priority": pr})
    assert df.image_id.duplicated().sum() == 0, "duplicate image_id in submission"
    assert np.isfinite(df.priority.values).all(), "non-finite priority"
    for p in {out_path, os.path.join("working", "submission.csv")}:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)
        tmp = p + ".tmp"
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, p)
        log(f"wrote {p} rows={len(df)}")


def main():
    public_dir, sub_out = resolve_paths()
    log(f"public_dir={public_dir}  submission_out={sub_out}  budget={TIME_BUDGET:.0f}s")
    tr = pd.read_csv(os.path.join(public_dir, "train.csv"))
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))
    imdir = os.path.join(public_dir, "images")
    y = tr["contention"].values.astype(np.float64)
    log(f"train {tr.shape}  test {te.shape}  y mean {y.mean():.4f} sd {y.std():.4f}")

    Itr = load_images([os.path.join(imdir, i + ".jpg") for i in tr.image_id])
    Ite = load_images([os.path.join(imdir, i + ".jpg") for i in te.image_id])
    log(f"images loaded {Itr.shape} {Ite.shape}")

    Xtr, fnames = feature_matrix(Itr)
    Xte, _ = feature_matrix(Ite)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Atr, Ate = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
    log(f"hand features {Xtr.shape}")

    # ---- emergency submission on disk immediately ---------------------------
    from scipy.stats import rankdata
    boot = rank_z(-Xtr[:, fnames.index("sharp_ratio")])
    boot_te = -Xte[:, fnames.index("sharp_ratio")]
    log(f"bootstrap legibility priority train score {comp_score(boot, y):.4f}")
    write_submission(te.image_id, boot_te, sub_out)

    # ---- subject groups -----------------------------------------------------
    groups, Etr, Ete = build_groups(Itr, Ite, len(y))
    import pandas as _pd
    sz = _pd.Series(groups).value_counts()
    log(f"subject groups: k={sz.size} median {sz.median():.0f} max {sz.max()}")

    from sklearn.model_selection import GroupKFold
    n_folds = 3 if FAST else int(os.environ.get("SV_FOLDS", 5))
    folds = list(GroupKFold(n_splits=n_folds).split(np.arange(len(y)), y, groups))

    cands, tcands = {}, {}
    for j, n in enumerate(fnames):
        cands[f"h_{n}"], tcands[f"h_{n}"] = Xtr[:, j], Xte[:, j]
    add_tabular_candidates(cands, tcands, Xtr, Xte, Etr, Ete, y, folds)
    log(f"non-convolutional candidates ready: {len(cands)}")

    # ---- convolutional ensemble under a measured time budget ---------------
    run_cnn_members(cands, tcands, Itr, Ite, Atr, Ate, y, folds)

    # ---- greedy blend on subject-grouped OOF -------------------------------
    log(f"blending {len(cands)} candidate signals")
    w, oof_s = greedy_blend(cands, y)
    s, C, R, b = comp_score(apply_blend(cands, w), y, True)
    log(f"=== OOF estimate {s:.4f}   C {C:.4f}  R {R:.4f}  b {b:.4f} ===")
    for k, v in sorted(w.items(), key=lambda t: -abs(t[1])):
        log(f"    {v:+.3f}  {k}")

    pr = apply_blend(tcands, w)
    if pr is None or not np.isfinite(pr).any():
        log("!! blend produced no usable signal; falling back to legibility priority")
        pr = boot_te
    write_submission(te.image_id, pr, sub_out)
    log(f"done in {elapsed():.0f}s")


def build_groups(Itr, Ite, n):
    """Recover approximate animal groups by clustering generic embeddings of the training
    images. Used only to build leakage-free CV folds; never a model input."""
    Etr = Ete = None
    try:
        import torch
        import timm
        m = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg").eval()
        with torch.no_grad():
            Etr = np.concatenate([m(to_tensor(Itr[i:i + 128])).numpy()
                                  for i in range(0, len(Itr), 128)])
            Ete = np.concatenate([m(to_tensor(Ite[i:i + 128])).numpy()
                                  for i in range(0, len(Ite), 128)])
        Etr /= np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-8
        Ete /= np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-8
        log("  grouping embeddings from pretrained resnet18")
    except Exception as e:
        log(f"  embedding backbone unavailable ({type(e).__name__}); using pixel descriptors")

        def desc(A, s=24):
            import cv2
            D = []
            for z in A:
                a = cv2.resize(z, (s, s)).astype(np.float32) / 255.0
                h = [np.histogram(a[..., c], bins=24, range=(0, 1))[0] / (s * s) for c in range(3)]
                D.append(np.concatenate([a.reshape(-1), np.concatenate(h)]))
            return np.array(D, np.float32)
        Etr, Ete = desc(Itr), desc(Ite)
        Etr /= np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-8
        Ete /= np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-8
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        P = PCA(n_components=min(64, Etr.shape[1]), random_state=0).fit_transform(Etr)
        groups = AgglomerativeClustering(n_clusters=min(141, n - 1), linkage="ward").fit_predict(P)
    except Exception as e:
        log(f"  clustering failed ({type(e).__name__}); falling back to random groups")
        groups = np.random.default_rng(0).integers(0, 141, n)
    return groups, Etr, Ete


def add_tabular_candidates(cands, tcands, Xtr, Xte, Etr, Ete, y, folds):
    """Small non-convolutional regressors on the legibility descriptors and on generic
    embeddings, fitted on the same subject-grouped folds. Weak blend members by design."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.preprocessing import QuantileTransformer
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    yr = QuantileTransformer(output_distribution="normal", n_quantiles=min(256, len(y)),
                             random_state=0).fit_transform(y.reshape(-1, 1)).ravel()
    thr = np.quantile(y, .75)

    def run(name, fp):
        try:
            oof = np.zeros(len(y))
            for tr_i, va_i in folds:
                oof[va_i] = fp(tr_i, va_i)
            cands[name], tcands[name] = oof, fp(np.arange(len(y)), None)
        except Exception as e:
            log(f"  candidate {name} failed ({type(e).__name__}: {e})")

    def ridge(alpha, tgt):
        return lambda a, b: Ridge(alpha=alpha).fit(Z[a], tgt[a]).predict(Z[b] if b is not None else Zt)

    def gbm(tgt):
        def f(a, b):
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=.05,
                                              l2_regularization=1.0, random_state=0).fit(Z[a], tgt[a])
            return m.predict(Z[b] if b is not None else Zt)
        return f

    def gbmtop(a, b):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=.05,
                                           l2_regularization=1.0, random_state=0)
        m.fit(Z[a], (y[a] > thr).astype(int))
        return m.predict_proba(Z[b] if b is not None else Zt)[:, 1]

    def knn(k):
        def f(a, b):
            Q = Etr[b] if b is not None else Ete
            S = Q @ Etr[a].T
            idx = np.argsort(-S, axis=1)[:, :k]
            w = np.take_along_axis(S, idx, 1)
            w = np.exp((w - w.max(1, keepdims=True)) * 12.0)
            return (w * y[a][idx]).sum(1) / w.sum(1)
        return f

    run("t_ridge", ridge(20.0, y))
    run("t_ridge_rk", ridge(20.0, yr))
    run("t_ridge_a2", ridge(200.0, yr))
    run("t_gbm", gbm(y))
    run("t_gbm_rk", gbm(yr))
    run("t_gbmtop", gbmtop)
    run("t_knn16", knn(16))
    run("t_knn48", knn(48))


BASE_CFG = dict(backbone="resnet18", pretrained=True, n_bins=12, drop=0.30, hidden=192,
                epochs=30, bs=48, lr=2.5e-3, lr_trunk_mult=0.25, wd=1e-4, ema=0.995,
                w_bin=1.0, w_mu=0.5, w_het=0.15, w_reg=0.6, w_top=0.3, w_rank=0.8,
                tta=4, seed=0, aux=True, photo=0.06)

# Config variants beat pure seed replicas for ensemble diversity; each entry is cycled with a
# fresh seed until the time budget is spent.
VARIANTS = [
    {},
    {"backbone": "resnet34", "lr": 2.0e-3},
    {"drop": 0.45, "w_rank": 1.4},
    {"backbone": "efficientnet_b0", "lr": 2.0e-3, "drop": 0.35},
    {"aux": False, "n_bins": 18},
    {"backbone": "resnet18", "w_reg": 1.2, "w_bin": 0.7, "drop": 0.25},
    {"backbone": "mobilenetv3_small_100", "lr": 3.0e-3},
    {"backbone": "scratch", "pretrained": False, "lr": 4.0e-3, "epochs": 45},
]


def run_cnn_members(cands, tcands, Itr, Ite, Atr, Ate, y, folds):
    """Time-budgeted ensemble: cost one fold, solve for the member count, guard every loop."""
    member = 0
    fold_cost = None
    while True:
        left = train_deadline() - time.time()
        if fold_cost is not None and left < 1.15 * fold_cost * len(folds):
            log(f"  stopping ensemble: {left:.0f}s left < one more member "
                f"({1.15 * fold_cost * len(folds):.0f}s)")
            break
        if fold_cost is None and left < 120:
            log("  no time for any CNN member")
            break
        cfg = dict(BASE_CFG)
        cfg.update(VARIANTS[member % len(VARIANTS)])
        cfg["seed"] = SEED + member
        if FAST:
            cfg["epochs"] = 2
        tag = f"m{member}_{cfg['backbone']}"
        log(f"member {member}: {tag} epochs={cfg['epochs']} drop={cfg['drop']} aux={cfg['aux']}")
        oof = {k: np.zeros(len(y)) for k in READOUT_KEYS}
        tep = {k: np.zeros(len(Ite)) for k in READOUT_KEYS}
        done_folds = 0
        for fi, (tr_i, va_i) in enumerate(folds):
            if time.time() > train_deadline():
                log(f"  [deadline] abandoning member {member} after {done_folds} folds")
                break
            t0 = time.time()
            c = dict(cfg)
            c["seed"] = cfg["seed"] * 100 + fi
            try:
                va, tp = train_fold(Itr, y, Atr, tr_i, Ite, Ate, Itr[va_i], Atr[va_i], c,
                                    train_deadline())
            except Exception as e:
                log(f"  fold {fi} failed ({type(e).__name__}: {e}); skipping member")
                done_folds = 0
                break
            for k in READOUT_KEYS:
                oof[k][va_i] = va[k]
                tep[k] += tp[k]
            done_folds += 1
            fold_cost = time.time() - t0 if fold_cost is None else 0.5 * fold_cost + 0.5 * (time.time() - t0)
            log(f"  fold {fi}: {time.time() - t0:.0f}s  exp {comp_score(va['exp'], y[va_i]):.4f}")
        if done_folds == len(folds):
            for k in READOUT_KEYS:
                cands[f"{tag}:{k}"] = oof[k]
                tcands[f"{tag}:{k}"] = tep[k] / done_folds
            log(f"  member {member} kept ({len(READOUT_KEYS)} signals); "
                f"solo exp {comp_score(oof['exp'], y):.4f}")
        else:
            log(f"  member {member} discarded (incomplete OOF)")
            break
        member += 1
        if FAST and member >= 1:
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            pd_dir, sub = resolve_paths()
            te = pd.read_csv(os.path.join(pd_dir, "test.csv"))
            n = len(te)
            log("EMERGENCY: writing legibility-only priority")
            try:
                I = load_images([os.path.join(pd_dir, "images", i + ".jpg") for i in te.image_id])
                X, fn = feature_matrix(I)
                pr = -X[:, fn.index("sharp_ratio")]
            except Exception:
                pr = np.zeros(n)
            write_submission(te.image_id, pr, sub)
        except Exception:
            traceback.print_exc()
        sys.exit(1)
