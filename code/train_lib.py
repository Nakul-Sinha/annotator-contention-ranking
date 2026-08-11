"""Core modelling library for contention triage.

Design notes
------------
* Target `contention` is the mean over five rubric regions of the variance of the panel's
  0/1/2 marks. It is therefore a *distributional* statistic of a small panel (3-5 assessors),
  so a plain L1/L2 regression head predicts a shrunk conditional centre. We follow the
  law-of-total-variance recipe: a soft-binned categorical head gives E[y] and E[y^2] from one
  output, and a structured "five region" head predicts, for each rubric region, the categorical
  distribution of marks whose implied variance is averaged and supervised against the label.
  That supervises the quantity the target is *made of*, not just the target.
* Augmentation is deliberately mild and geometry-only: blur/rescale/noise would destroy the
  very legibility cues (sharpness, motion smear) that drive panel disagreement.
* Every head yields several read-outs; all of them enter the OOF blend as candidates.
"""
import math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- data ------
def load_images(paths, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def one(p):
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(one, paths)))


def to_tensor(batch_u8, mean=IMNET_MEAN, std=IMNET_STD):
    x = batch_u8.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def augment(batch_u8, rng, geom=True, photo=0.06):
    """Mild geometric + photometric jitter. No blur/rescale: those erase the legibility cue."""
    import cv2
    out = np.empty_like(batch_u8)
    H, W = batch_u8.shape[1:3]
    for i in range(len(batch_u8)):
        im = batch_u8[i]
        if rng.random() < 0.5:
            im = im[:, ::-1]
        if geom:
            ang = rng.uniform(-12, 12)
            sc = rng.uniform(0.94, 1.08)
            tx, ty = rng.uniform(-0.05, 0.05, 2) * W
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
            M[0, 2] += tx
            M[1, 2] += ty
            im = cv2.warpAffine(np.ascontiguousarray(im), M, (W, H),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if photo > 0:
            b = 1.0 + rng.uniform(-photo, photo)
            c = 1.0 + rng.uniform(-photo, photo)
            imf = im.astype(np.float32)
            m = imf.mean()
            imf = (imf - m) * c + m * b
            im = np.clip(imf, 0, 255).astype(np.uint8)
        out[i] = im
    return out


# ---------------------------------------------------------------- model -----
class SmallNet(nn.Module):
    """Compact from-scratch encoder (fallback when no pretrained weights are reachable)."""

    def __init__(self, widths=(32, 64, 96, 128), in_ch=3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(),
                       nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(),
                       nn.MaxPool2d(2)]
            c = w
        self.body = nn.Sequential(*layers)
        self.num_features = c * 2

    def forward(self, x):
        h = self.body(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class Backbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        self.kind = "small"
        if name != "scratch":
            try:
                import timm
                self.net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                             global_pool="avg")
                self.num_features = self.net.num_features
                self.kind = "timm"
            except Exception as e:  # offline / missing weights -> scratch encoder
                print(f"  [backbone] {name} unavailable ({type(e).__name__}: {e}); using scratch",
                      flush=True)
        if self.kind == "small":
            self.net = SmallNet()
            self.num_features = self.net.num_features

    def forward(self, x):
        return self.net(x)


class ContentionNet(nn.Module):
    """One trunk, four heads: binned distribution, scalar mean, heteroscedastic scale,
    and a structured five-region rubric head whose implied scatter reconstructs the target."""

    def __init__(self, backbone="resnet18", pretrained=True, n_bins=12, n_aux=0,
                 drop=0.30, hidden=192):
        super().__init__()
        self.trunk = Backbone(backbone, pretrained)
        d = self.trunk.num_features + n_aux
        self.n_aux = n_aux
        self.drop = nn.Dropout(drop)
        self.neck = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(drop))
        self.head_bins = nn.Linear(hidden, n_bins)
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logs = nn.Linear(hidden, 1)
        self.head_reg = nn.Linear(hidden, 15)      # 5 regions x 3 marks
        self.head_top = nn.Linear(hidden, 1)       # P(top-quantile contention)
        nn.init.zeros_(self.head_logs.weight); nn.init.zeros_(self.head_logs.bias)

    def forward(self, x, aux=None):
        h = self.drop(self.trunk(x))
        if self.n_aux:
            h = torch.cat([h, aux], 1)
        h = self.neck(h)
        return {
            "bins": self.head_bins(h),
            "mu": self.head_mu(h).squeeze(1),
            "logs": self.head_logs(h).squeeze(1),
            "reg": self.head_reg(h).view(-1, 5, 3),
            "top": self.head_top(h).squeeze(1),
        }


MARKS = torch.tensor([0.0, 1.0, 2.0])


def region_scatter(reg_logits):
    """Mean over the five rubric regions of the variance of the predicted mark distribution."""
    p = F.softmax(reg_logits, dim=2)                       # B,5,3
    m = MARKS.to(p.device)
    e1 = (p * m).sum(2)
    e2 = (p * m * m).sum(2)
    v = (e2 - e1 * e1).clamp_min(0)                        # B,5
    return v


# ---------------------------------------------------------------- targets ---
def make_bins(y, n_bins=12):
    """Quantile bins on the target; centres are within-bin means of y and y^2."""
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = y.min() - 1e-6, y.max() + 1e-6
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs, y, side="right") - 1, 0, len(qs) - 2)
    nb = len(qs) - 1
    c1 = np.array([y[idx == k].mean() if (idx == k).any() else 0.5 * (qs[k] + qs[k + 1])
                   for k in range(nb)])
    c2 = np.array([(y[idx == k] ** 2).mean() if (idx == k).any() else c1[k] ** 2
                   for k in range(nb)])
    return qs, c1, c2, nb


def soft_bin_targets(y, qs, c1):
    """Two-bin linear interpolation between neighbouring bin centres (soft labels)."""
    nb = len(c1)
    idx = np.clip(np.searchsorted(qs, y, side="right") - 1, 0, nb - 1)
    T = np.zeros((len(y), nb), dtype=np.float32)
    for i, (yi, k) in enumerate(zip(y, idx)):
        if yi <= c1[k] and k > 0:
            lo, hi = k - 1, k
        elif yi > c1[k] and k < nb - 1:
            lo, hi = k, k + 1
        else:
            T[i, k] = 1.0
            continue
        span = c1[hi] - c1[lo]
        w = 0.5 if span <= 1e-9 else np.clip((yi - c1[lo]) / span, 0, 1)
        T[i, lo] = 1 - w
        T[i, hi] = w
    return T


# ---------------------------------------------------------------- loss ------
def pairwise_rank_loss(pred, y, weight_by_gap=True):
    """Soft pairwise ranking loss over the batch, weighted by the target gap so that
    getting the most-contentious faces to the head of the queue dominates."""
    d = pred[:, None] - pred[None, :]
    t = y[:, None] - y[None, :]
    m = t > 1e-6
    if m.sum() == 0:
        return pred.sum() * 0.0
    l = F.softplus(-d[m] * 4.0)
    if weight_by_gap:
        w = t[m]
        # magnitude weight: pairs involving a high-contention face count more (mirrors facet C)
        hi = torch.maximum(y[:, None].expand_as(t), y[None, :].expand_as(t))[m]
        w = w * (0.5 + hi / (y.max() + 1e-6))
        return (l * w).sum() / w.sum()
    return l.mean()


def compute_loss(out, tgt, cfg):
    logp = F.log_softmax(out["bins"], 1)
    l_bin = -(tgt["soft"] * logp).sum(1).mean()
    l_mu = F.smooth_l1_loss(out["mu"], tgt["yt"], beta=0.5)
    s = out["logs"].clamp(-4, 4)
    l_het = (0.5 * ((tgt["yt"] - out["mu"].detach()) ** 2) * torch.exp(-2 * s) + s).mean()
    v = region_scatter(out["reg"])
    l_reg = F.smooth_l1_loss(v.mean(1), tgt["y"], beta=0.05)
    l_top = F.binary_cross_entropy_with_logits(out["top"], tgt["top"])
    p = F.softmax(out["bins"], 1)
    e = (p * tgt["c1"]).sum(1)
    l_rank = pairwise_rank_loss(e, tgt["y"])
    total = (cfg["w_bin"] * l_bin + cfg["w_mu"] * l_mu + cfg["w_het"] * l_het
             + cfg["w_reg"] * l_reg + cfg["w_top"] * l_top + cfg["w_rank"] * l_rank)
    return total, dict(bin=l_bin.item(), mu=l_mu.item(), het=l_het.item(),
                       reg=l_reg.item(), top=l_top.item(), rank=l_rank.item())


# ---------------------------------------------------------------- readouts --
@torch.no_grad()
def readouts(model, imgs_u8, aux, c1, c2, bs=128, tta=4, device="cpu", seed=0):
    """Return a dict of decorrelated candidate ranking signals for a set of images."""
    model.eval()
    rng = np.random.default_rng(seed)
    acc = None
    mu_runs = []
    for t in range(max(tta, 1)):
        preds = []
        for i in range(0, len(imgs_u8), bs):
            chunk = imgs_u8[i:i + bs]
            if t > 0:
                ch = chunk[:, :, ::-1] if t % 2 == 1 else chunk
                if t >= 2:
                    import cv2
                    H, W = ch.shape[1:3]
                    ang = (-6.0 if t == 2 else 6.0)
                    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
                    ch = np.stack([cv2.warpAffine(np.ascontiguousarray(z), M, (W, H),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REFLECT_101) for z in ch])
                chunk = np.ascontiguousarray(ch)
            x = to_tensor(chunk).to(device)
            a = None if aux is None else torch.from_numpy(aux[i:i + bs]).float().to(device)
            o = model(x, a)
            p = F.softmax(o["bins"], 1).cpu().numpy()
            v = region_scatter(o["reg"]).cpu().numpy()
            preds.append(dict(p=p, mu=o["mu"].cpu().numpy(), logs=o["logs"].cpu().numpy(),
                              v=v, top=torch.sigmoid(o["top"]).cpu().numpy()))
        agg = {k: np.concatenate([q[k] for q in preds]) for k in preds[0]}
        mu_runs.append((agg["p"] * c1).sum(1))
        acc = agg if acc is None else {k: acc[k] + agg[k] for k in acc}
    n = max(tta, 1)
    agg = {k: v / n for k, v in acc.items()}
    p = agg["p"]
    e1 = (p * c1).sum(1)
    e2 = (p * c2).sum(1)
    spread = np.sqrt(np.maximum(e2 - e1 ** 2, 0))              # law of total variance
    ent = -(p * np.log(p + 1e-9)).sum(1)
    nb = p.shape[1]
    tail = p[:, int(nb * 0.75):].sum(1)                        # P(top quartile of contention)
    tail9 = p[:, int(nb * 0.9):].sum(1)
    # magnitude-weighted read-outs: facet C rewards the head of the queue, so score the
    # predicted distribution with increasing emphasis on its upper mass.
    e2r = (p * c2).sum(1)                                      # E[y^2]
    cum = np.cumsum(p, 1)
    q90 = c1[np.argmax(cum >= 0.90, axis=1)]                   # 90th pct of predicted dist
    q75 = c1[np.argmax(cum >= 0.75, axis=1)]
    w_up = np.maximum(cum - 0.75, 0)
    w_up = np.diff(np.concatenate([np.zeros((len(p), 1)), w_up], 1), axis=1)
    cvar = (w_up * c1).sum(1) / (w_up.sum(1) + 1e-9)           # mean of the upper quarter
    v = agg["v"]
    return {
        "cnn_exp": e1,                       # binned expectation
        "cnn_spread": spread,                # model's own uncertainty (LTV)
        "cnn_ent": ent,                      # entropy of the predicted contention distribution
        "cnn_tail": tail,                    # head-of-queue oriented tail mass
        "cnn_tail9": tail9,
        "cnn_mu": agg["mu"],                 # scalar regression head
        "cnn_sigma": agg["logs"],            # heteroscedastic scale head
        "cnn_top": agg["top"],               # explicit top-quantile classifier
        "cnn_reg": v.mean(1),                # five-region implied scatter
        "cnn_regmax": v.max(1),              # worst single region
        "cnn_regstd": v.std(1),              # disagreement across regions
        "cnn_tta": np.std(np.stack(mu_runs), 0) if len(mu_runs) > 1 else np.zeros(len(e1)),
        "cnn_e2": e2r,
        "cnn_q90": q90,
        "cnn_q75": q75,
        "cnn_cvar": cvar,
        "cnn_regent": -(np.log(np.maximum(v, 1e-6))).mean(1),  # log-scale region scatter
    }


READOUT_KEYS = ["cnn_exp", "cnn_spread", "cnn_ent", "cnn_tail", "cnn_tail9", "cnn_mu",
                "cnn_sigma", "cnn_top", "cnn_reg", "cnn_regmax", "cnn_regstd", "cnn_tta",
                "cnn_e2", "cnn_q90", "cnn_q75", "cnn_cvar", "cnn_regent"]
