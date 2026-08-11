"""Fold runner: subject-grouped OOF training + candidate read-out extraction."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import (ContentionNet, augment, compute_loss, load_images, make_bins,
                       readouts, soft_bin_targets, to_tensor, READOUT_KEYS)
from metric import score

DEFAULT_CFG = dict(
    backbone="resnet18", pretrained=True, n_bins=12, drop=0.30, hidden=192,
    epochs=26, bs=48, lr=2.5e-3, lr_trunk_mult=0.25, wd=1e-4, ema=0.995,
    w_bin=1.0, w_mu=0.5, w_het=0.15, w_reg=0.6, w_top=0.3, w_rank=0.8,
    tta=4, seed=0, aux=True, photo=0.06,
)


def train_one(imgs, y, aux, tr_idx, va_idx, te_imgs, te_aux, cfg, device="cpu", log=print,
              deadline=None):
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    ytr = y[tr_idx]
    qs, c1, c2, nb = make_bins(ytr, cfg["n_bins"])
    soft = soft_bin_targets(ytr, qs, c1)
    ymax = float(ytr.max())
    yt = (np.sqrt(np.maximum(ytr, 0)) / math.sqrt(ymax + 1e-9)).astype(np.float32)
    thr = np.quantile(ytr, 0.75)
    top = (ytr > thr).astype(np.float32)

    n_aux = aux.shape[1] if (aux is not None and cfg["aux"]) else 0
    model = ContentionNet(cfg["backbone"], cfg["pretrained"], nb, n_aux,
                          cfg["drop"], cfg["hidden"]).to(device)
    trunk_p = list(model.trunk.parameters())
    trunk_ids = {id(p) for p in trunk_p}
    head_p = [p for p in model.parameters() if id(p) not in trunk_ids]
    opt = torch.optim.AdamW(
        [{"params": trunk_p, "lr": cfg["lr"] * cfg["lr_trunk_mult"]},
         {"params": head_p, "lr": cfg["lr"]}], weight_decay=cfg["wd"])

    ntr = len(tr_idx)
    steps_per_ep = max(1, ntr // cfg["bs"])
    total_steps = steps_per_ep * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg["lr"] * cfg["lr_trunk_mult"], cfg["lr"]],
        total_steps=total_steps, pct_start=0.25)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    Xtr_u8 = imgs[tr_idx]
    c1_t = torch.from_numpy(c1.astype(np.float32)).to(device)
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(ntr)
        for b in range(steps_per_ep):
            sel = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            if len(sel) < 4:
                continue
            xb = augment(Xtr_u8[sel], rng, photo=cfg["photo"])
            x = to_tensor(xb).to(device)
            a = None
            if n_aux:
                a = torch.from_numpy(aux[tr_idx][sel]).float().to(device)
            tgt = dict(
                soft=torch.from_numpy(soft[sel]).to(device),
                yt=torch.from_numpy(yt[sel]).to(device),
                y=torch.from_numpy(ytr[sel].astype(np.float32)).to(device),
                top=torch.from_numpy(top[sel]).to(device),
                c1=c1_t)
            out = model(x, a)
            loss, parts = compute_loss(out, tgt, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            d = cfg["ema"] if ep > cfg["epochs"] * 0.3 else 0.9
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v.float(), alpha=1 - d)
                    else:
                        ema[k] = v.float()
        if deadline is not None and time.time() > deadline:
            log(f"    [deadline] stopped after epoch {ep+1}/{cfg['epochs']}")
            break
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()})

    va = readouts(model, imgs[va_idx], None if not n_aux else aux[va_idx], c1, c2,
                  tta=cfg["tta"], device=device, seed=cfg["seed"])
    te = readouts(model, te_imgs, None if not n_aux else te_aux, c1, c2,
                  tta=cfg["tta"], device=device, seed=cfg["seed"])
    return va, te


def run_cv(imgs, y, aux, groups, te_imgs, te_aux, cfg, n_splits=5, device="cpu",
           log=print, deadline=None):
    gkf = GroupKFold(n_splits=n_splits)
    oof = {k: np.zeros(len(y)) for k in READOUT_KEYS}
    te_acc = {k: np.zeros(len(te_imgs)) for k in READOUT_KEYS}
    nf = 0
    for f, (tr_i, va_i) in enumerate(gkf.split(np.arange(len(y)), y, groups)):
        t0 = time.time()
        c = dict(cfg); c["seed"] = cfg["seed"] * 100 + f
        va, te = train_one(imgs, y, aux, tr_i, va_i, te_imgs, te_aux, c, device, log, deadline)
        for k in READOUT_KEYS:
            oof[k][va_i] = va[k]
            te_acc[k] += te[k]
        nf += 1
        s = score(va["cnn_exp"], y[va_i], True)
        log(f"  fold {f}: {time.time()-t0:.0f}s  n_va={len(va_i)}  "
            f"cnn_exp score {s[0]:.4f} (C {s[1]:.3f} R {s[2]:.3f})")
        if deadline is not None and time.time() > deadline:
            log(f"  [deadline] stopping after fold {f}")
            break
    for k in READOUT_KEYS:
        te_acc[k] /= max(nf, 1)
    return oof, te_acc, nf
