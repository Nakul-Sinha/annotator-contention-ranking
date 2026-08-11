"""Classical image-legibility / indeterminacy descriptors for the rodent-face crops.

These are cheap, hand-engineered proxies for the visual drivers of panel disagreement:
sharpness, motion blur, exposure, framing, occlusion and pose asymmetry. They are used
(a) as candidate ranking signals in the OOF blend and (b) as auxiliary inputs alongside
the convolutional model.
"""
import numpy as np
from PIL import Image


def _gray(a):
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_features(a):
    """a: HxWx3 float32 in [0,1]. Returns dict of scalar descriptors."""
    f = {}
    g = _gray(a)
    H, W = g.shape

    # --- sharpness / blur family -------------------------------------------------
    lap = (-4 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    f["lap_var"] = float(np.var(lap))
    f["lap_abs"] = float(np.mean(np.abs(lap)))
    f["lap_p99"] = float(np.percentile(np.abs(lap), 99))

    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    ten = gx[:-1, :] ** 2 + gy[:, :-1] ** 2
    f["tenengrad"] = float(np.mean(ten))
    f["grad_p90"] = float(np.percentile(np.sqrt(ten), 90))
    f["edge_dens"] = float(np.mean(np.sqrt(ten) > 0.06))

    # frequency-domain sharpness: high-band energy fraction
    F = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    tot = F.sum() + 1e-8
    f["fft_hi"] = float(F[r > 0.30 * H].sum() / tot)
    f["fft_mid"] = float(F[(r > 0.12 * H) & (r <= 0.30 * H)].sum() / tot)
    # anisotropy of the gradient field = directional (motion) blur proxy
    sxx = float(np.mean(gx[:-1, :] ** 2))
    syy = float(np.mean(gy[:, :-1] ** 2))
    sxy = float(np.mean(gx[:-1, :] * gy[:, :-1]))
    tr_, det = sxx + syy, sxx * syy - sxy ** 2
    disc = max(tr_ ** 2 / 4 - det, 0.0) ** 0.5
    l1, l2 = tr_ / 2 + disc, max(tr_ / 2 - disc, 1e-12)
    f["grad_aniso"] = float((l1 - l2) / (l1 + l2 + 1e-12))
    f["grad_coher"] = float(np.log(l1 / l2 + 1e-12))

    # --- exposure / contrast -----------------------------------------------------
    f["mean"] = float(g.mean())
    f["std"] = float(g.std())
    f["p01"] = float(np.percentile(g, 1))
    f["p99"] = float(np.percentile(g, 99))
    f["dyn_range"] = f["p99"] - f["p01"]
    f["clip_lo"] = float(np.mean(g < 0.02))
    f["clip_hi"] = float(np.mean(g > 0.98))
    hist = np.histogram(g, bins=32, range=(0, 1))[0].astype(np.float64) + 1e-9
    hist /= hist.sum()
    f["hist_ent"] = float(-(hist * np.log(hist)).sum())
    f["skew"] = float(np.mean(((g - g.mean()) / (g.std() + 1e-8)) ** 3))
    f["kurt"] = float(np.mean(((g - g.mean()) / (g.std() + 1e-8)) ** 4))

    # --- colour ------------------------------------------------------------------
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    f["sat_mean"] = float(sat.mean())
    f["sat_std"] = float(sat.std())
    f["rg"] = float(a[..., 0].mean() - a[..., 1].mean())
    f["by"] = float(a[..., 2].mean() - 0.5 * (a[..., 0].mean() + a[..., 1].mean()))
    f["chan_std"] = float(np.std([a[..., c].mean() for c in range(3)]))

    # --- framing / pose ----------------------------------------------------------
    # centre-vs-border energy: how well the subject fills the crop
    c = g[H // 4:3 * H // 4, W // 4:3 * W // 4]
    f["ctr_std"] = float(c.std())
    f["ctr_mean"] = float(c.mean())
    f["ctr_border_ratio"] = float(c.std() / (g.std() + 1e-8))
    f["ctr_sharp"] = float(np.var(lap[H // 4:3 * H // 4, W // 4:3 * W // 4]))
    f["sharp_ratio"] = float(f["ctr_sharp"] / (f["lap_var"] + 1e-12))

    # left/right mirror asymmetry = head-turn / pose proxy (several scales)
    for name, im in (("g", g),):
        mir = im[:, ::-1]
        f[f"asym_{name}"] = float(np.mean(np.abs(im - mir)))
        f[f"asym_{name}_c"] = float(1.0 - np.corrcoef(im.ravel(), mir.ravel())[0, 1])
    # energy centroid offset from crop centre = off-centre framing
    e = np.sqrt(ten) + 1e-8
    ey, ex = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    cy = float((e * ey).sum() / e.sum()) / e.shape[0] - 0.5
    cx = float((e * ex).sum() / e.sum()) / e.shape[1] - 0.5
    f["cen_off"] = float(np.hypot(cy, cx))
    f["cen_x"] = abs(cx)
    f["cen_y"] = abs(cy)
    # spatial spread of edge energy (tight vs diffuse structure)
    f["e_spread"] = float(np.sqrt(((e * ((ey / e.shape[0] - 0.5) ** 2 +
                                         (ex / e.shape[1] - 0.5) ** 2)).sum()) / e.sum()))

    # --- occlusion / texture homogeneity ----------------------------------------
    # fraction of 8x8 blocks that are near-flat (occluder, blown-out or empty area)
    bs = 8
    blk = g[:H // bs * bs, :W // bs * bs].reshape(H // bs, bs, W // bs, bs)
    bstd = blk.std(axis=(1, 3))
    f["flat_frac"] = float(np.mean(bstd < 0.02))
    f["blk_std_std"] = float(bstd.std())
    f["blk_std_min"] = float(bstd.min())
    f["blk_mean_range"] = float(np.ptp(blk.mean(axis=(1, 3))))

    # local-entropy style measure over 16x16 tiles
    bs = 16
    blk2 = g[:H // bs * bs, :W // bs * bs].reshape(H // bs, bs, W // bs, bs)
    bm = blk2.mean(axis=(1, 3))
    f["tile_mean_std"] = float(bm.std())
    f["tile_sharp_std"] = float(np.log1p(blk2.std(axis=(1, 3))).std())
    return f


FEATURE_NAMES = None


def feature_matrix(paths, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    global FEATURE_NAMES

    def one(p):
        return image_features(load(p))

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, paths))
    FEATURE_NAMES = list(rows[0].keys())
    return np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float64), FEATURE_NAMES
