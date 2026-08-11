"""Global metric-greedy blend across every saved run's read-outs."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import score
from blend import greedy_blend, rank_z, apply_blend

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/nakul/sv/out")
ap.add_argument("--data", default="/home/nakul/sv/dataset")
ap.add_argument("--min_gain", type=float, default=0.0015)
ap.add_argument("--rounds", type=int, default=18)
ap.add_argument("--exclude", default="")
ap.add_argument("--sub", default="")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.out, "oof_*.npz")))
ex = set(filter(None, a.exclude.split(",")))
files = [f for f in files if os.path.basename(f)[4:-4] not in ex]
oof, tep, y = {}, {}, None
for f in files:
    tag = os.path.basename(f)[4:-4]
    z = np.load(f, allow_pickle=True)
    y = z["y"] if y is None else y
    for k in z.files:
        if k.startswith("oof_"):
            name = f"{tag}:{k[4:]}"
            tk = "te_" + k[4:]
            if tk in z.files and np.isfinite(z[k]).all() and np.std(z[k]) > 0:
                oof[name] = z[k]; tep[name] = z[tk]
print(f"loaded {len(files)} runs -> {len(oof)} candidate signals", flush=True)

solo = {k: max(score(v, y), score(-v, y)) for k, v in oof.items()}
print("\ntop 25 solo:")
for k in sorted(solo, key=lambda k: -solo[k])[:25]:
    print(f"  {k:34s} {solo[k]:.4f}", flush=True)

w, s, bl = greedy_blend(oof, y, rounds=a.rounds, min_gain=a.min_gain, verbose=False)
sc, C, R, b = score(bl, y, True)
print(f"\n=== global blend: OOF {sc:.4f}  C {C:.4f}  R {R:.4f}  b {b:.4f} ===")
for k, v in sorted(w.items(), key=lambda t: -abs(t[1])):
    print(f"   {v:+.3f}  {k}", flush=True)

json.dump({"weights": w, "oof": sc, "C": C, "R": R}, open(os.path.join(a.out, "blend.json"), "w"),
          indent=1, default=float)

if a.sub:
    te = pd.read_csv(os.path.join(a.data, "test.csv"))
    pr = apply_blend(tep, w)
    pd.DataFrame({"image_id": te.image_id, "priority": pr}).to_csv(a.sub, index=False)
    print(f"wrote {a.sub}", flush=True)
