"""Probe several clustering recipes for recovering animal identity in the training faces."""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import load_images, to_tensor

DATA = "/home/nakul/sv/dataset"; OUT = "/home/nakul/sv/out"
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
img = lambda i: os.path.join(DATA, "images", i + ".jpg")
y = tr.contention.values
I = load_images([img(i) for i in tr.image_id])

E = None
p = os.path.join(OUT, "emb.npy")
if os.path.exists(p):
    E = np.load(p)
else:
    import timm
    m = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg").eval()
    es = []
    with torch.no_grad():
        for i in range(0, len(I), 128):
            es.append(m(to_tensor(I[i:i + 128])).numpy())
    E = np.concatenate(es); np.save(p, E)
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

# raw-pixel + colour-histogram descriptor (session/cage/coat cue)
from PIL import Image
def desc(i, s=24):
    a = np.asarray(Image.open(img(i)).convert("RGB").resize((s, s), Image.BILINEAR),
                   dtype=np.float32) / 255.
    h = [np.histogram(a[..., c], bins=24, range=(0, 1))[0] / (s * s) for c in range(3)]
    return np.concatenate([a.reshape(-1), np.concatenate(h)])
P = np.stack([desc(i) for i in tr.image_id])
Pn = (P - P.mean(0)) / (P.std(0) + 1e-6); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-8

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

def report(name, lab):
    sz = pd.Series(lab).value_counts()
    wv = np.mean([np.var(y[lab == c]) for c in np.unique(lab) if (lab == c).sum() > 1])
    rg = np.random.default_rng(1).permutation(lab)
    wr = np.mean([np.var(y[rg == c]) for c in np.unique(rg) if (rg == c).sum() > 1])
    print(f"{name:34s} k={sz.size:4d} med {sz.median():3.0f} max {sz.max():4d} "
          f"sing {(sz==1).sum():3d}  ICC {1-wv/wr:+.3f}", flush=True)

for nm, X in (("resnet18-emb", En), ("pixel+hist", Pn)):
    Xp = PCA(n_components=64, random_state=0).fit_transform(X)
    for link in ("ward", "complete", "average"):
        met = "euclidean" if link == "ward" else "cosine"
        Z = Xp if link == "ward" else X
        report(f"{nm} agg-{link}", AgglomerativeClustering(n_clusters=141, metric=met,
                                                           linkage=link).fit_predict(Z))
    report(f"{nm} kmeans", KMeans(141, n_init=4, random_state=0).fit_predict(Xp))

# combined descriptor
C = np.hstack([En, Pn])
C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
Cp = PCA(n_components=64, random_state=0).fit_transform(C)
report("combo agg-ward", AgglomerativeClustering(n_clusters=141, linkage="ward").fit_predict(Cp))
report("combo kmeans", KMeans(141, n_init=4, random_state=0).fit_predict(Cp))

print("\n--- top-20 most similar train pairs (are they the same animal?) ---")
S = En @ En.T; np.fill_diagonal(S, -9)
iu = np.triu_indices(len(S), 1)
o = np.argsort(-S[iu])[:20]
for k in o[:20]:
    i, j = iu[0][k], iu[1][k]
    print(f"  {tr.image_id[i]} <-> {tr.image_id[j]}  cos {S[i,j]:.4f}  y {y[i]:.3f}/{y[j]:.3f}")
