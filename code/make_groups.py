"""Build the subject grouping used by every CV split (Ward on generic image embeddings).

Probe results (982 train faces, target k=141 animals):
  resnet18-emb + PCA64 + ward     med 6  max 21  sing 1   ICC +0.146   <- chosen (balanced)
  resnet18-emb + complete-linkage med 4  max 61  sing 21  ICC +0.207   (degenerate sizes)
  resnet18-emb + kmeans           med 6  max 22  sing 4   ICC +0.138
  pixel+hist   + ward             med 5  max 44  sing 5   ICC +0.130
Average-linkage collapses into one 171-face cluster and is unusable.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import load_images, to_tensor

DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/nakul/sv/dataset"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/nakul/sv/out"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 141
os.makedirs(OUT, exist_ok=True)

tr = pd.read_csv(os.path.join(DATA, "train.csv"))
I = load_images([os.path.join(DATA, "images", i + ".jpg") for i in tr.image_id])

import timm
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
m = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg").eval()
es = []
with torch.no_grad():
    for i in range(0, len(I), 128):
        es.append(m(to_tensor(I[i:i + 128])).numpy())
E = np.concatenate(es)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
P = PCA(n_components=64, random_state=0).fit_transform(E)
lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(P)

y = tr.contention.values
sz = pd.Series(lab).value_counts()
wv = np.mean([np.var(y[lab == c]) for c in np.unique(lab) if (lab == c).sum() > 1])
rg = np.random.default_rng(1).permutation(lab)
wr = np.mean([np.var(y[rg == c]) for c in np.unique(rg) if (rg == c).sum() > 1])
print(f"k={K} med {sz.median():.0f} max {sz.max()} sing {(sz==1).sum()} ICC {1-wv/wr:+.3f}")
np.save(os.path.join(OUT, "groups.npy"), lab)
np.save(os.path.join(OUT, "emb.npy"), E)
print("saved", os.path.join(OUT, "groups.npy"))
