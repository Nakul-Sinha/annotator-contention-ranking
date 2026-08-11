"""Group-aware CV support.

The challenge states each training animal contributes several faces while the test set serves
exactly one face per animal, and asks for subject-aware validation -- but no subject column is
provided and the served ids are random. We therefore recover approximate subject groups by
unsupervised clustering of the *provided training images only* (coat colour, cage background and
illumination are near-constant within an animal's session). The clustering never sees the target;
it exists only to build leakage-free CV folds.
"""
import numpy as np
from PIL import Image


def appearance_descriptor(path, s=16):
    a = np.asarray(Image.open(path).convert("RGB").resize((s, s), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    d = [a.reshape(-1)]
    # coarse colour histogram (coat + bedding palette)
    for c in range(3):
        h = np.histogram(a[..., c], bins=16, range=(0, 1))[0].astype(np.float32)
        d.append(h / (h.sum() + 1e-8))
    return np.concatenate(d)


def descriptors(paths, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return np.array(list(ex.map(appearance_descriptor, paths)), dtype=np.float32)


def cluster_subjects(paths, n_clusters=141, workers=8):
    from sklearn.cluster import AgglomerativeClustering
    D = descriptors(paths, workers)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    # average linkage on correlation distance: merges faces sharing coat/background/lighting
    lab = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine",
                                  linkage="average").fit_predict(D)
    return lab, D
