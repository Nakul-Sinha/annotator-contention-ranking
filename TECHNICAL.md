# Annotator Disagreement Ranking: technical notes

Rank ~173 held-out mouse-face crops by how far an expert panel's independent 0/1/2 marks
scatter. Only the **ordering** of `priority` is scored.

## Metric (replicated in `code/metric.py`, inlined in `solution.py`)

    C = magnitude-weighted contention-capture area   (random = 0, oracle = 1)
    R = Spearman(priority, contention)
    b = clip(C,0,1)^0.6 * clip(R,0,1)^0.4
    score = mean of six endpoint-matched response curves g_k(b)

Anchors reproduce exactly: oracle → 1.0000, random → 0.000, reverse-order → 0 (clipped),
constant → 0. The response bank is close to the identity in the region of interest
(b = 0.30 → 0.320, b = 0.50 → 0.519, b = 0.685 → 0.689), so gains in `b` pass through
almost one-for-one.

## Data profile

| quantity | value |
|---|---|
| train faces | 982 (≈141 animals) |
| test faces | 173 (one per animal, animals disjoint from train) |
| images | 1155 × 112×112 RGB JPEG |
| contention | mean 0.1270, sd 0.0886, max 0.5333, 8.8 % exact zeros, skew +1.18 |

The label is a **small-sample statistic**: denominators of the exact values divide 5·k² for
k ≈ 3 to 5 assessors, so realized scatter carries roughly 35 % relative sampling noise. That
dictates the estimator design, a shrinking L1/L2 head is the wrong tool for a spread target.

## Validation design

No subject column is provided and the ids are verified random (602 distinct `fc-<n>` prefixes
across 982 train rows, 109 of them shared with test, the prefix is **not** an animal id).
Subject groups are recovered by Ward clustering of generic ResNet-18 embeddings of the
training images (k = 141):

| recipe | median size | max | singletons | target ICC |
|---|---|---|---|---|
| resnet18-emb + PCA64 + **ward** | 6 | 21 | 1 | **+0.146** |
| resnet18-emb + complete linkage | 4 | 61 | 21 | +0.207 (degenerate sizes) |
| resnet18-emb + kmeans | 6 | 22 | 4 | +0.138 |
| average linkage | 2 | 171 | 57 | −0.005 (collapses) |

Groups are used only to build `GroupKFold` folds; they are never a model input.

### Leakage audit (`code/leak_audit.py`)

Test animals are disjoint from train, so any signal that works by finding a *same-animal*
neighbour would not transfer. Re-scoring under progressively harsher separation:

| regime | knn16 | knn48 | ridge | gbm_rk |
|---|---|---|---|---|
| ward141 (training folds) | 0.3412 | 0.3067 | 0.2429 | 0.2243 |
| ward70 (over-merged) | 0.3416 | 0.2968 | 0.2601 | 0.2867 |
| ward35 (extreme) | 0.3261 | 0.2832 | 0.2122 | 0.2016 |
| ward141 + cos>0.90 firewall | 0.3267 | 0.2973 | 0.2414 | 0.2540 |
| ward141 + cos>0.85 firewall | 0.2997 | 0.2617 | 0.2102 | 0.1958 |

The neighbour signal survives, so it is appearance-class information (coat, lighting, cage,
framing), not animal identity. Test faces have comparable neighbour geometry to train faces
(max-cos p50 0.873 vs 0.887; 70 % of test faces have a train neighbour above cos 0.85).

## Approach

Many weak, decorrelated ranking signals, greedy-blended on subject-grouped OOF **by the actual
competition metric**, never one "best" estimator.

* **Convolutional core** (fine-tuned ImageNet backbone, one trunk, five heads)
  * soft-binned categorical head over quantile bins of contention → `E[y]`, `E[y²]`,
    law-of-total-variance spread, entropy, upper-tail mass, predicted quantiles, CVaR
  * scalar regression head and a heteroscedastic scale head
  * a structured **five-region rubric head**: per region a distribution over marks {0,1,2},
    whose implied variance is averaged and supervised against the label, this supervises the
    quantity the target is built from, and its inductive bias matches the fact that scatter
    peaks for intermediate/ambiguous grades
  * a top-quartile classifier, because facet C weights the head of the queue heavily
  * a gap-weighted pairwise ranking loss aligned to the scored ordering
* **Classical legibility descriptors** (42): sharpness, Laplacian energy, FFT band ratios,
  gradient anisotropy (directional/motion blur), exposure and clipping, colour, framing and
  energy centroid, mirror asymmetry (pose), flat-block occlusion proxies. Fed both as extra
  model inputs and as standalone blend candidates.
* **Small non-convolutional regressors** on those descriptors and on generic embeddings,
  fitted on the same folds, weak blend members by design, never load-bearing alone.

Augmentation is geometry-only (flip, ±12° rotation, ±8 % scale, ±5 % translate) plus mild
brightness/contrast. Blur, rescale and noise are deliberately excluded: they destroy the
legibility cues that drive panel disagreement.

## Layout

    solution.py            self-contained deliverable (argv + no-arg fallback, dual-write)
    code/metric.py         scorer replica
    code/feats.py          classical descriptors
    code/train_lib.py      model, losses, read-outs
    code/runner.py         grouped-OOF fold runner
    code/blend.py          metric-greedy blending
    code/make_groups.py    subject clustering
    code/leak_audit.py     transfer/leakage audit
    code/blend_cv.py       nested validation of the blending procedure
    code/epochprobe.py     epoch-response measurement
