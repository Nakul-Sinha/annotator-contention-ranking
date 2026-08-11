"""Replica of the Split Verdict scorer.

score = mean_k g_k(b),  b = clip(C,0,1)^0.6 * clip(R,0,1)^0.4
C = magnitude-weighted contention-capture area (random=0, oracle=1)
R = Spearman rank correlation (average ranks for ties)
"""
import numpy as np
from scipy.stats import spearmanr


def capture_C(priority, contention):
    p = np.asarray(priority, dtype=np.float64)
    y = np.asarray(contention, dtype=np.float64)
    n = len(y)
    # invalid priorities go to the bottom of the queue
    bad = ~np.isfinite(p)
    if bad.any():
        p = p.copy()
        p[bad] = -np.inf
    # queue position 1..n, most urgent first -> weight w = n - pos + 1
    order = np.argsort(-p, kind="stable")
    w = np.empty(n, dtype=np.float64)
    w[order] = np.arange(n, 0, -1, dtype=np.float64)
    A_you = float(np.sum(y * w))
    A_rand = float(np.sum(y) * (n + 1) / 2.0)
    oracle_order = np.argsort(-y, kind="stable")
    wo = np.empty(n, dtype=np.float64)
    wo[oracle_order] = np.arange(n, 0, -1, dtype=np.float64)
    A_or = float(np.sum(y * wo))
    if A_or == A_rand:
        return 0.0
    return (A_you - A_rand) / (A_or - A_rand)


def curves(b):
    g1 = (np.exp(2.5 * b) - 1.0) / (np.exp(2.5) - 1.0)
    g2 = np.log(1.0 + 4.0 * b) / np.log(5.0)
    g3 = np.sin(np.pi * b / 2.0)
    g4 = (1.0 - np.cos(np.pi * b)) / 2.0
    g5 = np.tan(1.3 * b) / np.tan(1.3)
    g6 = np.tanh(2.0 * b) / np.tanh(2.0)
    return np.mean([g1, g2, g3, g4, g5, g6], axis=0)


def score(priority, contention, detail=False):
    p = np.asarray(priority, dtype=np.float64)
    y = np.asarray(contention, dtype=np.float64)
    C = capture_C(p, y)
    pf = np.where(np.isfinite(p), p, -1e300)
    R = spearmanr(pf, y).statistic
    R = 0.0 if not np.isfinite(R) else float(R)
    b = np.clip(C, 0, 1) ** 0.6 * np.clip(R, 0, 1) ** 0.4
    s = float(np.clip(curves(b), 0, 1))
    if detail:
        return s, C, R, b
    return s


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 0.06, 400)
    print("oracle ", score(y, y, True))
    print("random ", score(rng.normal(size=400), y, True))
    print("reverse", score(-y, y, True))
    print("const  ", score(np.ones(400), y, True))
    noisy = y + rng.normal(0, y.std() * 0.8, 400)
    print("noisy  ", score(noisy, y, True))
