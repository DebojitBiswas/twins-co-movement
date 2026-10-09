"""Hierarchical bootstrap (seeds × start states) of the learned/true ratio from c22 per-start arrays, same statistic as analyze_v1."""
import json, numpy as np
def ratio_ci(rows, tag, h, n=1000):
    rng = np.random.default_rng(sum(map(ord, f"{rows[0]['family']}|{rows[0]['train']}|{rows[0]['period']}|{tag}|{h}")))
    """rows: c22 rows of one family/regime; tag 'big' or 'small'; h horizon key. Returns (mean of per-seed ratios, lo, hi)."""
    S = np.array([r[tag]['per_start'][h] for r in rows]); t = np.array(rows[0][tag]['truth_per_start'][h]); ns, nk = S.shape
    rb = [np.sum(S[i][:, j]) / (ns * np.sum(t[j])) for i, j in ((rng.choice(ns, ns), rng.choice(nk, nk)) for _ in range(n))]
    return float(np.mean(S.mean(1) / t.mean())), float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))
def load(P):
    try:
        C = json.load(open(f'c7_v1/{P}/c22_small_step.json'))
        return C if C and 'per_start' in C[0].get('big', {}) else None
    except FileNotFoundError: return None
