#!/usr/bin/env python3
"""
C19 — referee (control MAJOR 9): does the simulation-error twin learn the controller's co-movement?  Direct test on the building.
Fit the embedded controller from the operational log, Ĉ: u_t = Σ a_i u_{t-i} + Σ b_j e_{t-j} + c, e = set-point − zone temperature (ARX form,
p = q = 6), then the identity's pure-feedback limit G_op = −1/Ĉ: the output change that would make the controller produce a +0.3 step in u.
Compare its 24-h value with the learned +0.3 steps of every family (c18 results) and with the truth, per period.  Also the linear law's
mixture prediction γ² G + (1 − γ²)(−1/Ĉ) with γ² = coherence of u with the schedule (band 3 h – 2 d) as a rough excitation share.
"""
import json, numpy as np, scipy.signal as ss

def fit_controller(L, p=6, q=6):
    u, y, r = L[:, 1], L[:, 0], L[:, 4]; e = r - y; X, T = [], []
    for t in range(max(p, q), len(u)):
        X.append(np.r_[u[t - p:t][::-1], e[t - q:t + 1][::-1], 1.0]); T.append(u[t])
    X, T = np.array(X), np.array(T); th = np.linalg.lstsq(X, T, rcond=None)[0]
    return th[:p], th[p:p + q + 1], 1 - np.var(T - X @ th) / np.var(T)

def inv_ctrl_step(a, b, du=0.3, n=192):
    u = np.full(n, du); e = np.zeros(n)
    for t in range(n):
        lhs = u[t] - sum(a[i] * (u[t - 1 - i] if t - 1 - i >= 0 else 0.0) for i in range(len(a))) - sum(b[j] * e[t - j] for j in range(1, len(b)) if t - j >= 0)
        e[t] = lhs / b[0]
    return -e                                                                  # y change = −(set-point error change)

if __name__ == '__main__':
    out = {}
    for P in ('P1', 'P2', 'P3'):
        L = np.load(f'c7_v1/{P}/log_operational.npy'); a, b, r2 = fit_controller(L); gop = inv_ctrl_step(a, b)
        tr = json.load(open(f'c7_v1/{P}/truth_step.json')); G = np.mean([t['effect'][95] for t in tr])
        f, C = ss.coherence(L[:, 1], L[:, 4], nperseg=384); g2 = float(C[(f > 1 / 192) & (f < 1 / 12)].mean())
        pred = g2 * G + (1 - g2) * gop[95]
        try:
            res = json.load(open(f'c7_v1/{P}/c18_results.json'))
            learned = {fam: np.mean([np.array(r['step']['learned'])[95] for r in res if r['family'] == fam and r['train'] == 'log_operational']) for fam in ('gru_sim', 'gru_1s_noise', 'tf_1s_noise', 'tf_1s', 'arx48', 'pem2')}
        except FileNotFoundError: learned = {}
        out[P] = dict(r2=float(r2), inv_ctrl_24h=float(gop[95]), truth_24h=float(G), gamma2=g2, law_pred_24h=float(pred), learned_24h={k: float(v) for k, v in learned.items()})
        print(f"{P}: controller fit R² {r2:.3f}; −1/Ĉ 24-h response to +0.3: {gop[95]:+.2f} K; truth {G:.2f}; γ² {g2:.2f}; law {pred:+.2f}; learned " + ', '.join(f'{k} {v:+.2f}' for k, v in learned.items()))
    json.dump(out, open('c7_v1/inverse_controller.json', 'w'), indent=1)
