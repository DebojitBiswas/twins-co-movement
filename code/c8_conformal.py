#!/usr/bin/env python3
"""
C8 — T2 test: uncertainty blindness.  Split-conformal predictive intervals for the rollout error of GRU / Transformer twins
(BOPTEST heat-pump building), calibrated on held-out OPERATIONAL days, evaluated on (a) further operational days and (b) the designed
test schedules.  Also the ensemble-spread/error ratio.  Prediction (theory_c3_new T2): operational coverage ≈ nominal, interventional
coverage ≪ nominal for operational-data twins; designed-data twins keep coverage on (b).

Training log for 'operational' twins: first 21 days of the operational log; calibration: days 22–25; operational test: days 26–28.
Designed twins: trained on designed s0, calibrated on designed s1 days 1–4, operational test as above, designed test as (b).
Interval: per-horizon absolute-residual quantile (1 − α), α = 0.1, over 48-h rollouts with 24-h warm-up; coverage reported per horizon
band (0–6 h, 6–24 h, 24–48 h).
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import c7_neural as N

DATA = 'c7_data'; ALPHA = 0.1; BANDS = {'0-6h': (0, 24), '6-24h': (24, 96), '24-48h': (96, 192)}


def train_on(kind, L, seed):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y, u, x = L[:, 0], L[:, 1], L[:, 2:4]
    nz = (y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, x.mean(0), x.std(0) + 1e-6)
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = N.build(kind, 2); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    for it in range(2000 if kind == 'gru_sim' else 6000):
        if kind == 'gru_sim':
            Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        else:
            Lw = N.WARM; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.step_all(yb[:, :Lw] + 0.1 * torch.randn_like(yb[:, :Lw]), ub, xb); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    return m, nz


def residuals(m, nz, L, stride=48):
    """signed rollout residuals (n_windows, ROLL) on log L: 24-h warm-up, 48-h rollout."""
    import torch
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32); R = []
    with torch.no_grad():
        for st in range(N.WARM, len(L) - N.ROLL, stride):
            seg = L[st - N.WARM:st + N.ROLL]
            pr = m.roll(T((seg[:N.WARM, 0] - my) / sy)[None], T((seg[:, 1] - mu) / su)[None], T((seg[:, 2:4] - mx) / sx)[None], N.WARM)[0, N.WARM - 1:N.WARM - 1 + N.ROLL].numpy() * sy + my
            R.append(pr - seg[N.WARM:, 0])
    return np.array(R)


def job(a):
    kind, regime, seed = a
    op = np.load(f'{DATA}/log_operational_A0.0_s0.npy'); d0 = np.load(f'{DATA}/log_designed_A0.0_s0.npy'); d1 = np.load(f'{DATA}/log_designed_A0.0_s1.npy')
    day = 96
    if regime == 'operational': train, cal = op[:21 * day], op[21 * day - N.WARM:25 * day]
    else: train, cal = d0, d1[:4 * day + N.WARM]
    op_test = op[25 * day - N.WARM:]
    m, nz = train_on(kind, train, seed)
    q = np.quantile(np.abs(residuals(m, nz, cal, stride=24)), 1 - ALPHA, axis=0)           # per-horizon radius
    out = dict(kind=kind, regime=regime, seed=seed, radius=q.tolist(), coverage={}, rmse={})
    tests = {'operational': residuals(m, nz, op_test, stride=24)}
    for s in (10, 11): tests[f'designed_s{s}'] = residuals(m, nz, np.load(f'{DATA}/test_designed_s{s}.npy'), stride=24)
    for name, R in tests.items():
        out['coverage'][name] = {b: float(np.mean(np.abs(R[:, lo:hi]) <= q[lo:hi])) for b, (lo, hi) in BANDS.items()}
        out['rmse'][name] = float(np.sqrt(np.mean(R ** 2)))
    return out


if __name__ == '__main__':
    jobs = [(k, r, s) for k in ('gru_sim', 'tf_1s') for r in ('operational', 'designed') for s in (0, 1, 2)]
    with Pool(int(os.environ.get('WORKERS', 6))) as p: rows = p.map(job, jobs)
    json.dump(rows, open(f'{DATA}/c8_conformal.json', 'w'), indent=1)
    for r in rows:
        print(f"{r['kind']:8s} {r['regime']:12s} s{r['seed']}: nominal {1-ALPHA:.2f} | coverage op-test " +
              ' '.join(f"{b} {r['coverage']['operational'][b]:.2f}" for b in BANDS) + ' | designed test ' +
              ' '.join(f"{b} {np.mean([r['coverage'][f'designed_s{s}'][b] for s in (10, 11)]):.2f}" for b in BANDS) +
              f" | rmse op {r['rmse']['operational']:.2f} des {np.mean([r['rmse'][f'designed_s{s}'] for s in (10, 11)]):.2f}", flush=True)
