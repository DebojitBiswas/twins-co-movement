#!/usr/bin/env python3
"""
C7b — (P5) instrumented twins and (P3) planning on the twin, BOPTEST bestest_hydronic_heat_pump.

IV: u is replaced by its projection on a filter bank of the logged DITHER component d = r − schedule (two-stage; the schedule is synchronous
with hidden occupancy and is not used as an instrument).  Same GRU / Transformer recipes as c7_neural.
Planning: from 10 start times in the test weeks, each twin (warm-up = true 24-h history, weather given) optimises a 48-h modulation
schedule u ∈ [0, 1] to track a target zone temperature (21.0 °C for 12 h, then 22.5 °C); the schedule is applied OPEN-LOOP to the emulator
(replayed from the same history) and scored by tracking RMSE.  Reference: the embedded controller given the same target as its set-point.
"""
import os, sys, json, numpy as np
DATA = {'bestest_hydronic_heat_pump': 'c7_data', 'singlezone_commercial_hydronic': 'c7_data_commercial'}[os.environ.get('C7_CASE', 'bestest_hydronic_heat_pump')]
from multiprocessing import Pool
import c7_neural as N
import c7_linear as L

PH = 192                                         # 48 h plan
TARGET = np.r_[np.full(48, 21.0), np.full(PH - 48, 22.5)]


def project_dither(name):
    Lg = np.load(f'{DATA}/log_{name}.npy'); sched = np.load(f'{DATA}/log_operational_A0.0_s0.npy')[:, 4]
    d = Lg[:, 4] - sched; Z = np.column_stack([L.bank(d), np.ones(len(d))])
    return Z @ np.linalg.lstsq(Z, Lg[:, 1], rcond=None)[0]


def train(kind, name, seed, iv=False):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Lg = np.load(f'{DATA}/log_{name}.npy'); y, u, x = Lg[:, 0], (project_dither(name) if iv else Lg[:, 1]), Lg[:, 2:4]
    uu = Lg[:, 1]                                                         # normalise with the REAL input's statistics (for planning inputs)
    my, sy, mu, su = y.mean(), y.std() + 1e-6, uu.mean(), uu.std() + 1e-6; mx, sx = x.mean(0), x.std(0) + 1e-6
    T = lambda a_: torch.tensor(a_, dtype=torch.float32)
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
    return m, (my, sy, mu, su, mx, sx)


def plan(m, nz, hist, wx):
    import torch
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yy = T((hist[:, 0] - my) / sy)[None]; uh = T((hist[:, 1] - mu) / su)[None]
    xx = T((np.r_[hist[:, 2:4], wx] - mx) / sx)[None]
    a = torch.zeros(1, PH, requires_grad=True); o = torch.optim.Adam([a], 0.05)
    tgt = T((TARGET - my) / sy)[None]
    for it in range(120):
        u = torch.sigmoid(a); useq = torch.cat([uh, (u - mu) / su], 1)
        yp = m.roll(yy, useq, xx, N.WARM)[:, N.WARM - 1:N.WARM - 1 + PH]
        lo = ((yp - tgt) ** 2).mean(); o.zero_grad(); lo.backward(); o.step()
    return torch.sigmoid(a).detach().numpy()[0], float(np.sqrt(lo.item()) * sy)


def apply_plan(ts, sched_u=None, setpoint=None):
    """replay embedded control to ts, then apply the plan open loop (or the embedded controller with the target as set-point)."""
    import bop, c7_boptest as C
    e = bop.Emulator(C.CASE, name=f'p{ts}{np.random.randint(1e9)}'); e.init(C.TRAIN0 - 7 * C.DAY); hist = []
    while e.t < ts - 1:
        e.step(C.DT)
        if e.t > ts - C.DAY - 1: v = e.get(C.NAMES); hist.append([v[0] - C.K, v[1], v[2] - C.K, v[3], v[4] - C.K])
    ys, wx = [], []
    for k in range(PH):
        ov = {C.CFG['u']: float(sched_u[k])} if sched_u is not None else ({C.CFG['r']: float(setpoint[k]) + C.K} if setpoint is not None else None)
        e.step(C.DT, ov); v = e.get(C.NAMES); ys.append(v[0] - C.K); wx.append([v[2] - C.K, v[3]])
    e.close(); return np.array(hist), np.array(ys), np.array(wx)


def job(a):
    kind, name, seed, iv, starts, ctx = a
    m, nz = train(kind, name, seed, iv)
    out = []
    for ts, hist, wx in ctx:
        u, believed = plan(m, nz, hist, wx)
        _, ytrue, _ = apply_plan(ts, sched_u=u)
        out.append(dict(ts=ts, track=float(np.sqrt(np.mean((ytrue - TARGET) ** 2))), believed=believed, energy=float(u.mean())))
    return dict(kind=kind, train=name, seed=seed, iv=iv, plans=out)


if __name__ == '__main__':
    import c7_boptest as C
    starts = [C.TEST0 + i * (C.TEST_DAYS * C.DAY / 10) + 6 * 3600 for i in range(10)]
    with Pool(10) as p: ctx_raw = p.starmap(apply_plan, [(ts, None, TARGET) for ts in starts])      # embedded controller with the target
    ctx = [(ts, h, wx) for ts, (h, y, wx) in zip(starts, ctx_raw)]
    emb = [float(np.sqrt(np.mean((y - TARGET) ** 2))) for (h, y, wx) in ctx_raw]
    print('embedded controller (target as set-point): tracking RMSE %.2f ± %.2f K' % (np.mean(emb), np.std(emb)), flush=True)
    jobs = []
    for kind in ('gru_sim', 'tf_1s'):
        for s in (0, 1, 2):
            jobs += [(kind, 'designed_A0.0_s0', s, False, starts, ctx), (kind, 'operational_A0.0_s0', s, False, starts, ctx)]
            jobs += [(kind, f'dither_A{A}_s0', s, iv, starts, ctx) for A in (0.5, 1.0, 2.0) for iv in (False, True)]
    with Pool(int(os.environ.get('WORKERS', 18))) as p: rows = p.map(job, jobs)
    json.dump(dict(embedded=emb, rows=rows), open(f'{DATA}/c7b_results.json', 'w'))
    for r in rows:
        tr = [q['track'] for q in r['plans']]; be = [q['believed'] for q in r['plans']]
        print(f"{r['kind']:8s} {r['train']:20s} {'IV ' if r['iv'] else '   '} s{r['seed']}: plan→emulator {np.mean(tr):.2f} K (twin believes {np.mean(be):.2f})", flush=True)
