#!/usr/bin/env python3
"""
C12 — referee M4/M6: do feedback (receding-horizon MPC) and conservatism (MOPO-style ensemble-disagreement penalty) rescue a twin
trained on operational data?  BOPTEST heat-pump building.

Twins: bootstrap deep ensembles of 5 GRU simulators (each member trained on a day-level bootstrap resample of the training log) for the
operational, dither ±2 K and designed logs.  Controller: receding-horizon MPC with a 24-h horizon, re-planned every 2 h from the TRUE
building state (24-h history), objective mean_ensemble (y − target)² + λ · spread (std across members along the plan), λ ∈ {0, 1, 3, 10}
(in K² and K units), 60 Adam steps per re-plan warm-started from the shifted previous plan; the first 2 h of the plan are applied to the
emulator.  Target: 21.0 °C for 12 h then 22.5 °C (as c7b).  Five start times in the test weeks, 48 h each.
Reported: tracking error on the emulator, energy (mean actuator), fraction of applied actuator values outside the training log's
[5, 95] percentile range (support), mean ensemble spread along the applied trajectory.  Reference: the embedded controller with the target.
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import c7_neural as N

DATA = 'c7_data'; WARM = 96; HOR = 96; REPLAN = 8; TOTAL = 192
T_HI = float(os.environ.get('MPC_TARGET_HI', 22.5)); TARGET_FN = lambda k: 21.0 if k < 48 else T_HI


def train_member(a):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    train_name, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    L = np.load(f'{DATA}/log_{train_name}.npy'); y, u, x = L[:, 0], L[:, 1], L[:, 2:4]
    my, sy, mu, su = y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6; mx, sx = x.mean(0), x.std(0) + 1e-6
    T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    ndays = len(y) // 96; days = rng.integers(0, ndays - 2, ndays)             # day-level bootstrap: windows start inside resampled days
    m = N.build('gru_sim', 2); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32; Lw = 48 + 144
    for it in range(2000):
        d = rng.choice(days, B); idx = d * 96 + rng.integers(0, 96, B); idx = np.minimum(idx, len(y) - Lw - 1)
        yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
        pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    return train_name, seed, {k: v.clone() for k, v in m.state_dict().items()}, (my, sy, mu, su, mx, sx), (float(np.percentile(u, 5)), float(np.percentile(u, 95)))


def mpc_run(a):
    import torch, bop, c7_boptest as C
    torch.set_num_threads(1)
    train_name, lam, ts, members, nz, support = a
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    ms = []
    for sd in members:
        m = N.build('gru_sim', 2); m.load_state_dict(sd)
        for p in m.parameters(): p.requires_grad_(False)
        ms.append(m)
    e = bop.Emulator(C.CASE, name=f'mpc{train_name}{lam}{ts}'); e.init(C.TRAIN0 - 7 * C.DAY); hist = []
    while e.t < ts - 1:
        e.step(C.DT)
        if e.t > ts - C.DAY - 1: v = e.get(C.NAMES); hist.append([v[0] - C.K, v[1], v[2] - C.K, v[3], v[4] - C.K])
    hist = np.array(hist)
    # weather over the horizon is taken as known (forecast = truth): read it from a base replay of the same period
    base = []
    eb = bop.Emulator(C.CASE, name=f'base{train_name}{lam}{ts}'); eb.init(C.TRAIN0 - 7 * C.DAY)
    while eb.t < ts - 1: eb.step(C.DT)
    for k in range(TOTAL + HOR):
        eb.step(C.DT); v = eb.get(C.NAMES); base.append([v[2] - C.K, v[3]])
    eb.close(); wx = np.array(base)
    ys, us, spreads = list(hist[:, 0]), list(hist[:, 1]), []
    a = torch.zeros(1, HOR); plan_prev = None
    for k0 in range(0, TOTAL, REPLAN):
        yh = np.array(ys[-WARM:]); uh = np.array(us[-WARM:]); xh = np.r_[hist[:, 2:4], wx[:k0]][-WARM:] if k0 > 0 else hist[:, 2:4]
        xf = wx[k0:k0 + HOR]
        yy = T((yh - my) / sy)[None].repeat(len(ms), 1); xx = T((np.r_[xh, xf] - mx) / sx)[None].repeat(len(ms), 1, 1)
        tgt = T([TARGET_FN(k0 + j) for j in range(HOR)])
        if plan_prev is not None: a = torch.cat([plan_prev[:, REPLAN:], plan_prev[:, -1:].repeat(1, REPLAN)], 1)
        a = a.detach().clone().requires_grad_(True); opt = torch.optim.Adam([a], 0.05)
        for it in range(60):
            u = torch.sigmoid(a); useq = torch.cat([T((uh - mu) / su)[None], (u - mu) / su], 1).repeat(len(ms), 1)
            yp = torch.stack([m.roll(yy[i:i + 1], useq[i:i + 1], xx[i:i + 1], WARM)[0, WARM - 1:WARM - 1 + HOR] for i, m in enumerate(ms)]) * sy + my
            mean, spread = yp.mean(0), yp.std(0)
            loss = ((mean - tgt) ** 2).mean() + lam * spread.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        plan_prev = a.detach(); plan = torch.sigmoid(plan_prev).numpy()[0]; spreads.append(float(spread.detach().mean()))
        for j in range(REPLAN):
            e.step(C.DT, {C.CFG['u']: float(plan[j])}); v = e.get(C.NAMES); ys.append(v[0] - C.K); us.append(float(plan[j]))
    e.close()
    yt = np.array(ys[-TOTAL:]); ut = np.array(us[-TOTAL:]); tgt = np.array([TARGET_FN(k) for k in range(TOTAL)])
    return dict(train=train_name, lam=lam, ts=ts, track=float(np.sqrt(np.mean((yt - tgt) ** 2))), energy=float(ut.mean()),
                out_of_support=float(np.mean((ut < support[0]) | (ut > support[1]))), spread=float(np.mean(spreads)), y=yt.round(3).tolist(), u=ut.round(3).tolist())


def embedded_ref(ts):
    import bop, c7_boptest as C
    e = bop.Emulator(C.CASE, name=f'emb{ts}'); e.init(C.TRAIN0 - 7 * C.DAY)
    while e.t < ts - 1: e.step(C.DT)
    ys, us = [], []
    for k in range(TOTAL):
        e.step(C.DT, {C.CFG['r']: TARGET_FN(k) + C.K}); v = e.get(C.NAMES); ys.append(v[0] - C.K); us.append(v[1])
    e.close(); tgt = np.array([TARGET_FN(k) for k in range(TOTAL)])
    return dict(ts=ts, track=float(np.sqrt(np.mean((np.array(ys) - tgt) ** 2))), energy=float(np.mean(us)))


if __name__ == '__main__':
    import c7_boptest as C
    trains = ['operational_A0.0_s0', 'dither_A2.0_s0', 'designed_A0.0_s0']
    with Pool(int(os.environ.get('WORKERS', 6))) as p:
        mem = p.map(train_member, [(t, s) for t in trains for s in range(5)])
    ens = {t: [m[2] for m in mem if m[0] == t] for t in trains}; nz = {m[0]: m[3] for m in mem}; sup = {m[0]: m[4] for m in mem}
    starts = [C.TEST0 + i * (C.TEST_DAYS * C.DAY / 5) + 6 * 3600 for i in range(5)]
    jobs = [(t, lam, ts, ens[t], nz[t], sup[t]) for t in trains for lam in (0.0, 1.0, 3.0, 10.0) for ts in starts]
    with Pool(int(os.environ.get('WORKERS', 6))) as p:
        emb = p.map(embedded_ref, starts); rows = p.map(mpc_run, jobs)
    json.dump(dict(embedded=emb, rows=rows, target_hi=T_HI), open(f'{DATA}/c12_mpc' + ('' if T_HI == 22.5 else f'_t{T_HI}') + '.json', 'w'))
    print(f"embedded controller (target as set-point, closed loop): {np.mean([r['track'] for r in emb]):.2f} K, energy {np.mean([r['energy'] for r in emb]):.2f}")
    for t in trains:
        for lam in (0.0, 1.0, 3.0, 10.0):
            R = [r for r in rows if r['train'] == t and r['lam'] == lam]
            print(f"MPC {t:20s} λ={lam:4.1f}: tracking {np.mean([r['track'] for r in R]):.2f} ± {np.std([r['track'] for r in R]):.2f} K  energy {np.mean([r['energy'] for r in R]):.2f}  "
                  f"out-of-support {np.mean([r['out_of_support'] for r in R]):.2f}  spread {np.mean([r['spread'] for r in R]):.2f} K", flush=True)
