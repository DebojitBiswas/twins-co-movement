#!/usr/bin/env python3
"""
C7c — referee M7(b): a proper nonlinear instrumental repair — the CONTROL-FUNCTION estimator.  First stage: regress the actuator u on a
filter bank of the logged dither d = r − schedule (exogenous); residual ε = u − û carries the controller's reaction to disturbances.  Second
stage: train the twin on (y, u, x, ε) and set ε = 0 at prediction and planning time (the intervention breaks the controller's reaction).
Compared with the plain twin (same inputs without ε) and the earlier two-stage twin (trained on û).  Heat-pump building, dither logs
A ∈ {0.5, 1, 2} K, GRU and transformer, 3 seeds; scores: do-RMSE on the designed test schedules (ε = 0), learned +0.3 step response,
and 48-h open-loop plans (as c7b).
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import c7_neural as N
import c7_linear as L
import c7b_iv_planning as P

DATA = 'c7_data'


def first_stage(name):
    Lg = np.load(f'{DATA}/log_{name}.npy'); sched = np.load(f'{DATA}/log_operational_A0.0_s0.npy')[:, 4]
    d = Lg[:, 4] - sched; Z = np.column_stack([L.bank(d), np.ones(len(d))])
    uh = Z @ np.linalg.lstsq(Z, Lg[:, 1], rcond=None)[0]
    return Lg, Lg[:, 1] - uh


def job(a):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    kind, name, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Lg, eps = first_stage(name); y, u = Lg[:, 0], Lg[:, 1]; x = np.column_stack([Lg[:, 2:4], eps])
    my, sy, mu, su = y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6; mx, sx = x.mean(0), x.std(0) + 1e-6
    T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = N.build(kind, 3); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
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
    # evaluation with ε = 0 (its normalised value = −mx/sx for the third column)
    res = dict(kind=kind, train=name, seed=seed, test={}, step_resp=[], plans=[])
    with torch.no_grad():
        for s in (10, 11):
            te = np.load(f'{DATA}/test_designed_s{s}.npy'); errs, resp = [], []
            for st in range(N.WARM, len(te) - N.ROLL, N.ROLL // 2):
                seg = te[st - N.WARM:st + N.ROLL]; xe = np.column_stack([seg[:, 2:4], np.zeros(len(seg))])
                yy = T((seg[:N.WARM, 0] - my) / sy)[None]; uu = T((seg[:, 1] - mu) / su)[None]; xx = T((xe - mx) / sx)[None]
                pr = m.roll(yy, uu, xx, N.WARM)[0, N.WARM - 1:N.WARM - 1 + N.ROLL].numpy() * sy + my; errs.append(np.mean((pr - seg[N.WARM:, 0]) ** 2))
                u2 = np.stack([np.r_[seg[:N.WARM, 1], np.full(N.ROLL, 0.4)], np.r_[seg[:N.WARM, 1], np.full(N.ROLL, 0.7)]])
                pr2 = m.roll(yy.repeat(2, 1), T((u2 - mu) / su), xx.repeat(2, 1, 1), N.WARM)[:, N.WARM - 1:N.WARM - 1 + N.ROLL].numpy() * sy; resp.append(pr2[1] - pr2[0])
            res['test'][s] = float(np.sqrt(np.mean(errs))); res['step_resp'].append(np.mean(resp, 0).round(3).tolist())
    # planning (c7b protocol) with ε = 0: the twin's x has three columns; P.plan expects nz with mx, sx of the twin's x
    starts, ctx = CTX
    nz3 = (my, sy, mu, su, mx, sx)
    for ts, hist, wx in ctx:
        h3 = np.column_stack([hist[:, :2], hist[:, 2:4], np.zeros(len(hist))])            # (y, u, tout, solar, ε=0) — P.plan uses hist[:, 0], hist[:, 1], hist[:, 2:4]
        hist3 = np.column_stack([hist[:, 0], hist[:, 1], hist[:, 2:4], np.zeros(len(hist))])
        upl, believed = plan3(m, nz3, hist3, np.column_stack([wx, np.zeros(len(wx))]))
        _, ytrue, _ = P.apply_plan(ts, sched_u=upl)
        res['plans'].append(dict(ts=ts, track=float(np.sqrt(np.mean((ytrue - P.TARGET) ** 2))), believed=believed, energy=float(upl.mean())))
    return res


def plan3(m, nz, hist, wx):
    """as P.plan but with three x columns (last = ε, zero)."""
    import torch
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yy = T((hist[:, 0] - my) / sy)[None]; uh = T((hist[:, 1] - mu) / su)[None]; xx = T((np.r_[hist[:, 2:5], wx] - mx) / sx)[None]
    a = torch.zeros(1, P.PH, requires_grad=True); o = torch.optim.Adam([a], 0.05); tgt = T((P.TARGET - my) / sy)[None]
    for it in range(120):
        u = torch.sigmoid(a); useq = torch.cat([uh, (u - mu) / su], 1)
        yp = m.roll(yy, useq, xx, N.WARM)[:, N.WARM - 1:N.WARM - 1 + P.PH]; lo = ((yp - tgt) ** 2).mean(); o.zero_grad(); lo.backward(); o.step()
    return torch.sigmoid(a).detach().numpy()[0], float(np.sqrt(lo.item()) * sy)


CTX = None
if __name__ == '__main__':
    import c7_boptest as C
    starts = [C.TEST0 + i * (C.TEST_DAYS * C.DAY / 10) + 6 * 3600 for i in range(10)]
    with Pool(10) as p: ctx_raw = p.starmap(P.apply_plan, [(ts, None, P.TARGET) for ts in starts])
    CTX = (starts, [(ts, h, wx) for ts, (h, y, wx) in zip(starts, ctx_raw)])
    jobs = [(k, f'dither_A{A}_s0', s) for k in ('gru_sim', 'tf_1s') for A in (0.5, 1.0, 2.0) for s in (0, 1, 2)]
    with Pool(int(os.environ.get('WORKERS', 4))) as p: rows = p.map(job, jobs)
    json.dump(rows, open(f'{DATA}/c7c_control_function.json', 'w'))
    for r in rows:
        sr = np.mean(r['step_resp'], 0)
        print(f"CF {r['kind']:8s} {r['train']:16s} s{r['seed']}: do-RMSE {r['test'][10]:.2f} / {r['test'][11]:.2f} K  step@24/48h {sr[95]:.2f} / {sr[191]:.2f}  plan→emulator {np.mean([q['track'] for q in r['plans']]):.2f} K (believed {np.mean([q['believed'] for q in r['plans']]):.2f})", flush=True)
