#!/usr/bin/env python3
"""
C9 — T6 test: upstream (set-point) vs downstream (actuator) what-ifs, BOPTEST (case from C7_CASE).

Twins trained on the OPERATIONAL log (28 days):
  CL  closed-loop twin  (r, x) -> y     [set-point schedule and weather as inputs; no actuator]
  AC  actuator twin     (u, x) -> y     [the c7 twin]
  CT  controller twin   (r, y_hist, x) -> u   [learned embedded controller; chained with AC for query A]
Queries in the test weeks (days 42–56), 10 start times, 48 h, warm-up 24 h of true history:
  A (upstream):   schedule with occupied set-point +1 K and the morning rise 2 h earlier; truth = emulator with embedded controller
                  under the new schedule.
  B (downstream): demand-response override — actuator forced to level U_DR for 4 h from 17:00 each day, embedded controller otherwise;
                  truth = emulator.
Scores: do-RMSE of zone temperature over the 48 h vs truth; persistence (hold T at start) as reference.  Prediction: CL on A < persistence
and comparable to designed-data AC on B-type queries (c7); AC on B worse than persistence; AC+CT chain reproduces CL on A (consistency)
but not B.
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import bop, c7_boptest as C, c7_neural as N

DATA = C.DATA; K = C.K; DT = C.DT; DAY = C.DAY; PH = 192
U_DR = 0.0; DR_LO, DR_HI = 5.0, 11.0          # v2: actuator OFF during the morning rise (v1: 0.2 from 17:00–21:00 had a 0.13-K effect, too weak to discriminate)


def new_schedule(r):
    """query A: occupied set-point +1 K, occupied period starts 2 h earlier (r in °C, 15-min steps)."""
    occ = r > r.min() + 0.1; r2 = r.copy(); r2[occ] += 1.0
    start = np.where(np.diff(occ.astype(int)) == 1)[0] + 1
    for s in start: r2[max(0, s - 8):s] = r2[min(s, len(r) - 1)]
    return r2


def emulate(ts, mode):
    """replay embedded control to ts; then 48 h under: 'base' embedded; 'A' new schedule via r override; 'B' DR override. Returns hist (24 h) and traj."""
    e = bop.Emulator(C.CASE, name=f'q{mode}{ts}'); e.init(C.TRAIN0 - 7 * DAY); hist = []
    while e.t < ts - 1:
        e.step(DT)
        if e.t > ts - DAY - 1: v = e.get(C.NAMES); hist.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
    hist = np.array(hist); traj = []
    # the embedded schedule over the horizon: read it from a base replay (r is reported as the tracked set-point)
    for k in range(PH):
        t_of_day = ((ts + k * DT) % DAY) / 3600.0
        ov = None
        if mode == 'A': ov = {C.CFG['r']: float(SCHED_A[k]) + K}
        if mode == 'B' and DR_LO <= t_of_day < DR_HI: ov = {C.CFG['u']: U_DR}
        e.step(DT, ov); v = e.get(C.NAMES); traj.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
    e.close(); return hist, np.array(traj)


SCHED_A = None


def _truth(ts):
    global SCHED_A
    hb, base = emulate(ts, 'base'); SCHED_A = new_schedule(base[:, 4])
    _, A = emulate(ts, 'A'); _, B = emulate(ts, 'B')
    return dict(ts=ts, hist=hb.tolist(), base=base.tolist(), A=A.tolist(), B=B.tolist(), schedA=SCHED_A.tolist())


def train(kind, inputs, L, seed):
    """inputs: columns of L used as exogenous inputs; target y = col 0 (CL/AC) or u = col 1 (CT, one-step given r, y history, x)."""
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y = L[:, 0] if kind != 'CT' else L[:, 1]
    X = L[:, inputs]; u = L[:, 1] if kind != 'CT' else L[:, 4]              # 'u' slot: AC uses the actuator; CL uses r; CT uses r
    if kind == 'CL': u = L[:, 4]
    if kind == 'CT': u = L[:, 0]                                            # CT: predicts the actuator from (y, r, x): y in the u-slot, r in X
    nz = (y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, X.mean(0), X.std(0) + 1e-6)
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((X - mx) / sx)
    m = N.build('gru_sim', X.shape[1]); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    for it in range(2000):
        Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
        yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
        pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    return m, nz


def roll(m, nz, yhist, useq, Xseq):
    import torch
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    with torch.no_grad():
        return m.roll(T((yhist - my) / sy)[None], T((useq - mu) / su)[None], T((Xseq - mx) / sx)[None], N.WARM)[0, N.WARM - 1:N.WARM - 1 + PH].numpy() * sy + my


def chain(AC, nzAC, CT, nzCT, h, rs, Xf, dr_mask=None):
    """closed-loop simulation of the twins: u_{t+1} = CT(u_<=t, y_<=t, r_<=t, x_<=t); y_{t+1} = AC(y_<=t, u_<=t, x_<=t); optional DR override on u."""
    ys = list(h[:, 0]); us = list(h[:, 1]); W = N.WARM
    for k in range(PH):
        t = len(ys)
        Xct = np.column_stack([rs[t - W:t], Xf[t - W:t]])
        u_next = float(np.clip(roll_last(CT, nzCT, np.array(us[-W:]), np.array(ys[-W:]), Xct), 0, 1))
        y_next = roll_last(AC, nzAC, np.array(ys[-W:]), np.array(us[-W:]), Xf[t - W:t])
        if dr_mask is not None and dr_mask[k]: u_next = U_DR
        us.append(u_next); ys.append(float(y_next))
    return np.array(ys[-PH:])


def job(a):
    seed, truths = a
    L = np.load(f'{DATA}/log_operational_A0.0_s0.npy')
    CL, nzCL = train('CL', [2, 3], L, seed); AC, nzAC = train('AC', [2, 3], L, seed); CT, nzCT = train('CT', [4, 2, 3], L, seed)
    res = dict(seed=seed, A={'CL': [], 'AC+CT': [], 'persist': []}, B={'AC': [], 'CL': [], 'AC+CT': [], 'persist': []},
               effA={'CL': [], 'AC+CT': []}, effB={'AC': [], 'CL': [], 'AC+CT': []}, effA_rms=[], effB_rms=[])
    rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
    for t in truths:
        h, base, A, B = (np.array(t[k]) for k in ('hist', 'base', 'A', 'B')); sA = np.array(t['schedA']); yh = h[:, 0]
        X0 = np.r_[h[:, 2:4], base[:, 2:4]]; r0 = np.r_[h[:, 4], base[:, 4]]; u0 = np.r_[h[:, 1], base[:, 1]]
        dr = np.array([DR_LO <= ((t['ts'] + k * DT) % DAY) / 3600.0 < DR_HI for k in range(PH)])
        # base predictions (embedded operation) by each twin
        bCL = roll(CL, nzCL, yh, r0, X0); bAC = roll(AC, nzAC, yh, u0, X0); bCH = chain(AC, nzAC, CT, nzCT, h, r0, X0)
        # query A (schedule): CL and chain
        XA = np.r_[h[:, 2:4], A[:, 2:4]]; rA = np.r_[h[:, 4], sA]
        pCL = roll(CL, nzCL, yh, rA, XA); pCH = chain(AC, nzAC, CT, nzCT, h, rA, XA)
        res['A']['CL'].append(rm(pCL, A[:, 0])); res['A']['AC+CT'].append(rm(pCH, A[:, 0])); res['A']['persist'].append(rm(yh[-1], A[:, 0]))
        tA = A[:, 0] - base[:, 0]; res['effA_rms'].append(float(np.sqrt(np.mean(tA ** 2))))
        res['effA']['CL'].append(rm(pCL - bCL, tA)); res['effA']['AC+CT'].append(rm(pCH - bCH, tA))
        # query B (actuator off 05–11 h): AC with the applied u; CL with the unchanged schedule (cannot see the override); chain with override
        XB = np.r_[h[:, 2:4], B[:, 2:4]]; rB = np.r_[h[:, 4], B[:, 4]]; uB = np.r_[h[:, 1], B[:, 1]]
        qAC = roll(AC, nzAC, yh, uB, XB); qCL = roll(CL, nzCL, yh, rB, XB); qCH = chain(AC, nzAC, CT, nzCT, h, rB, XB, dr)
        res['B']['AC'].append(rm(qAC, B[:, 0])); res['B']['CL'].append(rm(qCL, B[:, 0])); res['B']['AC+CT'].append(rm(qCH, B[:, 0])); res['B']['persist'].append(rm(yh[-1], B[:, 0]))
        tB = B[:, 0] - base[:, 0]; res['effB_rms'].append(float(np.sqrt(np.mean(tB ** 2))))
        res['effB']['AC'].append(rm(qAC - bAC, tB)); res['effB']['CL'].append(rm(qCL - bCL, tB)); res['effB']['AC+CT'].append(rm(qCH - bCH, tB))
    return res


def roll_last(m, nz, yseg, useg, Xseg):
    """one-step prediction at the end of a fully observed segment (teacher forcing over the segment)."""
    import torch
    my, sy, mu, su, mx, sx = nz; T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    with torch.no_grad():
        out = m.roll(T((yseg - my) / sy)[None], T((useg - mu) / su)[None], T((Xseg - mx) / sx)[None], len(yseg))
    return out[0, -1].item() * sy + my


if __name__ == '__main__':
    starts = [C.TEST0 + i * (C.TEST_DAYS * DAY / 10) + 6 * 3600 for i in range(10)]
    with Pool(10) as p: truths = p.map(_truth, starts)
    json.dump(truths, open(f'{DATA}/c9_truth.json', 'w'))
    dA = np.mean([np.sqrt(np.mean((np.array(t['A'])[:, 0] - np.array(t['base'])[:, 0]) ** 2)) for t in truths])
    dB = np.mean([np.sqrt(np.mean((np.array(t['B'])[:, 0] - np.array(t['base'])[:, 0]) ** 2)) for t in truths])
    print(f'query effects vs base: A (schedule) {dA:.2f} K rms, B (DR override) {dB:.2f} K rms', flush=True)
    with Pool(3) as p: rows = p.map(job, [(s, truths) for s in (0, 1, 2)])
    json.dump(rows, open(f'{DATA}/c9_updown.json', 'w'), indent=1)
    for q in ('A', 'B'):
        eff = np.mean([np.mean(r[f'eff{q}_rms']) for r in rows])
        for mdl in rows[0][q]:
            v = [np.mean(r[q][mdl]) for r in rows]; e = [np.mean(r[f'eff{q}'][mdl]) for r in rows] if mdl in rows[0][f'eff{q}'] else None
            print(f'query {q} {mdl:7s}: do-RMSE {np.mean(v):.2f} ± {np.std(v):.2f} K' + (f'   effect-RMSE {np.mean(e):.2f} ± {np.std(e):.2f} K (effect rms {eff:.2f}; relative {np.mean(e)/eff:.2f})' if e else ''), flush=True)
