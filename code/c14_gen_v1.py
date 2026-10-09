#!/usr/bin/env python3
"""C14 — generate the v1 building data (protocol_v1_buildings.md): three periods, operational / dither / designed training logs, test material
(T-op, T-in, T-out), and the within-support step truth from non-overlapping operational starts.  Output: c7_v1/P{1,2,3}/*.npy, truth_step.json."""
import os, sys, json, numpy as np
from multiprocessing import Pool
import bop, c7_boptest as C

DAY, DT, K = C.DAY, C.DT, C.K
PERIODS = {'P1': (7, 42), 'P2': (45, 80), 'P3': (300, 335)}           # (train start day, test start day); train 28 d, test 14 d
IN_MULT = [1/3, 2/3, 1.0, 4/3]              # in-support levels = period's operational mean actuator × IN_MULT (P1: 0.1,0.2,0.3,0.4)
OUT_MULT = [1/3, 5/6, 4/3, 11/6, 7/3]          # designed / out-of-range levels (P1: 0.1,0.25,0.4,0.55,0.7)
IN_HOLD = (1, 6)                     # 15 min – 1.5 h holds for the in-support test (tuned on the emulator 2026-10-08, see protocol addendum)


def run(regime, t0, days, A=0.0, seed=0, levels=None, warm_days=7, sched=None, hold_range=(4, 49)):
    rng = np.random.default_rng(seed); e = bop.Emulator(C.CASE, name=f'{regime}{t0}{A}{seed}'); e.init(t0 - warm_days * DAY)
    while e.t < t0 - 1: e.step(DT)
    n = int(days * DAY / DT); out = np.empty((n, 5)); hold = 0; off = 0.0; lvl = 0.0
    for k in range(n):
        ov = None
        if regime == 'dither':
            if hold <= 0: off = rng.uniform(-A, A); hold = rng.integers(4, 25)
            hold -= 1; ov = {C.CFG['r']: sched[k] + K + off}
        elif regime == 'designed':
            if hold <= 0: lvl = rng.choice(levels); hold = rng.integers(hold_range[0], hold_range[1] + 1)
            hold -= 1; ov = {C.CFG['u']: lvl}
        e.step(DT, ov); v = e.get(C.NAMES); out[k] = [v[0] - K, v[1], v[2] - K, v[3], v[4] - K]
    e.close(); return out


def step_truth(ts):
    res = []
    for u in (0.4, 0.7):
        e = bop.Emulator(C.CASE, name=f's{ts}{u}'); e.init(ts - 8 * DAY); pre = []
        while e.t < ts - 1:
            e.step(DT)
            if e.t > ts - DAY - 1: v = e.get(C.NAMES); pre.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
        traj = []
        for k in range(192):
            e.step(DT, {C.CFG['u']: u}); v = e.get(C.NAMES); traj.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
        e.close(); res.append((np.array(pre), np.array(traj)))
    return dict(ts=ts, pre=res[0][0].tolist(), base=res[0][1].tolist(), interv=res[1][1].tolist(), effect=(res[1][1][:, 0] - res[0][1][:, 0]).tolist())


def _job(a):
    name, kw = a; return name, run(**kw)


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None; truth_only = 'truth' in sys.argv
    for P, (d0, dt0) in PERIODS.items():
        if only and P != only: continue
        D = f'c7_v1/{P}'; os.makedirs(D, exist_ok=True); t0, tt = d0 * DAY, dt0 * DAY
        if truth_only:
            starts = [tt + 6 * 3600 + i * 2 * DAY for i in range(7)]
            with Pool(7) as p: tr = p.map(step_truth, starts)
            json.dump(tr, open(f'{D}/truth_step.json', 'w')); E = np.array([t['effect'] for t in tr])
            print(f'{P} step truth v1.1 (7 starts, 48 h apart) @6/24/48 h: {E.mean(0)[[23,95,191]].round(2)} ± {E.std(0)[[23,95,191]].round(2)}; per-start 24 h {E[:,95].round(2).tolist()}', flush=True); continue
        op = run('operational', t0, 28); np.save(f'{D}/log_operational.npy', op); sched = op[:, 4]
        um = op[:, 1].mean(); IN_LEVELS = [um * m for m in IN_MULT]; OUT_LEVELS = [um * m for m in OUT_MULT]; print(f'{P}: operational mean u {um:.3f} → in-levels {np.round(IN_LEVELS, 3).tolist()}', flush=True)
        jobs = [(f'log_dither_A2.0_s{s}', dict(regime='dither', t0=t0, days=28, A=2.0, seed=s, sched=sched)) for s in range(3)]
        if P == 'P1': jobs += [(f'log_dither_A{A}_s0', dict(regime='dither', t0=t0, days=28, A=A, seed=0, sched=sched)) for A in (0.5, 1.0)]
        jobs += [(f'log_designed_s{s}', dict(regime='designed', t0=t0, days=28, seed=s, levels=OUT_LEVELS)) for s in range(2)]
        jobs += [('test_operational', dict(regime='operational', t0=tt, days=14)),
                 ('test_in', dict(regime='designed', t0=tt, days=14, seed=10, levels=IN_LEVELS, hold_range=IN_HOLD)),
                 ('test_out', dict(regime='designed', t0=tt, days=14, seed=11, levels=OUT_LEVELS))]
        with Pool(10) as p:
            for name, out in p.imap_unordered(_job, jobs):
                np.save(f'{D}/{name}.npy', out)
                inside = np.mean((out[:, 0] >= op[:, 0].min()) & (out[:, 0] <= op[:, 0].max()))
                print(f'{P} {name:22s}: y {out[:,0].mean():.2f}±{out[:,0].std():.2f} [{out[:,0].min():.1f},{out[:,0].max():.1f}]  u {out[:,1].mean():.2f}±{out[:,1].std():.2f}  inside operational range {inside:.2f}', flush=True)
            starts = [tt + 6 * 3600 + i * 2 * DAY for i in range(7)]            # seven starts exactly 48 h apart (v1.1: the v1 list interleaved three extra starts at 18/30-h spacing)
            tr = p.map(step_truth, starts)
        json.dump(tr, open(f'{D}/truth_step.json', 'w'))
        E = np.array([t['effect'] for t in tr]); print(f'{P} step truth @6/24/48 h: {E.mean(0)[[23,95,191]].round(2)} ± {E.std(0)[[23,95,191]].round(2)}', flush=True)
