#!/usr/bin/env python3
"""
C22 — referee round 3 (control MAJOR 2): the +0.3 step's intervention arm leaves the training-log temperature range for most of the first 24 h in
periods 1–2.  Here the same seven operational start states (48 h apart) receive a +0.1 step (modulation 0.4 → 0.5) whose intervention arm stays
in range; the base arm (0.4) is reused from truth_step.json (identical procedure).  Every saved v1 model (c15 + c16, no retraining) is scored
on this step at 3/6/12/24/48 h, alongside its +0.3 fractions at the same horizons and the in-range fraction of each arm.
Stage 1 (emulator):  C7_V1_PERIOD=P1 python c22_small_step.py truth      -> c7_v1/P*/truth_step_small.json
Stage 2 (models):    C7_V1_PERIOD=P1 python c22_small_step.py models     -> c7_v1/P*/c22_small_step.json
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'; HZ = {'3h': 11, '6h': 23, '12h': 47, '24h': 95, '48h': 191}


def small_arm(ts):
    import bop, c7_boptest as C
    DAY, DT, K = C.DAY, C.DT, C.K
    e = bop.Emulator(C.CASE, name=f'ss{ts}'); e.init(ts - 8 * DAY)
    while e.t < ts - 1: e.step(DT)
    traj = []
    for k in range(192):
        e.step(DT, {C.CFG['u']: 0.5}); v = e.get(C.NAMES); traj.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
    e.close(); return dict(ts=ts, interv=traj)


def score(a):
    import torch; torch.set_num_threads(1)
    import c18_reeval as R18
    fam, train, seed = a
    try:
        sim = R18.load_sim(fam, train, seed)
        big = json.load(open(f'{D}/truth_step.json')); small = json.load(open(f'{D}/truth_step_small.json'))
        op = np.load(f'{D}/log_operational.npy'); lo, hi = op[:, 0].min(), op[:, 0].max()
        out = dict(family=fam, train=train, seed=seed, period=P)
        for tag, T in (('big', big), ('small', small)):
            resp, tr, fi = [], [], []
            for t in T:
                pre = np.array(t['pre']); base = np.array(t['base']); interv = np.array(t['interv'])
                wb = np.r_[pre[:, 1:4], base[:, 1:4]]; wi = np.r_[pre[:, 1:4], interv[:, 1:4]]
                resp.append(sim(pre[:, 0], wi) - sim(pre[:, 0], wb)); tr.append(interv[:, 0] - base[:, 0])
                fi.append(np.mean((interv[:96, 0] >= lo) & (interv[:96, 0] <= hi)))
            resp, tr = np.array(resp), np.array(tr)
            out[tag] = dict(frac_inside_interv=float(np.mean(fi)), truth={h: float(tr[:, i].mean()) for h, i in HZ.items()},
                            ratio={h: float(resp[:, i].mean() / tr[:, i].mean()) for h, i in HZ.items()},
                            per_start_24h=[float(v) for v in resp[:, 95]], truth_per_start_24h=[float(v) for v in tr[:, 95]],
                            per_start={h: [float(v) for v in resp[:, i]] for h, i in HZ.items()}, truth_per_start={h: [float(v) for v in tr[:, i]] for h, i in HZ.items()})
        print(f"{P} {fam:12s} {train:18s} s{seed}: +0.3 @6/24h {out['big']['ratio']['6h']:.2f}/{out['big']['ratio']['24h']:.2f} | +0.1 @6/24h {out['small']['ratio']['6h']:.2f}/{out['small']['ratio']['24h']:.2f} (in-range {out['small']['frac_inside_interv']:.2f})", flush=True)
        return out
    except Exception as e:
        print(f'FAILED {fam} {train} s{seed}: {e}', flush=True); return None


if __name__ == '__main__':
    if sys.argv[1] == 'truth':
        big = json.load(open(f'{D}/truth_step.json'))
        with Pool(7) as p: arms = p.map(small_arm, [t['ts'] for t in big])
        out = []
        for t, a in zip(big, arms):
            assert a['ts'] == t['ts']; interv = np.array(a['interv']); base = np.array(t['base'])
            out.append(dict(ts=t['ts'], pre=t['pre'], base=t['base'], interv=a['interv'], effect=(interv[:, 0] - base[:, 0]).tolist()))
        json.dump(out, open(f'{D}/truth_step_small.json', 'w')); E = np.array([t['effect'] for t in out])
        print(f'{P} +0.1 step truth (7 starts) @6/24/48 h: {E.mean(0)[[23, 95, 191]].round(2)} ± {E.std(0)[[23, 95, 191]].round(2)}', flush=True)
    else:
        rows = json.load(open(f'{D}/c15_results.json')); jobs = [(r['family'], r['train'], r['seed']) for r in rows]
        if os.path.exists(f'{D}/c16_results.json'): jobs += [(r['family'], r['train'], r['seed']) for r in json.load(open(f'{D}/c16_results.json'))]
        with Pool(int(os.environ.get('WORKERS', 12))) as p: res = [r for r in p.map(score, jobs) if r]
        json.dump(res, open(f'{D}/c22_small_step.json', 'w')); print(f'{P}: {len(res)} models scored on the +0.1 step')
