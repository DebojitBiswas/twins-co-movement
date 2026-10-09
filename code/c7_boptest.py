#!/usr/bin/env python3
"""
C7 — BOPTEST bestest_hydronic_heat_pump (IBPSA emulator, fmpy co-simulation via bop.py).  Design: c7_boptest_design.md.

Logs at 15 min (Δ = 900 s):  y = zone temperature reaTZon_y [°C]; u = applied heat-pump modulation oveHeaPumY_y [0..1];
drivers x = outdoor dry-bulb [°C], global horizontal solar [W/m²]; r = set-point tracked by the embedded controller oveTSet_y [°C] (v0 wrongly used reaTSetHea_y, the KPI comfort band).
Hidden: internal gains / occupancy, slab and wall temperatures.
Regimes (training: days 7–35 of the year, 4 weeks):
  operational   embedded controller and set-back schedule
  dither_A      set-point = schedule + random offset U(−A, A) K held 1–6 h (oveTSet), A ∈ {0.5, 1, 2}
  designed      heat-pump modulation overridden by random levels {0, .25, .5, .75, 1} held 1–12 h (capability control)
Truth (query): from 20 start times in days 42–56, replay the embedded controller up to the start, then hold modulation at u0 = 0.4
(reference branch) or u0 + 0.3 (intervention branch) for 24 h; true effect = difference of zone temperature (exact, deterministic).
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import bop

DT, K = 900.0, 273.15
DAY = 86400.0
TRAIN0, TRAIN_DAYS = 7 * DAY, 28
TEST0, TEST_DAYS = 42 * DAY, 14
H = 96                                   # 24 h
U0, DU = 0.4, 0.3
CASES = {   # zone temperature, applied input, outdoor T, solar, set-point the embedded controller tracks; input / set-point override bases
    'bestest_hydronic_heat_pump': dict(names=['reaTZon_y', 'oveHeaPumY_y', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHGloHor_y', 'oveTSet_y'],
                                       u='oveHeaPumY', r='oveTSet', levels=[0.1, 0.25, 0.4, 0.55, 0.7], data='c7_data'),
    'singlezone_commercial_hydronic': dict(names=['reaTZon_y', 'oveValRad_y', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHGloHor_y', 'oveTZonSet_y'],
                                       u='oveValRad', r='oveTZonSet', levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], data='c7_data_commercial'),
}
CASE = os.environ.get('C7_CASE', 'bestest_hydronic_heat_pump'); CFG = CASES[CASE]; DATA = CFG['data']
NAMES = CFG['names']   # oveTSet_y = set-point the embedded PI tracks (21.2 occ / 20.5 unocc); reaTSetHea_y is only the 15/21 KPI band


SCHED = None
LEVELS = CFG['levels']                       # heat pump v1: around the operating region (v0 {0..1} drove the zone to 27±4 °C, outside the query's region)
def run_log(regime, A=0.0, seed=0, t0=TRAIN0, days=TRAIN_DAYS, warm_days=7):
    rng = np.random.default_rng(seed)
    e = bop.Emulator(CASE, name=f'{regime}{A}{seed}'); e.init(t0 - warm_days * DAY)
    while e.t < t0 - 1: e.step(DT)                                  # warm-up under the embedded controller
    n = int(days * DAY / DT); out = np.empty((n, 5)); hold = 0; off = 0.0; lvl = 0.0
    for k in range(n):
        ov = None
        if regime == 'dither':
            if hold <= 0: off = rng.uniform(-A, A); hold = rng.integers(4, 25)
            hold -= 1
            sched = SCHED[k]                                         # embedded schedule, from the operational log (reaTSetHea_y reports the
            ov = {CFG['r']: sched + K + off}                        # OVERRIDDEN value once oveTSet is active — v0 bug: offsets accumulated)
        elif regime == 'designed':
            if hold <= 0: lvl = rng.choice(LEVELS); hold = rng.integers(4, 49)
            hold -= 1; ov = {CFG['u']: lvl}
        e.step(DT, ov)
        v = e.get(NAMES); out[k] = [v[0] - K, v[1], v[2] - K, v[3], v[4] - K]
    e.close(); return out


def truth_effect(ts):
    """true 24-h effect of +DU modulation from start time ts (embedded control before ts), plus the 24-h pre-history."""
    res = []
    for u in (U0, U0 + DU):
        e = bop.Emulator(CASE, name=f'tr{ts}{u}'); e.init(TRAIN0 - 7 * DAY)
        pre = []
        while e.t < ts - 1:
            e.step(DT)
            if e.t > ts - DAY - 1: v = e.get(NAMES); pre.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
        traj = []
        for k in range(H):
            e.step(DT, {CFG['u']: u}); v = e.get(NAMES); traj.append([v[0] - K, v[1], v[2] - K, v[3], v[4] - K])
        e.close(); res.append((np.array(pre), np.array(traj)))
    return dict(ts=ts, pre=res[0][0].tolist(), base=res[0][1].tolist(), interv=res[1][1].tolist(),
                effect=(res[1][1][:, 0] - res[0][1][:, 0]).tolist())


def _log_job(a): return a, run_log(*a)


def _test_job(sd): return sd, run_log('designed', 0.0, sd, t0=TEST0, days=TEST_DAYS)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    fo = f'{DATA}/log_operational_A0.0_s0.npy'
    if not os.path.exists(fo): np.save(fo, run_log('operational', 0.0, 0))
    SCHED = np.load(fo)[:, 4]                                               # °C, embedded set-point schedule (set before forking the pool)
    jobs = [('dither', A, 0) for A in (0.5, 1.0, 2.0)] + [('designed', 0.0, s) for s in (0, 1)]
    starts = [TEST0 + i * (TEST_DAYS * DAY / 20) + 6 * 3600 for i in range(20)]
    with Pool(12) as p:
        for (reg, A, s), out in p.imap_unordered(_log_job, jobs):
            np.save(f'{DATA}/log_{reg}_A{A}_s{s}.npy', out)
        for sd, out in p.map(_test_job, [10, 11]): np.save(f'{DATA}/test_designed_s{sd}.npy', out)
        tr = p.map(truth_effect, starts)
    json.dump(tr, open(f'{DATA}/truth.json', 'w'))
    for f in sorted(os.listdir(DATA)):
        if f.endswith('.npy'):
            out = np.load(f'{DATA}/{f}')
            print(f'{f:32s} y {out[:,0].mean():.2f}±{out[:,0].std():.2f}  u {out[:,1].mean():.2f}±{out[:,1].std():.2f}  '
                  f'corr(u,y) {np.corrcoef(out[:,0],out[:,1])[0,1]:+.2f}  r {out[:,4].min():.1f}..{out[:,4].max():.1f}')
    E = np.array([t['effect'] for t in tr])
    print('true effect of +0.3 input @1/3/6/12/24 h (mean over 20 starts):', E.mean(0)[[3, 11, 23, 47, 95]].round(2), ' sd', E.std(0)[[3, 11, 23, 47, 95]].round(2))
