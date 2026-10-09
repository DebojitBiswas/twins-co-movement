#!/usr/bin/env python3
"""
C23 — round-4 control referee ("no test separates controller co-movement from a generic low-frequency gain deficit"): the SHUFFLED-DAY REPLAY.
The operational log's modulation sequence is cut into its 28 days, the days are randomly permuted and the permuted sequence is replayed
open-loop in the emulator over the same training window (same weather, same occupancy, same marginal distribution and nearly the same
spectrum of u, but no alignment between u and the hidden disturbance the controller was reacting to).  Twins trained on the replay log are
scored exactly as the operational ones (step from operational start states, c15.evaluate).  If the shortfall is the controller's
co-movement, it should vanish here; if it is the input spectrum (informativity), it should remain.
Stage 1: C7_V1_PERIOD=P1 venv_ap/bin/python c23_shuffled_replay.py log      -> c7_v1/P*/log_shuffled_s{0,1}.npy
Stage 2: C7_V1_PERIOD=P1 python3 c23_shuffled_replay.py models              -> c7_v1/P*/c23_shuffled.json
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'
PERIODS = {'P1': 7, 'P2': 45, 'P3': 300}


def make_log(seed):
    import bop, c7_boptest as C, c14_gen_v1 as G
    DAY, DT, K = C.DAY, C.DT, C.K
    op = np.load(f'{D}/log_operational.npy'); n = len(op); per_day = int(DAY / DT); days = n // per_day
    rng = np.random.default_rng(seed); MODE = os.environ.get('C23_MODE', 'shuffled')
    if MODE == 'strat':   # weather-stratified: permute days only within quartiles of the daily mean outdoor temperature (keeps corr(u, T_out))
        tm = np.array([op[d * per_day:(d + 1) * per_day, 2].mean() for d in range(days)]); q = np.argsort(np.argsort(tm)) * 4 // days
        perm = np.arange(days)
        for k in range(4):
            idx = np.where(q == k)[0]; perm[idx] = rng.permutation(idx)
    else: perm = rng.permutation(days)
    u = np.concatenate([op[d * per_day:(d + 1) * per_day, 1] for d in perm])
    t0 = PERIODS[P] * DAY; e = bop.Emulator(C.CASE, name=f'{MODE}{P}{seed}'); e.init(t0 - 7 * DAY)
    while e.t < t0 - 1: e.step(DT)
    out = np.empty((n, 5))
    for k in range(n):
        e.step(DT, {C.CFG['u']: float(u[k])}); v = e.get(C.NAMES); out[k] = [v[0] - K, v[1], v[2] - K, v[3], v[4] - K]
    e.close(); np.save(f'{D}/log_{MODE}_s{seed}.npy', out)
    print(f'{P} {MODE} replay s{seed}: corr(u,Tout) {np.corrcoef(out[:, 1], out[:, 2])[0, 1]:.2f} (op {np.corrcoef(op[:, 1], op[:, 2])[0, 1]:.2f}), perm {perm[:6].tolist()}..., y range {out[:, 0].min():.1f}-{out[:, 0].max():.1f} (operational {op[:, 0].min():.1f}-{op[:, 0].max():.1f}), u mean {out[:, 1].mean():.3f} vs {op[:, 1].mean():.3f}, corr(u_replay, u_op) {np.corrcoef(out[:, 1], op[:, 1])[0, 1]:.2f}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'log':
        with Pool(2) as p: p.map(make_log, [0, 1])
    else:
        import torch; torch.set_num_threads(1)
        import c15_v1_models as V
        fams = {'gru_sim': 5, 'tf_1s': 3, 'gru_1s_noise': 3, 'tf_1s_noise': 3, 'arx12': 1, 'arx48': 1, 'pem2': 1}
        MODE = os.environ.get('C23_MODE', 'shuffled')
        jobs = [(f, f'log_{MODE}_s{s}', k) for f, ns in fams.items() for s in (0, 1) for k in range(ns)]
        with Pool(int(os.environ.get('WORKERS', 12))) as p: res = p.map(V.job, jobs)
        json.dump(res, open(f'{D}/c23_{MODE}.json', 'w')); print(f'{P}: {len(res)} {MODE}-replay models')
