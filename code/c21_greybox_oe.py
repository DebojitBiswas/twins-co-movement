#!/usr/bin/env python3
"""
C21 — referee (control MAJOR 7): a grey-box 2R2C model fitted the way building practice does it — by SIMULATION ERROR with non-negative parameters
(no innovations gain), so the structure cannot be bypassed: C_a dT_a = h (T_m − T_a) + k_a (T_out − T_a) + g_u u + g_s S;  C_m dT_m = h (T_a − T_m)
+ k_m (T_out − T_m).  Exact discretisation; initial mass temperature per 48-h window estimated by a short least-squares on the window's first 24 h.
Trained on operational / dither / designed logs of each period (6 random inits, best), evaluated as in c18 (step from operational contexts, in-range
test, operational test).  Output c7_v1/P*/c21_greybox_oe.json.
"""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
from scipy.linalg import expm
from multiprocessing import Pool
import c15_v1_models as V

P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'


def mats(theta):
    Ca, Cm, h, ka, km, gu, gs = np.log1p(np.exp(theta)) + 1e-6
    Ac = np.array([[-(h + ka) / Ca, h / Ca], [h / Cm, -(h + km) / Cm]]); Bc = np.array([[gu / Ca, ka / Ca, gs / Ca], [0.0, km / Cm, 0.0]])
    M = expm(np.block([[Ac, Bc], [np.zeros((3, 5))]])); return M[:2, :2], M[:2, 2:]


def simulate(A, B, x0, w):
    x = x0.copy(); out = np.empty(len(w))
    for t in range(len(w)):
        out[t] = x[0]; x = A @ x + B @ w[t]
    return out


def fit(L, seed):
    y, w = L[:, 0], L[:, 1:4]; rng = np.random.default_rng(seed); best = None
    def resid(th):
        A, B = mats(th[:7]); x0 = np.array([y[0], th[7]]); return simulate(A, B, x0, w) - y
    for trial in range(6):
        th0 = np.r_[rng.normal([0, 3, 0, -1, -2, 1, -3], 0.5), y[0]]
        try:
            r = least_squares(resid, th0, method='trf', max_nfev=300)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return best.x


def make_sim(theta):
    A, B = mats(theta[:7])
    def sim(yh, w):
        # estimate the mass temperature at the start of the warm-up by least squares on the warm-up, then simulate
        def res(tm0): return simulate(A, B, np.array([yh[0], tm0[0]]), w[:len(yh)]) - yh
        tm0 = least_squares(res, [yh[0]], method='trf', max_nfev=50).x[0]
        full = simulate(A, B, np.array([yh[0], tm0]), w); return full[len(yh):]
    return sim


def job(a):
    train, seed = a; L = np.load(f'{D}/{train}.npy'); theta = fit(L, seed); sim = make_sim(theta)
    r = dict(family='grey_oe', train=train, seed=seed, period=P, theta=theta.tolist(), **V.evaluate(sim))
    print(f"{P} grey_oe {train:18s} s{seed}: in {r['test']['test_in']['rmse']:.2f} op {r['test']['test_operational']['rmse']:.2f} | step24 ratio {r['step']['ratio_24h']:.2f}", flush=True); return r


if __name__ == '__main__':
    trains = ['log_operational', 'log_dither_A2.0_s0', 'log_dither_A2.0_s1', 'log_dither_A2.0_s2', 'log_designed_s0', 'log_designed_s1']
    with Pool(int(os.environ.get('WORKERS', 3))) as p: rows = p.map(job, [(t, 0) for t in trains])
    json.dump(rows, open(f'{D}/c21_greybox_oe.json', 'w'))
