#!/usr/bin/env python3
"""
C13 — referee M2/M10: the classical closed-loop-consistent baselines on the BOPTEST heat-pump building (and commercial, via C7_CASE).
  ARX(p)      direct method with an autoregressive noise model: y_{t+1} on y_{t-i} (i=0..p-1), u_{t-i}, x_{t-i}, 1; least squares. p ∈ {2, 6, 12, 24, 48}.
  PEM-SS(n)   direct prediction-error method with a full noise model: innovations-form state space x_{t+1} = A x + B w + K e, y = C x + D w + e,
              w = (u, T_out, solar); n ∈ {2, 3, 4}; fitted by minimising the one-step prediction error (Gaussian ML) with L-BFGS from several inits.
  GREY(2R2C)  sign-constrained grey-box: C_a dT_a = h (T_m − T_a) + k_a (T_out − T_a) + g_u u + g_s solar; C_m dT_m = h (T_a − T_m) + k_m (T_out − T_m);
              exact discretisation (expm); innovations gain K free; parameters ≥ 0 (softplus); fitted by PEM.
All are evaluated as SIMULATORS: the predictor (with innovations) runs through the 24-h warm-up to set the state, then 48 h of pure simulation
(e = 0) with the true inputs → do-RMSE on the designed test schedules (as c7_neural); learned +0.3 step response from the same contexts, and the
within-support step from OPERATIONAL contexts (test weeks), both reported.  Training logs: operational, dither ±2 K, designed; truth: 1.92 K (24 h).
"""
import os, sys, json, numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from multiprocessing import Pool

DATA = {'bestest_hydronic_heat_pump': 'c7_data', 'singlezone_commercial_hydronic': 'c7_data_commercial'}[os.environ.get('C7_CASE', 'bestest_hydronic_heat_pump')]
WARM, ROLL = 96, 192


def norm_stats(L):
    y, w = L[:, 0], L[:, 1:4]; return y.mean(), y.std() + 1e-9, w.mean(0), w.std(0) + 1e-9


# ------------------------------------------------------------------ ARX
def arx_fit(L, p):
    y, w = L[:, 0], L[:, 1:4]; n = len(y)
    X = np.column_stack([y[p - 1 - i:n - 1 - i] for i in range(p)] + [w[p - 1 - i:n - 1 - i, j] for i in range(p) for j in range(3)] + [np.ones(n - p)])
    th = np.linalg.lstsq(X, y[p:], rcond=None)[0]; return dict(p=p, th=th)


def arx_sim(mdl, yh, w):
    """yh: warm-up outputs (≥ p); w: inputs over warm-up + horizon; returns simulated outputs over the horizon."""
    p, th = mdl['p'], mdl['th']; ys = list(yh); out = []
    for t in range(len(yh), len(w)):
        reg = np.r_[[ys[t - 1 - i] for i in range(p)], [w[t - 1 - i, j] for i in range(p) for j in range(3)], 1.0]
        ys.append(float(reg @ th)); out.append(ys[-1])
    return np.array(out)


# ------------------------------------------------------------------ innovations-form state space (PEM)
def ss_unpack(theta, n, grey=False):
    if grey:
        Ca, Cm, h, ka, km, gu, gs = np.log1p(np.exp(theta[:7])) + 1e-6
        Ac = np.array([[-(h + ka) / Ca, h / Ca], [h / Cm, -(h + km) / Cm]]); Bc = np.array([[gu / Ca, ka / Ca, gs / Ca], [0.0, km / Cm, 0.0]])
        M = expm(np.block([[Ac, Bc], [np.zeros((3, 5))]]) * 1.0); A, B = M[:2, :2], M[:2, 2:]
        C = np.array([[1.0, 0.0]]); D = np.zeros((1, 3)); K = theta[7:9].reshape(2, 1); c0 = theta[9]
        return A, B, C, D, K, c0
    i = 0; A = theta[i:i + n * n].reshape(n, n); i += n * n; B = theta[i:i + 3 * n].reshape(n, 3); i += 3 * n
    C = theta[i:i + n].reshape(1, n); i += n; D = theta[i:i + 3].reshape(1, 3); i += 3; K = theta[i:i + n].reshape(n, 1); i += n; c0 = theta[i]
    return A, B, C, D, K, c0


def ss_predict(theta, n, y, w, grey, burn=48):
    A, B, C, D, K, c0 = ss_unpack(theta, n, grey); x = np.zeros((A.shape[0], 1)); err = np.empty(len(y))
    for t in range(len(y)):
        yp = (C @ x)[0, 0] + (D @ w[t])[0] + c0; e = y[t] - yp; err[t] = e
        x = A @ x + (B @ w[t])[:, None] + K * e
        if not np.isfinite(x).all() or np.abs(x).max() > 1e6: return np.full(len(y), 1e3)
    return err[burn:]


def ss_fit(L, n, grey=False, seed=0):
    my, sy, mw, sw = norm_stats(L); y = (L[:, 0] - my) / sy; w = (L[:, 1:4] - mw) / sw
    rng = np.random.default_rng(seed); best = None
    for trial in range(6):
        if grey:
            th0 = np.r_[rng.normal(0, 1, 7), rng.normal(0, 0.1, 2), 0.0]
        else:
            A0 = np.diag(rng.uniform(0.6, 0.98, n)) + rng.normal(0, 0.02, (n, n))
            th0 = np.r_[A0.ravel(), rng.normal(0, 0.1, 3 * n), rng.normal(0, 0.3, n), np.zeros(3), rng.normal(0, 0.1, n), 0.0]
        f = lambda th: float(np.mean(ss_predict(th, n, y, w, grey) ** 2))
        r = minimize(f, th0, method='L-BFGS-B', options=dict(maxiter=400))
        if best is None or r.fun < best.fun: best = r
    return dict(n=n, grey=grey, theta=best.x, norm=(my, sy, mw, sw), fit_rmse=float(np.sqrt(best.fun) * sy))


def ss_sim(mdl, yh, w):
    my, sy, mw, sw = mdl['norm']; A, B, C, D, K, c0 = ss_unpack(mdl['theta'], mdl['n'], mdl['grey'])
    yn = (yh - my) / sy; wn = (w - mw) / sw; x = np.zeros((A.shape[0], 1))
    for t in range(len(yh)):                                              # predictor through the warm-up
        e = yn[t] - ((C @ x)[0, 0] + (D @ wn[t])[0] + c0); x = A @ x + (B @ wn[t])[:, None] + K * e
    out = []
    for t in range(len(yh), len(w)):                                      # pure simulation
        yp = (C @ x)[0, 0] + (D @ wn[t])[0] + c0; out.append(yp * sy + my); x = A @ x + (B @ wn[t])[:, None]
    return np.array(out)


# ------------------------------------------------------------------ evaluation
def evaluate(sim, name):
    out = dict(model=name, test={}, step_designed_ctx=[], step_operational_ctx=[])
    for s in (10, 11):
        te = np.load(f'{DATA}/test_designed_s{s}.npy'); errs, pers, resp = [], [], []
        for st in range(WARM, len(te) - ROLL, ROLL // 2):
            seg = te[st - WARM:st + ROLL]; pr = sim(seg[:WARM, 0], seg[:, 1:4])
            errs.append(np.mean((pr - seg[WARM:, 0]) ** 2)); pers.append(np.mean((seg[WARM - 1, 0] - seg[WARM:, 0]) ** 2))
            w2 = [np.column_stack([np.r_[seg[:WARM, 1], np.full(ROLL, lv)], seg[:, 2:4]]) for lv in (0.4, 0.7)]
            resp.append(sim(seg[:WARM, 0], w2[1]) - sim(seg[:WARM, 0], w2[0]))
        out['test'][s] = dict(do_rmse=float(np.sqrt(np.mean(errs))), persistence=float(np.sqrt(np.mean(pers)))); out['step_designed_ctx'].append(np.mean(resp, 0).round(3).tolist())
    op = np.load(f'{DATA}/test_operational.npy'); resp, errs = [], []
    for st in range(WARM, len(op) - ROLL, ROLL // 2):
        seg = op[st - WARM:st + ROLL]; pr = sim(seg[:WARM, 0], seg[:, 1:4]); errs.append(np.mean((pr - seg[WARM:, 0]) ** 2))
        w2 = [np.column_stack([np.r_[seg[:WARM, 1], np.full(ROLL, lv)], seg[:, 2:4]]) for lv in (0.4, 0.7)]
        resp.append(sim(seg[:WARM, 0], w2[1]) - sim(seg[:WARM, 0], w2[0]))
    out['pred_rmse_operational'] = float(np.sqrt(np.mean(errs))); out['step_operational_ctx'] = np.mean(resp, 0).round(3).tolist()
    return out


def job(a):
    kind, train, arg = a
    L = np.load(f'{DATA}/log_{train}.npy')
    if kind == 'arx': m = arx_fit(L, arg); sim = lambda yh, w: arx_sim(m, yh, w); name = f'ARX({arg})'
    elif kind == 'ss': m = ss_fit(L, arg); sim = lambda yh, w: ss_sim(m, yh, w); name = f'PEM-SS({arg})'
    else: m = ss_fit(L, 2, grey=True); sim = lambda yh, w: ss_sim(m, yh, w); name = 'GREY-2R2C'
    r = evaluate(sim, name); r['train'] = train
    if kind != 'arx': r['fit_rmse'] = m['fit_rmse']
    print(f"{name:12s} {train:20s}: do-RMSE {r['test'][10]['do_rmse']:.2f} / {r['test'][11]['do_rmse']:.2f} K (persistence {r['test'][10]['persistence']:.2f} / {r['test'][11]['persistence']:.2f})  "
          f"pred-op {r['pred_rmse_operational']:.2f}  step24h designed-ctx {np.mean(r['step_designed_ctx'], 0)[95]:.2f} / operational-ctx {np.array(r['step_operational_ctx'])[95]:.2f} K", flush=True)
    return r


if __name__ == '__main__':
    trains = ['operational_A0.0_s0', 'dither_A2.0_s0', 'designed_A0.0_s0']
    jobs = [('arx', t, p) for t in trains for p in (2, 6, 12, 24, 48)] + [('ss', t, n) for t in trains for n in (2, 3, 4)] + [('grey', t, 0) for t in trains]
    if 'ssonly' in sys.argv: jobs = [j for j in jobs if j[0] != 'arx']
    with Pool(int(os.environ.get('WORKERS', 6))) as p: rows = p.map(job, jobs)
    out = f'{DATA}/c13_pem.json'
    old = json.load(open(out)) if os.path.exists(out) else []
    json.dump(old + rows, open(out, 'w'))
