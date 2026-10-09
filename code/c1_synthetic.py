#!/usr/bin/env python3
"""
C1 — synthetic kill-test for paper C.

Plant: two-node thermal (air Ta observed through a noisy, quantised sensor; mass Tm hidden), hidden slow disturbance q (AR(1)) on the air node.
Controller: PI on the measured air temperature to a reference r_t, optional feed-forward of the hidden disturbance (gain d), input clipped.
Dither: the reference follows random steps of amplitude sigma_r (0 = pure operational log).  The reference is logged (usable as an instrument).
Query: the causal EFFECT of a designed open-loop input sequence (ROLBS-like) relative to holding the operating input, from steady state,
horizon H — exactly the quantity of paper B's B9.  Score: relative RMSE of the model's effect trajectory against the true effect.

Models (all see only y and u; IV additionally sees r):
  arx{p}   one-step least squares (the 'direct method'), orders 2 and 20
  iv{p}    two-stage least squares with lagged reference as instruments (needs sigma_r > 0)
  gru      simulation-error-trained GRU, ensemble of E members; ensemble spread reported (is the model aware of its error?)

Usage: python c1_synthetic.py --models lin            # full grid, linear models (seconds)
       python c1_synthetic.py --models gru --grid sub # GRU ensembles on a sub-grid (minutes, CPU pool)
"""
import argparse, itertools, json, os, time
import numpy as np
from scipy.linalg import expm

# ------------------------------------------------------------------ plant
CA, CM, H_AM, KA, KM, B = 1.0, 30.0, 0.3, 0.05, 0.02, 1.0
T0, TREF = 0.0, 20.0
SIG_WA, SIG_WM, SIG_V = 0.01, 0.002, 0.02
RHO_Q, SIG_Q = 0.99, 0.5
UMAX = 6.0
Ac = np.array([[-(H_AM + KA) / CA, H_AM / CA], [H_AM / CM, -(H_AM + KM) / CM]])
Bc = np.array([[B / CA, 1.0 / CA], [0.0, 0.0]])          # columns: u, q
Ad = expm(Ac)
Bd = np.linalg.solve(Ac, (Ad - np.eye(2))) @ Bc           # zero-order hold, dt = 1


def steady_input(T):
    # steady state with q = 0: solve Ac x + Bc[:,0] u = 0 with x_a = T
    # from mass row: x_m = H_AM T / (H_AM + KM)
    xm = H_AM * T / (H_AM + KM)
    return ((H_AM + KA) * T - H_AM * xm) / B, xm


U_OP, XM_OP = steady_input(TREF)


def simulate(N, kp, d, sigma_r, res, seed, burn=2000):
    rng = np.random.default_rng(seed)
    ki = kp / 20.0
    x = np.array([TREF, XM_OP]); q = 0.0; integ = 0.0
    r_val = TREF; hold = 0
    Y, U, R, Q = (np.empty(N) for _ in range(4))
    for t in range(N + burn):
        if hold <= 0:
            r_val = TREF + sigma_r * rng.uniform(-1, 1); hold = rng.integers(20, 200)
        hold -= 1
        y = x[0] + SIG_V * rng.standard_normal()
        if res > 0: y = np.round(y / res) * res
        e = r_val - y
        u_raw = U_OP + kp * e + ki * integ - d * q
        u = min(max(u_raw, 0.0), UMAX)
        if u == u_raw: integ += e                       # anti-windup
        if t >= burn:
            i = t - burn; Y[i], U[i], R[i], Q[i] = y, u, r_val, q
        x = Ad @ x + Bd @ np.array([u, q]) + np.array([SIG_WA, SIG_WM]) * rng.standard_normal(2)
        q = RHO_Q * q + np.sqrt(1 - RHO_Q ** 2) * SIG_Q * rng.standard_normal()
    return Y, U, R, Q


def design_sequence(H, seed=12345):
    rng = np.random.default_rng(seed)
    u = np.empty(H); t = 0
    while t < H:
        L = int(np.exp(rng.uniform(np.log(5), np.log(200)))); lvl = rng.choice([0.0, 0.5 * UMAX])
        u[t:t + L] = lvl; t += L
    return u


H = 600
U_DES = design_sequence(H)


def true_effect(u_des=U_DES):
    x = np.zeros(2); out = np.empty(len(u_des))
    for t, u in enumerate(u_des):
        out[t] = x[0]; x = Ad @ x + Bd[:, 0] * (u - U_OP)
    return out


EFF_TRUE = true_effect()

# ------------------------------------------------------------------ linear models
def lagmat(z, p, start):
    return np.column_stack([z[start - i:len(z) - i - 1] for i in range(p)]) if p else np.empty((len(z) - start - 1, 0))


def arx_design(Y, U, p):
    s = p
    X = np.column_stack([lagmat(Y, p, s - 1), lagmat(U, p, s - 1), np.ones(len(Y) - s)])
    return X, Y[s:]


def arx_effect(a, b, u_des=U_DES):
    p = len(a); ybuf = np.zeros(p); ubuf = np.zeros(p); out = np.empty(len(u_des))
    for t, u in enumerate(u_des):
        out[t] = ybuf[0]
        ubuf = np.r_[u - U_OP, ubuf[:-1]]
        yn = a @ ybuf + b @ ubuf
        if not np.isfinite(yn) or abs(yn) > 1e6: out[t:] = np.nan; return out
        ybuf = np.r_[yn, ybuf[:-1]]
    return out


def fit_arx(Y, U, p):
    X, y = arx_design(Y, U, p)
    th = np.linalg.lstsq(X, y, rcond=None)[0]
    return arx_effect(th[:p], th[p:2 * p])


def fit_iv(Y, U, R, p, m=None):
    m = m or 2 * p + 4
    X, y = arx_design(Y, U, p)
    s = p
    # instruments: lags of r (exogenous), going back far enough to cover the regressors
    Z = np.column_stack([np.r_[np.full(i, R[0]), R[:len(R) - i]][s:][:len(y)] for i in range(1, m + 1)] + [np.ones(len(y))])
    # 2SLS
    PZ = np.linalg.lstsq(Z, X, rcond=None)[0]
    Xh = Z @ PZ
    th = np.linalg.lstsq(Xh, y, rcond=None)[0]
    return arx_effect(th[:p], th[p:2 * p])


def _oe_fit(Y, Uin, u0):
    """2nd-order output-error model y - c = (b1 z^-1 + b2 z^-2)/(1 + a1 z^-1 + a2 z^-2) (u - u0), simulation-error least squares."""
    from scipy.signal import lfilter
    from scipy.optimize import least_squares
    du = Uin - u0
    X, y = arx_design(Y, Uin, 2); th = np.linalg.lstsq(X, y, rcond=None)[0]
    a0 = -th[:2]; 
    if np.max(np.abs(np.roots(np.r_[1, a0]))) >= 0.999: a0 = np.array([-1.5, 0.55])
    def res(p):
        a1, a2, b1, b2, c = p
        if np.max(np.abs(np.roots([1, a1, a2]))) >= 0.9999: return np.full(len(Y), 1e3)
        return lfilter([0, b1, b2], [1, a1, a2], du) + c - Y
    p0 = np.r_[a0, th[2:4], Y.mean()]
    best = None
    for init in (p0, np.r_[-1.9, 0.905, 0.1, 0.0, Y.mean()], np.r_[-1.0, 0.1, 0.5, 0.0, Y.mean()]):
        try:
            r = least_squares(res, init, method="lm", max_nfev=4000)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    a1, a2, b1, b2, c = best.x
    return lfilter([0, b1, b2], [1, a1, a2], U_DES - U_OP)


def fit_oe(Y, U):
    return _oe_fit(Y, U, U_OP)


def fit_twostage(Y, U, R, M=None):
    """Two-stage closed-loop identification (Van den Hof & Schrama 1993): u_hat = FIR(r) by least squares, then OE fit of y on u_hat."""
    M = M or int(min(300, len(Y) // 20))
    Z = np.column_stack([np.r_[np.full(i, R[0]), R[:len(R) - i]] for i in range(M + 1)] + [np.ones(len(R))])
    uh = Z @ np.linalg.lstsq(Z, U, rcond=None)[0]
    return _oe_fit(Y, uh, U_OP)


def rel_err(eff):
    if eff is None or not np.all(np.isfinite(eff)): return float("inf")
    return float(np.sqrt(np.mean((eff - EFF_TRUE) ** 2)) / np.sqrt(np.mean(EFF_TRUE ** 2)))


# ------------------------------------------------------------------ certificate quantities (log only)
def certificate(Y, U, R, p=20, tau=100):
    """(a) share of input variance NOT predictable from past y, u  (innovation share);
       (b) excitation number: s.d. of y smoothed over the slow time-scale tau, divided by the one-step innovation s.d. of y."""
    X, uy = arx_design(U, Y, p)  # regress u_{t} on past u and y: reuse design with roles (target U)
    Xu = np.column_stack([lagmat(Y, p, p - 1), lagmat(U, p, p - 1), np.ones(len(Y) - p)])
    tu = U[p:]
    resid_u = tu - Xu @ np.linalg.lstsq(Xu, tu, rcond=None)[0]
    share = float(np.var(resid_u) / np.var(U))
    Xy, ty = arx_design(Y, U, p)
    resid_y = ty - Xy @ np.linalg.lstsq(Xy, ty, rcond=None)[0]
    k = np.ones(tau) / tau
    ys = np.convolve(Y - Y.mean(), k, mode="valid")
    xi = float(np.std(ys) / max(np.std(resid_y), 1e-9))
    return dict(innov_share_u=share, xi=xi, sd_y=float(np.std(Y)), sd_u=float(np.std(U)))


# ------------------------------------------------------------------ GRU ensemble
def gru_member(args):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    Y, U, seed, steps, warm, roll = args
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    my, sy, mu, su = Y.mean(), Y.std() + 1e-6, U.mean(), U.std() + 1e-6
    yn = torch.tensor((Y - my) / sy, dtype=torch.float32); un = torch.tensor((U - mu) / su, dtype=torch.float32)

    class Sim(nn.Module):
        def __init__(s, h=32):
            super().__init__(); s.cell = nn.GRUCell(2, h); s.out = nn.Linear(h, 1); s.h = h
        def forward(s, y0, u, warm):  # y0: (B, warm) observed, u: (B, warm+roll)
            Bsz, L = u.shape; hs = torch.zeros(Bsz, s.h); yprev = y0[:, 0]; outs = []
            for t in range(L):
                yin = y0[:, t] if t < warm else yprev
                hs = s.cell(torch.stack([yin, u[:, t]], 1), hs)
                yprev = yin + 0.1 * s.out(hs)[:, 0]
                outs.append(yprev)
            return torch.stack(outs, 1)

    m = Sim(); opt = torch.optim.Adam(m.parameters(), 3e-3)
    L = warm + roll; Bsz = 32
    for it in range(steps):
        idx = rng.integers(0, len(Y) - L - 1, Bsz)
        yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
        pred = m(yb[:, :warm], ub, warm)
        loss = ((pred[:, warm - 1:] - yb[:, warm:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    # effect: from a steady-state warm-up at (TREF, U_OP) roll designed vs baseline input
    with torch.no_grad():
        y0 = torch.full((2, warm), (TREF - my) / sy)
        useq = np.r_[np.full(warm, U_OP), U_DES]
        ub = torch.tensor(np.stack([(useq - mu) / su, (np.r_[np.full(warm, U_OP), np.full(H, U_OP)] - mu) / su]), dtype=torch.float32)
        out = m(y0, ub, warm).numpy() * sy + my
    eff = out[0, warm - 1:warm - 1 + H] - out[1, warm - 1:warm - 1 + H]
    return eff, float(loss.item())


# ------------------------------------------------------------------ driver
def lin_cell(cell):
    kp, d, sr, res, N, s = cell
    Y, U, R, Q = simulate(N, kp, d, sr, res, seed=1000 * s + 7)
    row = dict(kp=kp, d=d, sigma_r=sr, res=res, N=N, seed=s, **certificate(Y, U, R))
    row["corr_uq"] = float(np.corrcoef(U, Q)[0, 1])
    for p in (2, 20):
        row[f"arx{p}"] = rel_err(fit_arx(Y, U, p))
    row["oe2"] = rel_err(fit_oe(Y, U))
    row["twostage"] = rel_err(fit_twostage(Y, U, R)) if sr > 0 else None
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="lin"); ap.add_argument("--grid", default="full")
    ap.add_argument("--ens", type=int, default=5); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.grid == "full":
        KP, D, SR, RES, NN, SEEDS = [0.5, 2.0, 8.0], [0.0, 0.8], [0.0, 0.05, 0.2, 1.0], [0.01, 0.5], [2000, 20000, 200000], [0, 1, 2]
    else:
        KP, D, SR, RES, NN, SEEDS = [0.5, 8.0], [0.0, 0.8], [0.0, 0.2], [0.01], [2000, 20000, 200000], [0]
    rows = []; t0 = time.time()
    cells = list(itertools.product(KP, D, SR, RES, NN, SEEDS))
    if a.models == "lin":
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            rows = pool.map(lin_cell, cells)
        print(f"{len(rows)} cells in {time.time() - t0:.0f}s")
    else:
        from multiprocessing import Pool
        jobs, meta = [], []
        for kp, d, sr, res, N, s in cells:
            Y, U, R, Q = simulate(N, kp, d, sr, res, seed=1000 * s + 7)
            for e in range(a.ens):
                jobs.append((Y, U, 100 * e + s, a.steps, 50, 150)); meta.append((kp, d, sr, res, N, s, e))
        with Pool(a.workers) as pool:
            res_ = pool.map(gru_member, jobs)
        by = {}
        for (kp, d, sr, res, N, s, e), (eff, loss) in zip(meta, res_):
            by.setdefault((kp, d, sr, res, N, s), []).append((eff, loss))
        for (kp, d, sr, res, N, s), lst in by.items():
            E = np.stack([x[0] for x in lst]); mean = E.mean(0)
            spread = float(np.sqrt(np.mean(E.var(0))) / np.sqrt(np.mean(EFF_TRUE ** 2)))
            rows.append(dict(kp=kp, d=d, sigma_r=sr, res=res, N=N, seed=s, gru=rel_err(mean), gru_members=[rel_err(x) for x in E],
                             eff_mean=mean[::10].round(3).tolist(), gru_spread=spread, train_loss=float(np.mean([x[1] for x in lst]))))
        print(f"{len(rows)} cells, {len(jobs)} GRUs in {time.time() - t0:.0f}s")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"c1_results_{a.models}_{a.grid}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
