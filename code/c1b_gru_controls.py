"""C1b — controls for the GRU result: (i) open-loop designed data (is the GRU capable?), (ii) larger dither,
(iii) instrumented GRU: trained on u_hat = projection of u on lagged reference (two-stage idea applied to a neural simulator)."""
import json, numpy as np
from multiprocessing import Pool
import c1_synthetic as c

def openloop(N, seed):
    rng = np.random.default_rng(seed)
    u = np.concatenate([c.design_sequence(600, seed=seed * 1000 + s) for s in range(N // 600 + 2)])[:N]
    x = np.array([c.TREF, c.XM_OP]); q = 0.0; Y = np.empty(N)
    for t in range(N):
        y = x[0] + c.SIG_V * rng.standard_normal(); Y[t] = np.round(y / 0.01) * 0.01
        x = c.Ad @ x + c.Bd @ np.array([u[t], q]) + np.array([c.SIG_WA, c.SIG_WM]) * rng.standard_normal(2)
        q = c.RHO_Q * q + np.sqrt(1 - c.RHO_Q ** 2) * c.SIG_Q * rng.standard_normal()
    return Y, u

def project(U, R, M=300):
    Z = np.column_stack([np.r_[np.full(i, R[0]), R[:len(R) - i]] for i in range(M + 1)] + [np.ones(len(R))])
    return Z @ np.linalg.lstsq(Z, U, rcond=None)[0]

if __name__ == "__main__":
    E = 5; jobs, meta = [], []
    for N in (2000, 20000, 200000):
        Y, U = openloop(N, 3)
        for e in range(E): jobs.append((Y, U, 100 * e, 1500, 50, 150)); meta.append(("open", 0, 0, 0, N))
    for d in (0.0, 0.8):
        for N in (20000, 200000):
            Y, U, R, Q = c.simulate(N, 2.0, d, 1.0, 0.01, seed=7)
            Uh = project(U, R)
            for e in range(E):
                jobs.append((Y, U, 100 * e, 1500, 50, 150)); meta.append(("closed_sr1", 2.0, d, 1.0, N))
                jobs.append((Y, Uh, 100 * e, 1500, 50, 150)); meta.append(("iv_gru_sr1", 2.0, d, 1.0, N))
    with Pool(22) as p: res = p.map(c.gru_member, jobs)
    by = {}
    for m, (eff, loss) in zip(meta, res): by.setdefault(m, []).append((eff, loss))
    rows = []
    for m, lst in by.items():
        Ef = np.stack([x[0] for x in lst]); mean = Ef.mean(0); sc = np.sqrt(np.mean(c.EFF_TRUE ** 2))
        rows.append(dict(regime=m[0], kp=m[1], d=m[2], sigma_r=m[3], N=m[4], gru=c.rel_err(mean), members=[c.rel_err(x) for x in Ef],
                         spread=float(np.sqrt(np.mean(Ef.var(0))) / sc), eff_mean=mean[::10].round(3).tolist(), loss=float(np.mean([x[1] for x in lst]))))
        print(rows[-1]["regime"], m[2], m[4], "err %.3f spread %.3f eff@50 %.2f (true %.2f)" % (rows[-1]["gru"], rows[-1]["spread"], mean[50], c.EFF_TRUE[50]), flush=True)
    json.dump(rows, open("c1b_results.json", "w"), indent=1)
