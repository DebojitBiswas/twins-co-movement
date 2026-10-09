"""C4 — Twin Houses: does a simulation-trained twin learn the negative inverse thermostat?
Fit the thermostat C (e = setpoint - T  ->  P) on O5's regulated phases, form -1/C, and compare its 500-W step response with the
step responses learned by the GRU / NODE simulators of paper B (B10, trained on the same O5 regulated data), with the empirical
(designed-data) response of N2 (B9) and with the structured model."""
import sys, json, numpy as np
sys.path.insert(0, '../paperB'); import twin_load as tl

def fit_controller(house='O5', p=3, q=3):
    d = tl.load(house); E, U, Yt = [], [], []
    X, T = [], []
    for ph, sp in [('const', 30.0), ('reinit', 25.0)]:
        s = d[d.phase == ph]; e = sp - s.T_liv.values; u = s.P_liv.values
        m = max(p, q)
        for t in range(m, len(u)):
            X.append(np.r_[u[t - p:t][::-1], e[t - q:t + 1][::-1], 1.0, float(ph == 'reinit')]); T.append(u[t])
    X, T = np.array(X), np.array(T)
    th, *_ = np.linalg.lstsq(X, T, rcond=None)
    r2 = 1 - np.var(T - X @ th) / np.var(T)
    a = th[:p]; b = th[p:p + q + 1]            # u_t = sum a_i u_{t-i} + sum_{j=0..q} b_j e_{t-j}
    return a, b, r2

def inv_ctrl_step(a, b, P=500.0, n=144):
    """y such that the controller would produce input step P from error e = -y:  (1 - sum a z^-i) u = (sum b z^-j) e
    => e = (1 - sum a z^-i)/(sum b z^-j) u,  y = -e.  Simulate the inverse filter."""
    u = np.full(n, P); e = np.zeros(n)
    for t in range(n):
        lhs = u[t] - sum(a[i] * (u[t - 1 - i] if t - 1 - i >= 0 else 0.0) for i in range(len(a)))
        lhs -= sum(b[j] * e[t - j] for j in range(1, len(b)) if t - j >= 0)
        e[t] = lhs / b[0]
    return -e

if __name__ == "__main__":
    out = {}
    for p, q in [(1, 1), (3, 3), (6, 6)]:
        a, b, r2 = fit_controller('O5', p, q)
        y = inv_ctrl_step(a, b)
        roots = np.abs(np.roots(b)) if len(b) > 1 else []
        print(f"C(p={p},q={q}) R2 {r2:.2f} sum a {a.sum():.3f} b0 {b[0]:.0f} sum b {b.sum():.0f}  |B roots| {np.round(roots,2)}  "
              f"-1/C step(500W) @1/3/6/12/24h: {y[[5,17,35,71,143]].round(2)}")
        out[f"p{p}q{q}"] = dict(a=a.tolist(), b=b.tolist(), r2=r2, step=y.tolist())
    r = json.load(open('../paperB/expB10_results.json'))
    for k in ['GRU', 'NODE']:
        for s in range(3):
            resp = np.array(r[f'{k}:house:seed{s}']['resp'])
            print(f"{k} seed{s} learned step @1/3/6/12/24h: {resp[[5,17,35,71,143]].round(2)}")
    json.dump(out, open('c4_house_invctrl.json', 'w'), indent=1)
