"""C4b — the linear simulation-error-limit twin on real data: y = sum over inputs of (filter bank of input) + phase constant,
fitted by least squares on O5 regulated data (heater + measured drivers).  Its heater step response is what an infinitely flexible
linear output-error simulator converges to (the partial Wiener solution).  Compare: -1/C (c4_house_invctrl), GRU/NODE, truth."""
import sys, json, numpy as np
from scipy.signal import lfilter
sys.path.insert(0, '../paperB'); import twin_load as tl
TAUS = [1, 3, 10, 30, 100, 300]          # 10-min steps
DRV = ['T_out', 'S_south', 'S_west', 'T_cellar', 'T_attic', 'T_sup', 'V_sup', 'P_other']

def bank(x):
    cols = [x]
    for tau in TAUS:
        a = np.exp(-1.0 / tau); cols.append(lfilter([1 - a], [1, -a], x - x[0]) + x[0])
    return np.column_stack(cols)

def design(s, sp_dummy):
    X = [bank(s.P_liv.values)] + [bank(s[k].values) for k in DRV]
    return np.column_stack(X + [np.ones(len(s)), np.full(len(s), sp_dummy)])

def fit(house, phases, drivers=True, ridge=1e-3):
    d = tl.load(house); Xs, Ys = [], []
    for i, ph in enumerate(phases):
        s = d[d.phase == ph]; X = design(s, float(i))
        if not drivers: X = np.column_stack([X[:, :len(TAUS) + 1], X[:, -2:]])
        Xs.append(X[300:]); Ys.append(s.T_liv.values[300:])        # drop filter warm-up
    X, Y = np.vstack(Xs), np.concatenate(Ys)
    mu, sd = X.mean(0), X.std(0) + 1e-9; sd[-2:] = 1; mu[-2:] = 0
    Z = (X - mu) / sd
    th = np.linalg.solve(Z.T @ Z + ridge * len(Y) * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean()))
    w = th[:len(TAUS) + 1] / sd[:len(TAUS) + 1]
    step = bank(np.r_[0.0, np.full(144, 500.0)])[1:]     # filters start at rest
    resp = step @ w
    r2 = 1 - np.var(Y - Y.mean() - Z @ th) / np.var(Y)
    return resp, r2

if __name__ == "__main__":
    out = {}
    for house in ['O5', 'N2']:
        for drivers in (True, False):
            for ph in (['const', 'reinit'], ['rolbs']):
                for ridge in (1e-4, 1e-2):
                    resp, r2 = fit(house, ph, drivers, ridge)
                    key = f"{house} {'+'.join(ph)} drivers={drivers} ridge={ridge}"
                    out[key] = resp.tolist()
                    print(f"{key:50s} R2 {r2:.3f}  step(500W)@1/3/6/12/24h {resp[[5,17,35,71,143]].round(2)}")
    json.dump(out, open('c4_house_lintwin.json', 'w'), indent=1)
