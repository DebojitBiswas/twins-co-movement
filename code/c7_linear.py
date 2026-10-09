"""C7 — linear simulation-error twin (filter bank on u, outdoor temperature, solar; ridge) and its two-stage IV version (u projected on
a filter bank of the logged set-point r), on each BOPTEST log.  Effect of +0.3 modulation over 24 h vs the exact truth (c7_data/truth.json)."""
import json, numpy as np
from scipy.signal import lfilter

TAUS = [1, 4, 12, 24, 48, 96, 192]          # 15-min steps: 15 min … 2 days
H = 96

def bank(x):
    cols = [x]
    for tau in TAUS:
        a = np.exp(-1.0 / tau); cols.append(lfilter([1 - a], [1, -a], x - x[0]) + x[0])
    return np.column_stack(cols)

def fit(y, U, X, ridge):
    Z = np.column_stack([bank(U)] + [bank(x) for x in X.T])[200:]; yy = y[200:]
    mu, sd = Z.mean(0), Z.std(0) + 1e-9; Zs = (Z - mu) / sd
    th = np.linalg.solve(Zs.T @ Zs + ridge * len(yy) * np.eye(Zs.shape[1]), Zs.T @ (yy - yy.mean()))
    w = (th / sd)[:len(TAUS) + 1]
    return bank(np.r_[0.0, np.full(H, 0.3)])[1:] @ w            # response to a +0.3 step from rest

def project(U, R):
    Z = np.column_stack([bank(R), np.ones(len(R))])
    return Z @ np.linalg.lstsq(Z, U, rcond=None)[0]

if __name__ == "__main__":
    tr = json.load(open('c7_data/truth.json')); T = np.mean([t['effect'] for t in tr], 0)
    idx = [3, 11, 23, 47, 95]; sc = np.sqrt(np.mean(T ** 2)); out = {}
    print('truth           ', T[idx].round(2))
    for name in ['designed_A0.0_s0', 'designed_A0.0_s1', 'operational_A0.0_s0', 'dither_A0.5_s0', 'dither_A1.0_s0', 'dither_A2.0_s0']:
        L = np.load(f'c7_data/log_{name}.npy'); y, U, X, R = L[:, 0], L[:, 1], L[:, 2:4], L[:, 4]
        for ridge in (1e-4, 1e-2):
            e = fit(y, U, X, ridge); err = np.sqrt(np.mean((e - T) ** 2)) / sc
            out[f'{name} plain r{ridge}'] = dict(eff=e.tolist(), err=float(err))
            print(f'{name:22s} plain ridge {ridge:g}: {e[idx].round(2)}  rel err {err:.2f}')
            if name.startswith('dither') or name.startswith('operational'):
                e = fit(y, project(U, R), X, ridge); err = np.sqrt(np.mean((e - T) ** 2)) / sc
                out[f'{name} iv r{ridge}'] = dict(eff=e.tolist(), err=float(err))
                print(f'{name:22s} IV    ridge {ridge:g}: {e[idx].round(2)}  rel err {err:.2f}')
    json.dump(out, open('c7_linear.json', 'w'), indent=1)
