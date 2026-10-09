"""C4c — log-only certificate on the house: gamma^2 = share of the heater's (slow-band) variance explained by LOGGED EXOGENOUS signals
(measured drivers + set-point), by cross-validated regression (leave-one-day-out).  Law: twin step ≈ gamma^2 * G + (1 - gamma^2) * (-1/C)."""
import sys, json, numpy as np
from scipy.signal import lfilter
sys.path.insert(0, '../paperB'); import twin_load as tl
from c4_house_lintwin import bank, DRV

def lowpass(x, tau=36):
    a = np.exp(-1.0 / tau); return lfilter([1 - a], [1, -a], x - x[0]) + x[0]

def gamma2(house, phases=('const', 'reinit'), band=None, ridge=1e-2):
    d = tl.load(house); Xs, Us, days = [], [], []
    for i, ph in enumerate(phases):
        s = d[d.phase == ph]
        X = np.column_stack([bank(s[k].values) for k in DRV] + [np.full(len(s), float(i))])
        u = s.P_liv.values
        if band: X = np.column_stack([lowpass(c, band) for c in X.T]); u = lowpass(u, band)
        Xs.append(X[300:]); Us.append(u[300:]); days.append((np.arange(len(s))[300:] // 144) + 100 * i)
    X, u, g = np.vstack(Xs), np.concatenate(Us), np.concatenate(days)
    pred = np.empty_like(u)
    for dd in np.unique(g):
        tr, te = g != dd, g == dd
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X[tr] - mu) / sd
        th = np.linalg.solve(Z.T @ Z + ridge * tr.sum() * np.eye(Z.shape[1]), Z.T @ (u[tr] - u[tr].mean()))
        pred[te] = (X[te] - mu) / sd @ th + u[tr].mean()
    return 1 - np.var(u - pred) / np.var(u)

if __name__ == "__main__":
    G24, C24 = 2.1, -0.8      # plant 500-W step at 24 h (designed-data fits) and -1/C step (c4_house_invctrl)
    tw = json.load(open('c4_house_lintwin.json'))
    for house in ['O5', 'N2']:
        for band in (None, 36, 144):
            g2 = gamma2(house, band=band)
            print(f"{house} band={band}: gamma2(CV) {g2:.2f}  -> law predicts 24-h step {g2*G24 + (1-g2)*C24:+.2f}   "
                  f"twin learned {tw[f'{house} const+reinit drivers=True ridge=0.0001'][143]:+.2f} / {tw[f'{house} const+reinit drivers=True ridge=0.01'][143]:+.2f}")
