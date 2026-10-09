"""C7 primary metric for linear twins: simulation RMSE on held-out designed test schedules (days 42–56, seeds 10/11), given weather."""
import json, numpy as np, c7_linear as L
def fitfull(y, U, X, ridge=1e-4, W=200):
    Z = np.column_stack([L.bank(U)] + [L.bank(x) for x in X.T]); mu, sd = Z[W:].mean(0), Z[W:].std(0) + 1e-9
    Zs = (Z - mu) / sd; th = np.linalg.solve(Zs[W:].T @ Zs[W:] + ridge * len(y[W:]) * np.eye(Zs.shape[1]), Zs[W:].T @ (y[W:] - y[W:].mean()))
    return lambda U2, X2: ((np.column_stack([L.bank(U2)] + [L.bank(x) for x in X2.T]) - mu) / sd) @ th + y[W:].mean()
tests = {s: np.load(f'c7_data/test_designed_s{s}.npy') for s in (10, 11)}
out = {}
for s, te in tests.items():
    print(f'test s{s}: persistence-of-mean RMSE {np.sqrt(np.mean((te[200:,0]-te[200:,0].mean())**2)):.2f} K')
for name in ['designed_A0.0_s0', 'designed_A0.0_s1', 'operational_A0.0_s0', 'dither_A0.5_s0', 'dither_A1.0_s0', 'dither_A2.0_s0']:
    tr = np.load(f'c7_data/log_{name}.npy')
    for kind in (['plain', 'iv'] if not name.startswith('designed') else ['plain']):
        U = tr[:, 1] if kind == 'plain' else L.project(tr[:, 1], tr[:, 4])
        for ridge in (1e-4, 1e-2):
            f = fitfull(tr[:, 0], U, tr[:, 2:4], ridge)
            r = [float(np.sqrt(np.mean((f(te[:, 1], te[:, 2:4]) - te[:, 0])[200:] ** 2))) for te in tests.values()]
            out[f'{name}|{kind}|{ridge}'] = r
            print(f'{name:22s} {kind:5s} ridge {ridge:g}: do-RMSE {r[0]:.2f} / {r[1]:.2f} K')
json.dump(out, open('c7_eval_linear.json', 'w'), indent=1)
