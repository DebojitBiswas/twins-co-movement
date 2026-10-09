#!/usr/bin/env python3
"""
C20 — referee (control MAJOR 8): a classical direct-method comparator on the pancreas pump logs.  ARX(p): G_{t+1} on G_{t..t-p+1}, U_{t..t-p+1}, 1
(least squares, p ∈ {6, 12, 24} 5-min steps); the twin's 1-U bolus effect is the difference of two free simulations from the fasting steady state
(constant basal delivery, G0 = fasting glucose from c6_results 'true' protocol) with and without 1 U at t = 0, over 6 h.  Logs: c6_logs (ten adults,
reactive d=0 and meal-bolused d=0.7, A=0), the v1 additive-dither logs A=0.5 for the IV comparison are not needed here.  Compare with the linear
simulation-error twin (c6_results) and the truth.
"""
import json, glob, numpy as np
import c6_pancreas as P

H6 = 72


def arx_fit(G, U, p):
    n = len(G); X = np.column_stack([G[p - 1 - i:n - 1 - i] for i in range(p)] + [U[p - 1 - i:n - 1 - i] for i in range(p)] + [np.ones(n - p)])
    return p, np.linalg.lstsq(X, G[p:], rcond=None)[0]


def arx_effect(m, g0, u0):
    p, th = m; out = []
    for bolus in (1.0, 0.0):
        Gs, Us = [g0] * p, [u0] * p; traj = []
        for t in range(H6):
            Us.append(u0 + (bolus if t == 0 else 0.0))
            reg = np.r_[Gs[-1::-1][:p], Us[-2::-1][:p], 1.0]            # G_{t}..G_{t-p+1}, U_{t}..U_{t-p+1}
            Gs.append(float(reg @ th)); traj.append(Gs[-1])
        out.append(np.array(traj))
    return out[0] - out[1]


if __name__ == '__main__':
    rows = json.load(open('c6_results.json')); res = []
    for r in rows:
        if r['A'] != 0.0 or r['hypo'] >= 0.5: continue
        z = np.load(f"c6_logs/{r['patient'].replace('#', '')}_d{r['d']}_A0.0.npz"); G, U = z['G'], z['U']
        bas = P.basal_Umin(r['patient']) * P.DT
        # fasting glucose: run the patient at basal for 6 h (as in true_bolus_effect)
        from simglucose.patient.t1dpatient import T1DPatient, Action
        pat = T1DPatient.withName(r['patient']); b = P.basal_Umin(r['patient'])
        for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=b))
        g0 = pat.observation.Gsub
        row = dict(patient=r['patient'], d=r['d'], true4=r['true'][47], lin4=r['lin'][47])
        for p in (6, 12, 24):
            e = arx_effect(arx_fit(G, U, p), g0, bas); row[f'arx{p}_4h'] = float(e[47]) if np.isfinite(e[47]) and abs(e[47]) < 1e3 else float('nan')
        res.append(row); print(f"{r['patient']} d={r['d']}: true 4 h {row['true4']:.1f}  lin {row['lin4']:.1f}  ARX6 {row['arx6_4h']:.1f}  ARX12 {row['arx12_4h']:.1f}  ARX24 {row['arx24_4h']:.1f}", flush=True)
    json.dump(res, open('c20_pancreas_arx.json', 'w'), indent=1)
    for d in (0.0, 0.7):
        S = [x for x in res if x['d'] == d]
        for k in ('lin4', 'arx6_4h', 'arx12_4h', 'arx24_4h'):
            v = np.array([x[k] / x['true4'] for x in S]); v = v[np.isfinite(v)]
            print(f"d={d} {k:9s}: learned/true 4 h median {np.median(v):.2f} (IQR {np.percentile(v,25):.2f}–{np.percentile(v,75):.2f}), positive {np.mean(np.array([x[k] for x in S]) > 0):.2f}, n={len(v)}")
