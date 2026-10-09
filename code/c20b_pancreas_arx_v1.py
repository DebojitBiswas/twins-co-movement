#!/usr/bin/env python3
"""
C20b — referee round 3 (control MAJOR 5): the direct-method ARX comparator on the v1 pancreas cohort (30 patients, three meal seeds, reactive
d=0 and meal-bolused d=0.7 loops, no dither), not the ten v0 adults.  c6d did not save its logs and seeded them with Python's per-process string
hash, so the logs are regenerated here under the same protocol with a stable seed (sum of character codes) and saved to c6d_logs/; the linear
simulation-error twin is refitted on the SAME regenerated log so the comparison is like for like.  ARX(p) as in c20 (orders 6, 12, 24; free
simulation of a 1-U bolus from the fasting steady state).  Output c20b_pancreas_arx_v1.json and a summary restricted to the 25 patients passing
the capability control (c6d_control.json, rel. 4-h error < 0.5).
"""
import os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import c6_pancreas as P
import c20_pancreas_arx as A20

H4 = 47


def cell(a):
    name, d, ms = a
    try:
        from simglucose.patient.t1dpatient import T1DPatient, Action
        seed = ms * 1000 + int(d * 10) * 10 + (sum(map(ord, name)) % 97)
        G, U, R, M = P.run_closed_loop(name, 14, d, 0.0, seed=seed)
        os.makedirs('c6d_logs', exist_ok=True); np.savez_compressed(f"c6d_logs/{name.replace('#', '')}_d{d}_ms{ms}.npz", G=G, U=U, R=R, M=M)
        tru = P.true_bolus_effect(name); lin = P.linear_twin(G, U); bas = P.basal_Umin(name) * P.DT
        pat = T1DPatient.withName(name); b = P.basal_Umin(name)
        for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=b))
        g0 = pat.observation.Gsub
        row = dict(patient=name, d=d, meal_seed=ms, hypo=float(np.mean(G < 70)), true4=float(tru[H4]), lin4=float(lin[H4]), lin2=float(lin[23]), true2=float(tru[23]))
        for p in (6, 12, 24):
            e = A20.arx_effect(A20.arx_fit(G, U, p), g0, bas)
            row[f'arx{p}_4h'] = float(e[H4]) if np.isfinite(e[H4]) and abs(e[H4]) < 1e3 else float('nan')
            row[f'arx{p}_2h'] = float(e[23]) if np.isfinite(e[23]) and abs(e[23]) < 1e3 else float('nan')
        print(f"{name} d={d} ms={ms}: true 4 h {row['true4']:.1f} lin {row['lin4']:.1f} ARX6 {row['arx6_4h']:.1f} ARX12 {row['arx12_4h']:.1f} ARX24 {row['arx24_4h']:.1f}", flush=True)
        return row
    except Exception as e:
        print(f'FAILED {name} d={d} ms={ms}: {e}', flush=True); return dict(patient=name, d=d, meal_seed=ms, error=str(e))


if __name__ == '__main__':
    ctrl = json.load(open('c6d_control.json')); names = [c['patient'] for c in ctrl if 'rel_err_4h' in c and c['rel_err_4h'] < 0.5]
    jobs = [(n, d, ms) for n in names for d in (0.0, 0.7) for ms in range(3)]
    with Pool(int(os.environ.get('WORKERS', 10))) as p: rows = p.map(cell, jobs)
    json.dump(rows, open('c20b_pancreas_arx_v1.json', 'w'), indent=1)
    for d in (0.0, 0.7):
        S = [x for x in rows if x['d'] == d and 'error' not in x and x['hypo'] < 0.5]
        for k in ('lin4', 'arx6_4h', 'arx12_4h', 'arx24_4h'):
            v = np.array([x[k] / x['true4'] for x in S]); v = v[np.isfinite(v)]
            print(f"d={d} {k:9s}: learned/true 4 h median {np.median(v):.2f} (IQR {np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f}), positive {np.mean(np.array([x[k] for x in S]) > 0):.2f}, n={len(v)}")
