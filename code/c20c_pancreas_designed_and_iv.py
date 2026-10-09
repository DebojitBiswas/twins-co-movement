#!/usr/bin/env python3
"""
C20c — round-4 statistics referee (MAJOR 4c, m6): (i) ARX(6/12/24) on the DESIGNED (open-loop) pancreas logs of the capability control
(c6d.designed_log, seed 0, reproducible), to separate model-class failure from feedback bias in the reactive loop; (ii) regenerated A=0.5
additive-dither logs (meal seed 0, stable seed as in c20b) with the linear and the two-stage IV twin, so that the dosing scenario can be
rerun entirely on saved logs (c6f).  Saves logs to c6d_logs/.  Output c20c_pancreas.json.
"""
import os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import c6_pancreas as P
import c20_pancreas_arx as A20
import c6d_pancreas_v1 as C6D

H4 = 47


def fasting_g0(name):
    from simglucose.patient.t1dpatient import T1DPatient, Action
    pat = T1DPatient.withName(name); b = P.basal_Umin(name)
    for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=b))
    return pat.observation.Gsub


def designed(name):
    try:
        G, U = C6D.designed_log(name); os.makedirs('c6d_logs', exist_ok=True); np.savez_compressed(f"c6d_logs/{name.replace('#', '')}_designed_s0.npz", G=G, U=U)
        tru = P.true_bolus_effect(name); g0 = fasting_g0(name); bas = P.basal_Umin(name) * P.DT
        row = dict(patient=name, kind='designed', true4=float(tru[H4]), lin4=float(P.linear_twin(G, U)[H4]))
        for p in (6, 12, 24):
            e = A20.arx_effect(A20.arx_fit(G, U, p), g0, bas); row[f'arx{p}_4h'] = float(e[H4]) if np.isfinite(e[H4]) and abs(e[H4]) < 1e3 else float('nan')
        print(f"{name} designed: true {row['true4']:.1f} lin {row['lin4']:.1f} ARX6 {row['arx6_4h']:.1f} ARX12 {row['arx12_4h']:.1f} ARX24 {row['arx24_4h']:.1f}", flush=True); return row
    except Exception as e:
        print(f'FAILED designed {name}: {e}', flush=True); return dict(patient=name, kind='designed', error=str(e))


def dither(a):
    name, d = a; ms = 0
    try:
        seed = ms * 1000 + int(d * 10) * 10 + 5 + (sum(map(ord, name)) % 97)
        G, U, R, M = P.run_closed_loop(name, 14, d, 0.5, seed=seed); np.savez_compressed(f"c6d_logs/{name.replace('#', '')}_d{d}_A0.5_ms{ms}.npz", G=G, U=U, R=R, M=M)
        tru = P.true_bolus_effect(name); lin = P.linear_twin(G, U); iv = P.iv_twin(G, U, R)
        row = dict(patient=name, kind='dither', d=d, hypo=float(np.mean(G < 70)), true4=float(tru[H4]), lin4=float(lin[H4]), iv4=float(iv[H4]))
        print(f"{name} d={d} A=0.5: true {row['true4']:.1f} lin {row['lin4']:.1f} IV {row['iv4']:.1f}", flush=True); return row
    except Exception as e:
        print(f'FAILED dither {name} d={d}: {e}', flush=True); return dict(patient=name, kind='dither', d=d, error=str(e))


if __name__ == '__main__':
    ctrl = json.load(open('c6d_control.json')); names = [c['patient'] for c in ctrl if 'rel_err_4h' in c and c['rel_err_4h'] < 0.5]
    with Pool(int(os.environ.get('WORKERS', 10))) as p:
        rows = p.map(designed, names) + p.map(dither, [(n, d) for n in names for d in (0.0, 0.7)])
    json.dump(rows, open('c20c_pancreas.json', 'w'), indent=1)
    S = [x for x in rows if x['kind'] == 'designed' and 'error' not in x]
    for k in ('lin4', 'arx6_4h', 'arx12_4h', 'arx24_4h'):
        v = np.array([x[k] / x['true4'] for x in S]); f = np.isfinite(v)
        print(f"DESIGNED {k:9s}: learned/true 4 h median {np.median(v[f]):.2f} (IQR {np.percentile(v[f], 25):.2f}–{np.percentile(v[f], 75):.2f}), positive {np.mean(np.array([x[k] for x in S])[f] > 0):.2f}, n={f.sum()} (NaN {(~f).sum()})")
