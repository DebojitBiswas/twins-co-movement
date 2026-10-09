#!/usr/bin/env python3
"""
C6c — decision consequence in the pancreas (T4): a correction-bolus calculator derived from each twin.
From a fasting steady state raised to BG0 = 180 mg/dL (glucose infusion ramp), the calculator doses insulin so that the twin predicts a
return to 120 mg/dL at 4 h: dose = 60 / |twin effect of 1 U at 4 h| (capped at 10 U; no dose if the twin's effect is non-negative).
The dose is applied to the TRUE patient; we record BG at 4 h, the minimum BG over 8 h and hypoglycaemia (< 70 mg/dL).
Twins: linear simulation-error twin on operational logs (c6_results: d ∈ {0, 0.7}, A = 0), the same on dithered logs with IV (A = 0.5),
and the designed-data reference (the true effect, i.e. a perfectly informed calculator).  10 virtual adults.
"""
import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from simglucose.patient.t1dpatient import T1DPatient, Action
import c6_pancreas as P

BG0, TARGET, H4, H8 = 180.0, 120.0, 48, 96          # 5-min steps


def raise_bg(name):
    """bring the patient to ≈ BG0 at basal insulin via a slow carbohydrate drip; return the patient (state) at that point."""
    pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=bas))
    drip = 0.0
    for t in range(12 * 60):
        bg = pat.observation.Gsub
        drip = float(np.clip(drip + 0.02 * (BG0 - bg), 0.0, 3.0))              # g/min, slow integral control of the drip
        pat.step(Action(CHO=drip, insulin=bas))
    return pat, bas, drip


def apply_dose(name, dose):
    pat, bas, drip = raise_bg(name); bgs = []
    for k in range(H8):
        for j in range(P.DT): pat.step(Action(CHO=drip, insulin=bas + (dose if (k == 0 and j == 0) else 0.0)))
        bgs.append(pat.observation.Gsub)
    return np.array(bgs)


def job(a):
    name, twins = a
    out = dict(patient=name, bg0=float(raise_bg(name)[0].observation.Gsub), results={})
    for label, eff4 in twins.items():
        dose = 0.0 if eff4 >= 0 else float(min(10.0, (BG0 - TARGET) / (-eff4)))
        bg = apply_dose(name, dose)
        out['results'][label] = dict(eff4=float(eff4), dose=dose, bg4h=float(bg[H4 - 1]), bgmin=float(bg.min()), hypo=bool(bg.min() < 70), severe=bool(bg.min() < 54))
    return out


if __name__ == '__main__':
    rows = json.load(open('c6_results.json')); jobs = []
    for name in sorted({r['patient'] for r in rows}):
        R = {(r['d'], r['A']): r for r in rows if r['patient'] == name}
        tw = {'true (designed-data reference)': R[(0.0, 0.0)]['true'][H4 - 1]}
        for d in (0.0, 0.7):
            if R[(d, 0.0)]['hypo'] < 0.5: tw[f'operational twin d={d}'] = R[(d, 0.0)]['lin'][H4 - 1]
            if R[(d, 0.5)]['hypo'] < 0.5: tw[f'IV twin (dither) d={d}'] = R[(d, 0.5)]['iv'][H4 - 1]
        jobs.append((name, tw))
    with Pool(5) as p: res = p.map(job, jobs)
    json.dump(res, open('c6c_dosing.json', 'w'), indent=1)
    labels = sorted({k for r in res for k in r['results']})
    for lab in labels:
        rr = [r['results'][lab] for r in res if lab in r['results']]
        print(f"{lab:34s} n={len(rr)}: dose {np.mean([x['dose'] for x in rr]):.1f} U (range {min(x['dose'] for x in rr):.1f}–{max(x['dose'] for x in rr):.1f}), "
              f"BG@4h {np.mean([x['bg4h'] for x in rr]):.0f} mg/dL, min BG {np.mean([x['bgmin'] for x in rr]):.0f}, hypo {sum(x['hypo'] for x in rr)}/{len(rr)}, severe {sum(x['severe'] for x in rr)}/{len(rr)}, no-dose {sum(x['dose']==0 for x in rr)}/{len(rr)}", flush=True)
