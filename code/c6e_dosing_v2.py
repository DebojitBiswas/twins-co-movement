#!/usr/bin/env python3
"""
C6e — correction-bolus scenario v2 (replaces c6d dosing; stats M7).  Steady hyperglycaemia: a constant carbohydrate drip is tuned by slow
integral control over 8 h to hold BG ≈ 180 mg/dL at basal insulin, then HELD CONSTANT.  Truth: the effect of a 1-U bolus from that state under
the same drip (so the reference calculator is exact by construction).  Each twin's calculator doses 60 / |twin 4-h effect| U (cap 10 U; no dose
if non-negative; twin effects from c6d_results.json, meal seed 0, A=0 (operational) and A=0.5 IV).  Outcomes: BG0, BG at 4 h, min over 8 h,
hypoglycaemia < 70 / < 54 mg/dL; exact tests vs the true-effect calculator.
"""
import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from simglucose.patient.t1dpatient import T1DPatient, Action
import c6_pancreas as P

H4 = 47


def steady_state(name):
    pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=bas))
    drip = 0.0
    for t in range(8 * 60):
        drip = float(np.clip(drip + 0.002 * (180.0 - pat.observation.Gsub), 0.0, 1.5)); pat.step(Action(CHO=drip, insulin=bas))
    return pat, bas, drip


def run_from_state(name, dose):
    pat, bas, drip = steady_state(name); bg0 = pat.observation.Gsub; bgs = []
    for k in range(96):
        for j in range(P.DT): pat.step(Action(CHO=drip, insulin=bas + (dose if (k == 0 and j == 0) else 0.0)))
        bgs.append(pat.observation.Gsub)
    return bg0, np.array(bgs)


def truth(name):
    b0, with1 = run_from_state(name, 1.0); _, without = run_from_state(name, 0.0)
    return dict(patient=name, bg0=float(b0), effect=(with1 - without).tolist())


def dosing(a):
    name, label, eff4 = a
    dose = 0.0 if eff4 >= 0 else float(min(10.0, 60.0 / (-eff4)))
    try:
        bg0, bgs = run_from_state(name, dose)
        return dict(patient=name, label=label, eff4=float(eff4), dose=dose, bg0=float(bg0), bg4h=float(bgs[H4]), bgmin=float(bgs.min()), hypo=bool(bgs.min() < 70), severe=bool(bgs.min() < 54))
    except Exception as e: return dict(patient=name, label=label, error=str(e))


if __name__ == '__main__':
    ctrl = json.load(open('c6d_control.json')); passed = sorted(c['patient'] for c in ctrl if 'error' not in c and c['rel_err_4h'] < 0.5)
    rows = [r for r in json.load(open('c6d_results.json')) if 'error' not in r and r['hypo'] < 0.5 and r['meal_seed'] == 0 and r['patient'] in passed]
    with Pool(6) as p:
        tr = p.map(truth, passed); T = {t['patient']: t for t in tr}
        print(f"steady state: BG0 {np.mean([t['bg0'] for t in tr]):.0f} (range {min(t['bg0'] for t in tr):.0f}–{max(t['bg0'] for t in tr):.0f}); true 1-U effect at 4 h under drip {np.mean([t['effect'][H4] for t in tr]):.1f} mg/dL", flush=True)
        jobs = [(n, 'true effect (at operating point)', T[n]['effect'][H4]) for n in passed]
        for r in rows:
            jobs.append((r['patient'], f"operational twin d={r['d']}" if r['A'] == 0 else f"IV twin d={r['d']}", (r['lin'] if r['A'] == 0 else r['iv'])[H4]))
        dos = p.map(dosing, jobs)
    json.dump(dict(truth=tr, dosing=dos), open('c6e_dosing_v2.json', 'w'))
    from scipy.stats import fisher_exact
    ref = [x for x in dos if x['label'].startswith('true') and 'error' not in x]; sref = sum(x['severe'] for x in ref)
    for lab in sorted({x['label'] for x in dos}):
        S = [x for x in dos if x['label'] == lab and 'error' not in x]; sv = sum(x['severe'] for x in S)
        pv = fisher_exact([[sv, len(S) - sv], [sref, len(ref) - sref]], alternative='greater')[1]
        print(f"DOSING {lab:34s} n={len(S)}: BG0 {np.mean([x['bg0'] for x in S]):.0f}, dose {np.mean([x['dose'] for x in S]):.1f} U, BG@4h {np.mean([x['bg4h'] for x in S]):.0f}, min {np.mean([x['bgmin'] for x in S]):.0f}, "
              f"hypo {sum(x['hypo'] for x in S)}/{len(S)}, severe {sv}/{len(S)} (Fisher vs reference p={pv:.3f}), no-dose {sum(x['dose']==0 for x in S)}/{len(S)}", flush=True)
