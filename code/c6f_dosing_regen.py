#!/usr/bin/env python3
"""
C6f — the c6e dosing scenario rerun with twin effects taken from SAVED, regenerated logs (round-4 referees, provenance): operational twins from
c20b (meal seed 0, A=0; linear simulation-error twin refitted on the regenerated log), IV twins from c20c (A=0.5, meal seed 0).  Scenario, truth
and calculator identical to c6e (imported).  Output c6f_dosing_regen.json.
"""
import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import c6e_dosing_v2 as E

if __name__ == '__main__':
    ctrl = json.load(open('c6d_control.json')); passed = sorted(c['patient'] for c in ctrl if 'error' not in c and c['rel_err_4h'] < 0.5)
    b = [r for r in json.load(open('c20b_pancreas_arx_v1.json')) if 'error' not in r and r['meal_seed'] == 0 and r['hypo'] < 0.5 and r['patient'] in passed]
    c = [r for r in json.load(open('c20c_pancreas.json')) if r['kind'] == 'dither' and 'error' not in r and r['hypo'] < 0.5 and r['patient'] in passed]
    T = {t['patient']: t for t in json.load(open('c6e_dosing_v2.json'))['truth']}
    jobs = [(n, 'true effect (at operating point)', T[n]['effect'][E.H4]) for n in passed]
    jobs += [(r['patient'], f"operational twin d={r['d']}", r['lin4']) for r in b] + [(r['patient'], f"IV twin d={r['d']}", r['iv4']) for r in c]
    with Pool(8) as p: dos = p.map(E.dosing, jobs)
    json.dump(dict(dosing=dos), open('c6f_dosing_regen.json', 'w'))
    from scipy.stats import fisher_exact
    ref = [x for x in dos if x['label'].startswith('true') and 'error' not in x]; sref = sum(x['severe'] for x in ref)
    for lab in sorted({x['label'] for x in dos}):
        S = [x for x in dos if x['label'] == lab and 'error' not in x]; sv = sum(x['severe'] for x in S)
        pv = fisher_exact([[sv, len(S) - sv], [sref, len(ref) - sref]], alternative='greater')[1]
        print(f"DOSING(regen) {lab:34s} n={len(S)}: BG0 {np.mean([x['bg0'] for x in S]):.0f}, dose {np.mean([x['dose'] for x in S]):.1f} U, min {np.mean([x['bgmin'] for x in S]):.0f}, hypo {sum(x['hypo'] for x in S)}/{len(S)}, severe {sv}/{len(S)} (Fisher p={pv:.3f}), no-dose {sum(x['dose']==0 for x in S)}/{len(S)}", flush=True)
