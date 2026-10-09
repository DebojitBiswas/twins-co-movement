"""C5d — referee (control M3): one-step neural families WITHOUT input-noise injection, d = 0 (pure feedback) and d = 0.8, N = 2e5, 10,000 steps,
3 seeds; ARX(20) yardstick on the same logs.  Does the direct method's consistency under pure feedback show up in neural one-step learners?"""
import os
os.environ['TF_NOISE'] = '0.0'
import json, numpy as np
from multiprocessing import Pool
import c1_synthetic as c
import c5_models_decision as m

if __name__ == '__main__':
    N = 200000; regimes = {}
    for d in (0.0, 0.8):
        Y, U, R, Q = c.simulate(N, 2.0, d, 0.0, 0.01, seed=7); regimes[f'cl_d{d}'] = (Y, U)
    jobs, meta = [], []
    for reg, (Y, U) in regimes.items():
        for kind in ('gru_1s', 'tf_1s'):
            for s in range(3): jobs.append((kind, Y, U, s, 10000)); meta.append((reg, kind, s))
    with Pool(int(os.environ.get('WORKERS', 6))) as p: res = p.map(m.train_eval, jobs)
    rows = [dict(regime=reg, model=kind, seed=s, eff_err=c.rel_err(r['eff']), track=r['track'], loss=r['loss']) for (reg, kind, s), r in zip(meta, res)]
    for reg, (Y, U) in regimes.items():
        rows.append(dict(regime=reg, model='arx20', seed=0, eff_err=c.rel_err(c.fit_arx(Y, U, 20)), track=None, loss=None))
    json.dump(rows, open('c5d_onestep_clean.json', 'w'), indent=1)
    for r in rows: print(f"{r['regime']:10s} {r['model']:7s} s{r['seed']}: effect err {r['eff_err']:.2f}" + (f"  plan {r['track']:.2f} K" if r['track'] else ''), flush=True)
