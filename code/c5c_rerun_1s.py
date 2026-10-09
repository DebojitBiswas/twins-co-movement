"""C5c — one-step model families with the fixed recipe (6,000 steps, input-noise 0.1), all regimes of c5, 3 seeds.
Validated on open-loop data in c5b (transformer effect err 0.07–0.35 at 6,000 steps with noise 0.05–0.2)."""
import os
os.environ["TF_NOISE"] = "0.1"
import json, numpy as np
from multiprocessing import Pool
import c1_synthetic as c
import c5_models_decision as m
from c1b_gru_controls import openloop

if __name__ == "__main__":
    N = 200000; regimes = {"open": openloop(N, 3)}
    for d in (0.0, 0.8):
        for sr in (0.0, 1.0):
            Y, U, R, Q = c.simulate(N, 2.0, d, sr, 0.01, seed=7); regimes[f"cl_d{d}_sr{sr}"] = (Y, U)
    jobs, meta = [], []
    for reg, (Y, U) in regimes.items():
        for kind in ("gru_1s", "tf_1s"):
            for s in range(3): jobs.append((kind, Y, U, s, 6000)); meta.append((reg, kind, s))
    with Pool(22) as p: res = p.map(m.train_eval, jobs)
    rows = []
    for (reg, kind, s), r in zip(meta, res):
        rows.append(dict(regime=reg, model=kind, seed=s, eff_err=c.rel_err(r["eff"]), track=r["track"], model_track=r["model_track"],
                         loss=r["loss"], eff=r["eff"][::10].round(3).tolist(), plan=r["plan"][::5].round(3).tolist()))
        print(f"{reg:16s} {kind:8s} s{s}: effect err {rows[-1]['eff_err']:.2f}  plan→true {r['track']:.2f} K (model thinks {r['model_track']:.2f})", flush=True)
    json.dump(rows, open("c5c_results.json", "w"), indent=1)
