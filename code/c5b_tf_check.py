"""C5b — fix the transformer baseline on OPEN-LOOP designed data first (steps × input-noise), 2 seeds each."""
import os, json
from multiprocessing import Pool
import c1_synthetic as c
from c1b_gru_controls import openloop


def job(a):
    steps, noise, seed = a
    os.environ["TF_NOISE"] = str(noise)
    import importlib, c5_models_decision as m; importlib.reload(m)
    Y, U = openloop(200000, 3)
    r = m.train_eval(("tf_1s", Y, U, seed, steps))
    return dict(steps=steps, noise=noise, seed=seed, eff_err=c.rel_err(r["eff"]), track=r["track"], model_track=r["model_track"])


if __name__ == "__main__":
    jobs = [(st, nz, s) for st in (1500, 6000) for nz in (0.0, 0.05, 0.2) for s in (0, 1)]
    with Pool(12) as p: rows = p.map(job, jobs)
    for r in rows: print(r, flush=True)
    json.dump(rows, open("c5b_tf_check.json", "w"))
