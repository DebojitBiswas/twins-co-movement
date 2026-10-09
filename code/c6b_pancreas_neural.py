#!/usr/bin/env python3
"""
C6b — neural twins on the closed-loop pump logs of c6 (c6_logs/*.npz): simulation-error GRU and one-step Transformer (the c5c recipe:
6,000 steps, input-noise 0.1).  Query as in c6: effect of a 1-U bolus over 6 h from fasting steady state (warm-up: constant basal BG and
basal insulin).  Control: the same models trained on an OPEN-LOOP designed log (random basal levels + random boluses, meals present but
unlogged as in the closed loop) — is the model family capable on this patient?
"""
import os, sys, json, glob, numpy as np
from multiprocessing import Pool
import warnings; warnings.filterwarnings("ignore")

H6, WARM = 72, 128


def designed_log(name, days=14, seed=0):
    """open-loop: insulin = basal × level (0, 0.5, 1, 2, held 30–180 min) + random 0–3 U boluses; meals as in c6 (unlogged)."""
    import c6_pancreas as P
    from simglucose.patient.t1dpatient import T1DPatient, Action
    rng = np.random.default_rng(seed); pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    meals = P.meal_plan(days, rng); mi = 0; n = days * 24 * 60 // P.DT
    G, U = np.empty(n), np.empty(n); lvl, hold = 1.0, 0
    for k in range(n):
        if hold <= 0: lvl = rng.choice([0.0, 0.5, 1.0, 2.0]); hold = rng.integers(6, 37)
        hold -= 1
        bolus = rng.uniform(0, 3) if rng.random() < 1 / 36 else 0.0
        tmin = k * P.DT; cho = 0.0
        while mi < len(meals) and meals[mi][0] < tmin + P.DT: cho += meals[mi][1]; mi += 1
        G[k] = np.round(pat.observation.Gsub + 5.0 * rng.standard_normal())
        for j in range(P.DT): pat.step(Action(CHO=(cho if j == 0 else 0.0), insulin=bas * lvl + (bolus if j == 0 else 0.0)))
        U[k] = bas * lvl * P.DT + bolus
    return G, U


def fasting_bg(name):
    import c6_pancreas as P
    from simglucose.patient.t1dpatient import T1DPatient, Action
    pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=bas))
    return pat.observation.Gsub, bas * P.DT


def job(a):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    kind, name, tag, G, U, seed = a
    g0, u0 = fasting_bg(name)
    os.environ["TF_NOISE"] = "0.1"
    import c5_models_decision as M
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    my, sy, mu, su = G.mean(), G.std() + 1e-6, U.mean(), U.std() + 1e-6
    yn = torch.tensor((G - my) / sy, dtype=torch.float32); un = torch.tensor((U - mu) / su, dtype=torch.float32)
    m = M.build(kind, my, sy, mu, su); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    steps = int(os.environ.get("GRU_STEPS", 1500)) if kind == "gru_sim" else 6000
    for it in range(steps):
        if kind == "gru_sim":
            L = 200; idx = rng.integers(0, len(G) - L - 1, B)
            yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
            pred = m.roll(yb[:, :50], ub, 50); loss = ((pred[:, 49:] - yb[:, 50:]) ** 2).mean()
        else:
            L = WARM; idx = rng.integers(0, len(G) - L - 1, B)
            yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
            pred = m.step_all(yb[:, :L] + 0.1 * torch.randn_like(yb[:, :L]), ub); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        y0 = torch.full((2, WARM), (g0 - my) / sy)
        useq = np.full((2, WARM + H6), u0); useq[0, WARM] += 1.0
        out = m.roll(y0, (torch.tensor(useq, dtype=torch.float32) - mu) / su, WARM).numpy() * sy + my
    eff = out[0, WARM - 1:WARM - 1 + H6] - out[1, WARM - 1:WARM - 1 + H6]
    return dict(model=kind, patient=name, tag=tag, seed=seed, eff=eff.round(3).tolist(), loss=float(loss.item()))


if __name__ == "__main__" and os.environ.get("CONTROL_ONLY"):
    rows = []; jobs = []
    for name in [f"adult#{i:03d}" for i in range(1, 6)]:
        G, U = designed_log(name)
        for s in (0, 1): jobs.append(("gru_sim", name, "designed", G, U, s))
    with Pool(10) as p: rows = p.map(job, jobs)
    true = {r["patient"]: r["true"] for r in json.load(open("c6_results.json")) if r["d"] == 0.0 and r["A"] == 0.0}
    for r in rows:
        e, t = np.array(r["eff"]), np.array(true[r["patient"]])
        print(f"{r['patient']} steps={os.environ.get('GRU_STEPS')} s{r['seed']}: @1/2/4/6 h {e[[11,23,47,71]].round(1)} true {t[[11,23,47,71]].round(1)}", flush=True)
    json.dump(rows, open(f"c6b_gru_control_{os.environ.get('GRU_STEPS')}.json", "w"))
    sys.exit(0)

if __name__ == "__main__":
    names = [f"adult#{i:03d}" for i in range(1, 11)]
    jobs = []
    for name in names:
        for tag in ("d0.0_A0.0", "d0.7_A0.0"):
            z = np.load(f"c6_logs/{name.replace('#', '')}_{tag}.npz")
            for kind in ("gru_sim", "tf_1s"):
                for s in (0, 1): jobs.append((kind, name, tag, z["G"], z["U"], s))
    for name in names[:5]:                                   # designed-data control, 5 patients
        G, U = designed_log(name)
        for kind in ("gru_sim", "tf_1s"):
            for s in (0, 1): jobs.append((kind, name, "designed", G, U, s))
    with Pool(22) as p: rows = p.map(job, jobs)
    true = {r["patient"]: r["true"] for r in json.load(open("c6_results.json")) if r["d"] == 0.0 and r["A"] == 0.0}
    for r in rows:
        e, t = np.array(r["eff"]), np.array(true[r["patient"]])
        print(f"{r['patient']} {r['tag']:10s} {r['model']:8s} s{r['seed']}: 1-U effect @1/2/4/6 h {e[[11,23,47,71]].round(1)}  true {t[[11,23,47,71]].round(1)}", flush=True)
    json.dump(rows, open("c6b_results.json", "w"))
