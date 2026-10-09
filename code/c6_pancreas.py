#!/usr/bin/env python3
"""
C6 — artificial pancreas: does a digital twin trained on closed-loop insulin-pump logs learn that insulin RAISES glucose?

Plant: simglucose T1DPatient (UVA/Padova-type, Dalla Man 2007), adults #001–#010, 1-min integration, 5-min control/CGM.
Controller (hybrid closed loop): insulin rate = basal · (1 + kp·(CGM − 120)/100 + ki·Σ(CGM − 120)/100 + kd·ΔCGM/100) · (1 + dither),
  clipped to [0, 5·basal]; plus a meal bolus of a fraction d of CHO/CR at meal onset (d = 0: fully reactive, unannounced meals).
Hidden disturbance: meals (3/day, random time ±1 h and size, plus random snacks) — NOT in the log.
Dither (repair): additive basal offset r·basal, r ~ U(−A, A), held 30–120 min, logged (v0 used a multiplier on the whole rate: not exogenous).
Log: CGM (Gaussian noise 5 mg/dL, 1-mg/dL resolution) and insulin delivered per 5 min.
Query: effect of a 1-U bolus on blood glucose over 6 h from fasting steady state (true: simulator with/without the bolus).
Twins: linear simulation-error twin (filter bank on insulin; the c4 recipe).  GRU / Transformer twins: see c6b (later).
"""
import sys, json, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from simglucose.patient.t1dpatient import T1DPatient, Action
from scipy.signal import lfilter
import pkg_resources

QUEST = pd.read_csv(pkg_resources.resource_filename("simglucose", "params/Quest.csv"))
PARAMS = pd.read_csv(pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv"))
DT = 5                                   # min per control step
H6 = 72                                  # 6 h in 5-min steps


def basal_Umin(name):
    p = PARAMS[PARAMS.Name == name].iloc[0]; return p.u2ss * p.BW / 6000.0


def meal_plan(days, rng):
    """list of (minute, grams)"""
    meals = []
    for dday in range(days):
        for hh, g in ((7.5, 50), (12.5, 70), (19, 80)):
            meals.append((int((dday * 24 + hh + rng.uniform(-1, 1)) * 60), max(10.0, rng.normal(g, 15))))
        for _ in range(rng.poisson(1.0)):
            meals.append((int((dday * 24 + rng.uniform(9, 22)) * 60), rng.uniform(10, 30)))
    return sorted(meals)


def run_closed_loop(name, days, d, A, seed, kp=1.0, ki=0.01, kd=2.0):
    rng = np.random.default_rng(seed)
    pat = T1DPatient.withName(name); bas = basal_Umin(name)
    CR = QUEST[QUEST.Name == name].CR.values[0]
    meals = meal_plan(days, rng); mi = 0
    n = days * 24 * 60 // DT
    G, U, Rr, M = np.empty(n), np.empty(n), np.empty(n), np.zeros(n)
    integ, prev = 0.0, None; r = 0.0; hold = 0
    for k in range(n):
        bg = pat.observation.Gsub
        cgm = np.round(bg + 5.0 * rng.standard_normal())
        if hold <= 0: r = rng.uniform(-A, A) if A > 0 else 0.0; hold = rng.integers(6, 25)
        hold -= 1
        e = (cgm - 120.0) / 100.0; de = 0.0 if prev is None else (cgm - prev) / 100.0; prev = cgm
        mult = 1 + kp * e + ki * integ + kd * de
        rate = float(np.clip(bas * mult + bas * r, 0.0, 5 * bas))          # additive basal dither, independent of the controller
        if 0 < mult + r < 5: integ += e
        integ = float(np.clip(integ, -100, 300))
        tmin = k * DT; cho_total = 0.0; bolus = 0.0
        while mi < len(meals) and meals[mi][0] < tmin + DT:
            cho_total += meals[mi][1]; bolus += d * meals[mi][1] / CR; mi += 1
        M[k] = cho_total
        for j in range(DT):
            ins = rate + (bolus if j == 0 else 0.0)          # bolus delivered in the first minute (U/min for 1 min = U)
            pat.step(Action(CHO=(cho_total / 1.0 if j == 0 else 0.0), insulin=ins))
        G[k], U[k], Rr[k] = cgm, rate * DT + bolus, r
    return G, U, Rr, M


def true_bolus_effect(name, units=1.0, hours_pre=6):
    """BG difference (with − without) after a bolus at fasting steady state, 6 h, 5-min grid."""
    out = []
    for b in (units, 0.0):
        pat = T1DPatient.withName(name); bas = basal_Umin(name)
        for _ in range(hours_pre * 60): pat.step(Action(CHO=0, insulin=bas))
        g = []
        for k in range(H6):
            g.append(pat.observation.Gsub)
            for j in range(DT): pat.step(Action(CHO=0, insulin=bas + (b if (k == 0 and j == 0) else 0.0)))
        out.append(np.array(g))
    return out[0] - out[1]


TAUS = [1, 3, 6, 12, 24, 48]
def bank(x):
    cols = [x]
    for tau in TAUS:
        a = np.exp(-1.0 / tau); cols.append(lfilter([1 - a], [1, -a], x))
    return np.column_stack(cols)


def linear_twin(G, U, ridge=1e-4):
    """simulation-error-limit linear twin: G_t = c + sum_k w_k bank_k(U)_t (no history of G; meals unlogged)."""
    X = bank(U - U.mean()); y = G - G.mean()
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    th = np.linalg.solve(Z.T @ Z + ridge * len(y) * np.eye(Z.shape[1]), Z.T @ y)
    w = th / sd
    imp = np.zeros(H6); imp[0] = 1.0
    return bank(imp) @ w


def iv_twin(G, U, R, ridge=1e-4, M=48):
    """two-stage: U projected on lagged dither r (bank + lags), then the linear twin on U_hat."""
    Z = np.column_stack([np.r_[np.zeros(i), R[:len(R) - i]] for i in range(M)] + [np.ones(len(R))])
    Uh = Z @ np.linalg.lstsq(Z, U, rcond=None)[0]
    return linear_twin(G, Uh, ridge)


def cell(args):
    name, d, A = args
    tru = true_bolus_effect(name)
    G, U, R, M = run_closed_loop(name, 14, d, A, seed=(abs(hash(name)) % 1000) * 100 + int(d * 10) * 10 + int(A * 10))
    lin = linear_twin(G, U)
    row = dict(patient=name, d=d, A=A, true=tru.tolist(), lin=lin.tolist(),
               bg_mean=float(G.mean()), bg_sd=float(G.std()), tir=float(np.mean((G >= 70) & (G <= 180))), hypo=float(np.mean(G < 70)))
    if A > 0: row["iv"] = iv_twin(G, U, R).tolist()
    np.savez_compressed(f"c6_logs/{name.replace('#','')}_d{d}_A{A}.npz", G=G, U=U, R=R, M=M)
    at = lambda v, h: round(float(np.asarray(v)[int(h * 12) - 1]), 1)
    print(f"{name} d={d} A={A}: BG {G.mean():.0f}±{G.std():.0f} TIR {row['tir']:.2f} hypo {row['hypo']:.3f} | 1-U effect @1/2/4/6 h "
          f"true {[at(tru,h) for h in (1,2,4,6)]} twin {[at(lin,h) for h in (1,2,4,6)]}"
          + (f" IV {[at(row['iv'],h) for h in (1,2,4,6)]}" if A > 0 else ""), flush=True)
    return row


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs("c6_logs", exist_ok=True)
    names = [f"adult#{i:03d}" for i in range(1, 11)] if len(sys.argv) < 2 else sys.argv[1].split(",")
    jobs = [(n, d, A) for n in names for d in (0.0, 0.7) for A in (0.0, 0.5)]
    with Pool(int(os.environ.get("WORKERS", 10))) as p: rows = p.map(cell, jobs)
    json.dump(rows, open("c6_results.json", "w"))
