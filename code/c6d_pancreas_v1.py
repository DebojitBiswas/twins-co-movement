#!/usr/bin/env python3
"""
C6d — pancreas v1 (referee stats M6/M7/M8/M13; control M7c): three cohorts (adult, adolescent, child; 30 virtual patients), three meal-schedule
seeds per patient-configuration, loops d ∈ {0 (reactive), 0.7 (meal-bolused)}, additive basal dither A ∈ {0, 0.5}; linear simulation-error twin
with a DESIGNED-DATA CAPABILITY CONTROL (open-loop insulin schedule: basal × {0, 0.5, 1, 2} held 30–180 min + random boluses, meals unlogged);
pre-stated pass criterion: relative error of the twin's 4-h effect on designed data < 0.5 for the patient.  Dosing calculator with a FIXED
protocol: raise glucose to 180 mg/dL by a carbohydrate drip, STOP the drip at dose time, dose = 60 / |twin 4-h effect| (cap 10 U; no dose if the
effect is non-negative), applied to the true patient; outcomes BG at 4 h, minimum over 8 h, hypoglycaemia < 70 and < 54 mg/dL.
Outputs c6d_results.json (one row per patient × loop × dither × meal seed) and c6d_dosing.json.
"""
import os, sys, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from simglucose.patient.t1dpatient import T1DPatient, Action
import c6_pancreas as P

H4 = 47


def designed_log(name, days=14, seed=0):
    rng = np.random.default_rng(seed); pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    meals = P.meal_plan(days, rng); mi = 0; n = days * 24 * 60 // P.DT; G, U = np.empty(n), np.empty(n); lvl, hold = 1.0, 0
    for k in range(n):
        if hold <= 0: lvl = rng.choice([0.0, 0.5, 1.0, 2.0]); hold = rng.integers(6, 37)
        hold -= 1; bolus = rng.uniform(0, 3) if rng.random() < 1 / 36 else 0.0; tmin = k * P.DT; cho = 0.0
        while mi < len(meals) and meals[mi][0] < tmin + P.DT: cho += meals[mi][1]; mi += 1
        G[k] = np.round(pat.observation.Gsub + 5.0 * rng.standard_normal())
        for j in range(P.DT): pat.step(Action(CHO=(cho if j == 0 else 0.0), insulin=bas * lvl + (bolus if j == 0 else 0.0)))
        U[k] = bas * lvl * P.DT + bolus
    return G, U


def cell(a):
    name, d, A, ms = a
    try:
        tru = P.true_bolus_effect(name)
        G, U, R, M = P.run_closed_loop(name, 14, d, A, seed=ms * 1000 + int(d * 10) * 10 + int(A * 10) + (abs(hash(name)) % 97))
        lin = P.linear_twin(G, U); row = dict(patient=name, d=d, A=A, meal_seed=ms, true=tru.tolist(), lin=lin.tolist(), hypo=float(np.mean(G < 70)), tir=float(np.mean((G >= 70) & (G <= 180))))
        if A > 0: row['iv'] = P.iv_twin(G, U, R).tolist()
        return row
    except Exception as e:
        return dict(patient=name, d=d, A=A, meal_seed=ms, error=str(e))


def control(name):
    try:
        tru = P.true_bolus_effect(name); G, U = designed_log(name); lin = P.linear_twin(G, U)
        return dict(patient=name, true=tru.tolist(), lin_designed=lin.tolist(), rel_err_4h=float(abs(lin[H4] - tru[H4]) / abs(tru[H4])))
    except Exception as e:
        return dict(patient=name, error=str(e))


def raise_and_dose(name, dose):
    pat = T1DPatient.withName(name); bas = P.basal_Umin(name)
    for _ in range(6 * 60): pat.step(Action(CHO=0, insulin=bas))
    drip = 0.0
    for t in range(10 * 60):
        bg = pat.observation.Gsub; drip = float(np.clip(drip + 0.01 * (180.0 - bg), 0.0, 2.0)); pat.step(Action(CHO=drip, insulin=bas))
        if bg >= 178 and t > 120: break
    bg0 = pat.observation.Gsub; bgs = []
    for k in range(96):
        for j in range(P.DT): pat.step(Action(CHO=0, insulin=bas + (dose if (k == 0 and j == 0) else 0.0)))      # drip STOPPED at dose time
        bgs.append(pat.observation.Gsub)
    bgs = np.array(bgs); return dict(bg0=float(bg0), bg4h=float(bgs[H4]), bgmin=float(bgs.min()), hypo=bool(bgs.min() < 70), severe=bool(bgs.min() < 54))


def dosing(a):
    name, label, eff4 = a
    dose = 0.0 if eff4 >= 0 else float(min(10.0, 60.0 / (-eff4)))
    try: return dict(patient=name, label=label, eff4=float(eff4), dose=dose, **raise_and_dose(name, dose))
    except Exception as e: return dict(patient=name, label=label, error=str(e))


if __name__ == '__main__':
    names = [f'{c}#{i:03d}' for c in ('adult', 'adolescent', 'child') for i in range(1, 11)]
    W = int(os.environ.get('WORKERS', 4))
    with Pool(W) as p:
        ctrl = p.map(control, names); json.dump(ctrl, open('c6d_control.json', 'w'))
        ok = [c for c in ctrl if 'error' not in c]; passed = {c['patient'] for c in ok if c['rel_err_4h'] < 0.5}
        print(f'capability control: {len(passed)}/{len(ok)} patients pass (rel. 4-h error < 0.5); median rel err {np.median([c["rel_err_4h"] for c in ok]):.2f}', flush=True)
        rows = p.map(cell, [(n, d, A, ms) for n in names for d in (0.0, 0.7) for A in (0.0, 0.5) for ms in range(3)])
        json.dump(rows, open('c6d_results.json', 'w'))
        R = [r for r in rows if 'error' not in r and r['hypo'] < 0.5]
        for d in (0.0, 0.7):
            for A in (0.0, 0.5):
                S = [r for r in R if r['d'] == d and r['A'] == A and r['patient'] in passed]
                t2 = np.array([r['true'][23] for r in S]); w2 = np.array([r['lin'][23] for r in S]); t4 = np.array([r['true'][H4] for r in S]); w4 = np.array([r['lin'][H4] for r in S])
                print(f'd={d} A={A} n={len(S)} (patients passing control, hypo<50 %): twin>0 at 2 h {np.mean(w2>0):.2f}, at 4 h {np.mean(w4>0):.2f}; '
                      f'learned/true 4 h median {np.median(w4/t4):.2f}; Spearman(true,twin 4 h) {np.corrcoef(np.argsort(np.argsort(t4)), np.argsort(np.argsort(w4)))[0,1]:+.2f}', flush=True)
        # dosing: true effect (reference), operational twins (A=0) for both loops, IV twins (A=0.5); meal seed 0; passing patients only
        jobs = []
        for n in sorted(passed):
            tru = [r for r in R if r['patient'] == n][0]['true'][H4] if any(r['patient'] == n for r in R) else None
            if tru is None: continue
            jobs.append((n, 'true effect', tru))
            for d in (0.0, 0.7):
                S = [r for r in R if r['patient'] == n and r['d'] == d and r['A'] == 0.0 and r['meal_seed'] == 0]
                if S: jobs.append((n, f'operational twin d={d}', S[0]['lin'][H4]))
                S = [r for r in R if r['patient'] == n and r['d'] == d and r['A'] == 0.5 and r['meal_seed'] == 0]
                if S: jobs.append((n, f'IV twin d={d}', S[0]['iv'][H4]))
        dos = p.map(dosing, jobs); json.dump(dos, open('c6d_dosing.json', 'w'))
        for lab in sorted({x['label'] for x in dos}):
            S = [x for x in dos if x['label'] == lab and 'error' not in x]
            print(f"DOSING {lab:26s} n={len(S)}: BG0 {np.mean([x['bg0'] for x in S]):.0f} (range {min(x['bg0'] for x in S):.0f}–{max(x['bg0'] for x in S):.0f}), dose {np.mean([x['dose'] for x in S]):.1f} U, BG@4h {np.mean([x['bg4h'] for x in S]):.0f}, "
                  f"min {np.mean([x['bgmin'] for x in S]):.0f}, hypo {sum(x['hypo'] for x in S)}/{len(S)}, severe {sum(x['severe'] for x in S)}/{len(S)}, no-dose {sum(x['dose']==0 for x in S)}/{len(S)}", flush=True)
