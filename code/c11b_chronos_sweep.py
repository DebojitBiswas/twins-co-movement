#!/usr/bin/env python3
"""
C11b — referee M3: Chronos-2 fine-tuning sweep on the heat-pump building.  Capability criterion (pre-stated here, before the sweep):
a fine-tuning recipe PASSES if, trained on the designed log, its learned 24-h step response is ≥ 70 % of the designed-data GRU/transformer
twins' mean (≈ 2.4 K → ≥ 1.7 K) AND its do-RMSE on the designed test schedules is below persistence.  The best passing recipe (lowest
designed-test do-RMSE) is then applied to the operational and dither ±2 K logs (3 seeds).  Run with PYTHONPATH=fm_pkgs thermal_env python.
Grid: learning rate {1e-5, 1e-4, 1e-3} × steps {1000, 5000}; batch 32; context 4 d; horizon 48 h; full fine-tuning.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c11_foundation as F
from chronos import Chronos2Pipeline

DATA = 'c7_data'; G_REF = 2.4; PASS_FRAC = 0.7

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    op_test = np.load(f'{DATA}/test_operational.npy'); tests = {s: np.load(f'{DATA}/test_designed_s{s}.npy') for s in ('10', '11')}
    base = Chronos2Pipeline.from_pretrained(F.MODEL, device_map=dev, dtype=torch.float32)
    Ld = np.load(f'{DATA}/log_designed_A0.0_s0.npy'); rows = []
    for lr in (1e-5, 1e-4, 1e-3):
        for steps in (1000, 5000):
            F.LR, F.STEPS = lr, steps; t0 = time.time()
            pipe = F.finetune(base, Ld, 0); r = dict(model='ft:designed', lr=lr, steps=steps, seed=0, **F.evaluate(pipe, tests, op_test)); rows.append(r)
            step24 = np.array(r['step_resp'])[95]; passed = step24 >= PASS_FRAC * G_REF and np.mean([r['do_rmse_10'], r['do_rmse_11']]) < np.mean([r['persist_10'], r['persist_11']])
            r['pass'] = bool(passed)
            print(f"designed lr={lr:g} steps={steps}: do-RMSE {r['do_rmse_10']:.2f}/{r['do_rmse_11']:.2f} pred-op {r['pred_rmse_operational']:.2f} step24h {step24:.2f} K  {'PASS' if passed else 'fail'}  [{time.time()-t0:.0f}s]", flush=True)
            json.dump(rows, open(f'{DATA}/c11b_sweep.json', 'w'), indent=1); del pipe; torch.cuda.empty_cache()
    passing = [r for r in rows if r['pass']]
    if not passing: print('no recipe passes the capability criterion'); sys.exit(0)
    best = min(passing, key=lambda r: np.mean([r['do_rmse_10'], r['do_rmse_11']])); F.LR, F.STEPS = best['lr'], best['steps']
    print(f"best passing recipe: lr={best['lr']:g} steps={best['steps']}", flush=True)
    for train in ('operational_A0.0_s0', 'dither_A2.0_s0', 'designed_A0.0_s0'):
        L = np.load(f'{DATA}/log_{train}.npy')
        for seed in (0, 1, 2):
            t0 = time.time(); pipe = F.finetune(base, L, seed); r = dict(model=f'ft:{train}', lr=best['lr'], steps=best['steps'], seed=seed, **F.evaluate(pipe, tests, op_test)); rows.append(r)
            print(f"ft:{train} s{seed} (best recipe): do-RMSE {r['do_rmse_10']:.2f}/{r['do_rmse_11']:.2f} pred-op {r['pred_rmse_operational']:.2f} step24h {np.array(r['step_resp'])[95]:.2f} K [{time.time()-t0:.0f}s]", flush=True)
            json.dump(rows, open(f'{DATA}/c11b_sweep.json', 'w'), indent=1); del pipe; torch.cuda.empty_cache()
