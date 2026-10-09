#!/usr/bin/env python3
"""
C11 — T3 headline test with a pretrained time-series foundation model: Chronos-2 (amazon/chronos-2, covariate-aware) on the BOPTEST
heat-pump building.  Run with:  PYTHONPATH=fm_pkgs ~/ndo/thermal_env/bin/python c11_foundation.py   (GPU)

Twins: zone temperature forecast given KNOWN FUTURE covariates (actuator u, outdoor T, solar) — the foundation-model digital twin.
  zero-shot            pretrained weights, no adaptation
  ft:operational       fine-tuned on the 28-day operational log
  ft:dither_A1 / A2    fine-tuned on dithered logs
  ft:designed          fine-tuned on the designed log (capability control)
Scores (48-h horizon, 4-day context, windows stride 24 h): do-RMSE on held-out designed test schedules (s10, s11); predictive RMSE on
an operational log of the same test weeks; learned +0.3 step response (contrast of forecasts with u held at 0.4 vs 0.7).
3 fine-tuning seeds.  Prediction: ft:operational — predictive RMSE small, do-RMSE worse than persistence, step response ≪ ft:designed.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts

DATA = {'bestest_hydronic_heat_pump': 'c7_data', 'singlezone_commercial_hydronic': 'c7_data_commercial'}[os.environ.get('C7_CASE', 'bestest_hydronic_heat_pump')]; CTX, H = 384, 192; COV = ['u', 'tout', 'solar']
MODEL = os.environ.get('FM_MODEL', 'amazon/chronos-2'); STEPS = int(os.environ.get('FM_STEPS', 1000)); LR = float(os.environ.get('FM_LR', 1e-5))


def ensure_test_operational():
    f = f'{DATA}/test_operational.npy'
    if not os.path.exists(f):
        import c7_boptest as C
        np.save(f, C.run_log('operational', 0.0, 0, t0=C.TEST0, days=C.TEST_DAYS))
    return np.load(f)


def windows(L, stride=96):
    for st in range(CTX, len(L) - H + 1, stride):
        yield L[st - CTX:st], L[st:st + H]


def inputs_from(L, u_future=None):
    items, truth = [], []
    for past, fut in windows(L):
        fu = fut[:, 1] if u_future is None else np.full(H, u_future)
        items.append(dict(target=past[:, 0].astype(np.float32),
                          past_covariates=dict(u=past[:, 1].astype(np.float32), tout=past[:, 2].astype(np.float32), solar=past[:, 3].astype(np.float32)),
                          future_covariates=dict(u=fu.astype(np.float32), tout=fut[:, 2].astype(np.float32), solar=fut[:, 3].astype(np.float32))))
        truth.append(fut[:, 0]);
    return items, np.array(truth)


def forecast(pipe, items):
    _, mean = pipe.predict_quantiles(items, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9], batch_size=64)
    return np.stack([m.squeeze().cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m).squeeze() for m in mean])


def evaluate(pipe, tests, op_test):
    out = {}
    for name, L in tests.items():
        items, truth = inputs_from(L); pred = forecast(pipe, items)
        out[f'do_rmse_{name}'] = float(np.sqrt(np.mean((pred - truth) ** 2)))
        out[f'persist_{name}'] = float(np.sqrt(np.mean((np.array([p[-1, 0] for p, _ in windows(L)])[:, None] - truth) ** 2)))
    items, truth = inputs_from(op_test); pred = forecast(pipe, items); out['pred_rmse_operational'] = float(np.sqrt(np.mean((pred - truth) ** 2)))
    # step response: same contexts (operational test weeks), u held at 0.4 vs 0.7
    lo, _ = inputs_from(op_test, 0.4); hi, _ = inputs_from(op_test, 0.7)
    r = forecast(pipe, hi) - forecast(pipe, lo); out['step_resp'] = r.mean(0).round(3).tolist()
    return out


def finetune(base, L, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    items = [dict(target=L[:, 0].astype(np.float32), past_covariates=dict(u=L[:, 1].astype(np.float32), tout=L[:, 2].astype(np.float32), solar=L[:, 3].astype(np.float32)))]
    prepared = from_list_of_dicts(items, prediction_length=H, known_covariates_names=COV)
    return base.fit(prepared, prediction_length=H, context_length=CTX, learning_rate=LR, num_steps=STEPS, batch_size=32,
                    output_dir=f'/tmp/chronos_ft_{seed}', remove_printer_callback=True, seed=seed)


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    op_test = ensure_test_operational()
    tests = {s: np.load(f'{DATA}/test_designed_s{s}.npy') for s in ('10', '11')}
    base = Chronos2Pipeline.from_pretrained(MODEL, device_map=dev, dtype=torch.float32)
    rows = []; t0 = time.time()
    r = dict(model='zero-shot', seed=0, **evaluate(base, tests, op_test)); rows.append(r)
    print(f"zero-shot: do-RMSE {r['do_rmse_10']:.2f} / {r['do_rmse_11']:.2f} (persist {r['persist_10']:.2f} / {r['persist_11']:.2f})  pred-op {r['pred_rmse_operational']:.2f}  step@6/24/48h {np.array(r['step_resp'])[[23,95,191]]}  [{time.time()-t0:.0f}s]", flush=True)
    for train in ('operational_A0.0_s0', 'designed_A0.0_s0', 'dither_A1.0_s0', 'dither_A2.0_s0'):
        L = np.load(f'{DATA}/log_{train}.npy')
        for seed in (0, 1, 2):
            t1 = time.time(); pipe = finetune(base, L, seed)
            r = dict(model=f'ft:{train}', seed=seed, **evaluate(pipe, tests, op_test)); rows.append(r)
            print(f"ft:{train} s{seed}: do-RMSE {r['do_rmse_10']:.2f} / {r['do_rmse_11']:.2f}  pred-op {r['pred_rmse_operational']:.2f}  step@6/24/48h {np.array(r['step_resp'])[[23,95,191]]}  [{time.time()-t1:.0f}s]", flush=True)
            json.dump(rows, open(f'{DATA}/c11_foundation.json', 'w'), indent=1)
            del pipe; torch.cuda.empty_cache()
