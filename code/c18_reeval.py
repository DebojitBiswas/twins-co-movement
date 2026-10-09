#!/usr/bin/env python3
"""
C18 — re-evaluate every saved v1 model (c15 + c16) against the corrected step truth (v1.1: seven starts 48 h apart, Boolean override fixed),
and add the referee-requested quantities: in-range-only RMSE (already), in-range-masked conformal coverage, in-range fraction of the step
trajectories, operational/designed ratio.  No retraining.  Output: c7_v1/P*/c18_results.json (same schema as c15 plus extras).
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import torch; torch.set_num_threads(1)
import c7_neural as N
import c13_pem_baselines as PB
import c15_v1_models as V

P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'; WARM, ROLL, ALPHA = 96, 192, 0.1
BANDS = {'0-6h': (0, 24), '6-24h': (24, 96), '24-48h': (96, 192)}


def load_sim(fam, train, seed):
    import torch
    L = np.load(f'{D}/{train}.npy'); tag = f'{fam}_{train}_s{seed}'
    if fam.startswith('arx'):
        th = np.load(f'{D}/models/{tag}.npy'); m = dict(p=int(fam[3:]), th=th); return lambda yh, w: PB.arx_sim(m, yh, w)
    if fam.startswith('pem') or fam == 'grey':
        theta = np.load(f'{D}/models/{tag}.npy'); my, sy, mw, sw = PB.norm_stats(L)
        m = dict(n=2 if fam == 'grey' else int(fam[3:]), grey=(fam == 'grey'), theta=theta, norm=(my, sy, mw, sw)); return lambda yh, w: PB.ss_sim(m, yh, w)
    kind = fam.replace('_noise', '').replace('_big', ''); big = fam.endswith('_big')
    h, layers = (256, 2) if (big and kind.startswith('gru')) else (128, 4) if big else (48, 2)
    m = N.build(kind, 2, h=h, layers=layers); m.load_state_dict(torch.load(f'{D}/models/{tag}.pt', map_location='cpu'))
    for p in m.parameters(): p.requires_grad_(False)
    y, u, x = L[:, 0], L[:, 1], L[:, 2:4]; my, sy, mu, su, mx, sx = y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, x.mean(0), x.std(0) + 1e-6
    T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    def sim(yh, w):
        with torch.no_grad():
            return m.roll(T((yh - my) / sy)[None], T((w[:, 0] - mu) / su)[None], T((w[:, 1:3] - mx) / sx)[None], len(yh))[0, len(yh) - 1:len(yh) - 1 + (len(w) - len(yh))].numpy() * sy + my
    return sim


def evaluate(sim):
    out = V.evaluate(sim)
    truth = json.load(open(f'{D}/truth_step.json')); op = np.load(f'{D}/log_operational.npy'); lo, hi = op[:, 0].min(), op[:, 0].max()
    # in-range fraction of the step trajectories (truth branches), both arms
    out['step']['frac_inside_base'] = float(np.mean([np.mean((np.array(t['base'])[:96, 0] >= lo) & (np.array(t['base'])[:96, 0] <= hi)) for t in truth]))
    out['step']['frac_inside_interv'] = float(np.mean([np.mean((np.array(t['interv'])[:96, 0] >= lo) & (np.array(t['interv'])[:96, 0] <= hi)) for t in truth]))
    # in-range-masked conformal coverage on test_in
    opt = np.load(f'{D}/test_operational.npy'); cal = opt[:7 * 96 + WARM]
    q = np.quantile(np.abs(np.array([sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0] for s in V.windows(cal, 24)])), 1 - ALPHA, axis=0)
    Lin = np.load(f'{D}/test_in.npy'); R, M = [], []
    for s in V.windows(Lin, 24):
        R.append(sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0]); M.append((s[WARM:, 0] >= lo) & (s[WARM:, 0] <= hi))
    R, M = np.array(R), np.array(M)
    out['coverage']['test_in_inrange'] = {b: float(np.mean((np.abs(R[:, a:c]) <= q[a:c])[M[:, a:c]])) if M[:, a:c].any() else None for b, (a, c) in BANDS.items()}
    return out


def job(a):
    fam, train, seed = a
    try:
        sim = load_sim(fam, train, seed); r = dict(family=fam, train=train, seed=seed, period=P, **evaluate(sim))
        print(f"{P} {fam:12s} {train:18s} s{seed}: in {r['test']['test_in']['rmse']:.2f} (inside {r['test']['test_in']['rmse_inside']:.2f}) op {r['test']['test_operational']['rmse']:.2f} | step24 ratio {r['step']['ratio_24h']:.2f} | cov in-range {r['coverage']['test_in_inrange']['6-24h']}", flush=True)
        return r
    except Exception as e:
        print(f'FAILED {fam} {train} s{seed}: {e}', flush=True); return None


if __name__ == '__main__':
    rows = json.load(open(f'{D}/c15_results.json')); jobs = [(r['family'], r['train'], r['seed']) for r in rows]
    if os.path.exists(f'{D}/c16_results.json'): jobs += [(r['family'], r['train'], r['seed']) for r in json.load(open(f'{D}/c16_results.json'))]
    with Pool(int(os.environ.get('WORKERS', 12))) as p: res = [r for r in p.map(job, jobs) if r]
    json.dump(res, open(f'{D}/c18_results.json', 'w')); print(f'{P}: {len(res)} models re-evaluated')
