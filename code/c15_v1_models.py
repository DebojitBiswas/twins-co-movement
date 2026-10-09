#!/usr/bin/env python3
"""
C15 — v1 building study (protocol_v1_buildings.md).  For a period P (c7_v1/P*/): train every family on each training log, save weights,
and evaluate: do-RMSE on test_in (primary) / test_out (secondary) / test_operational (predictive), with the same windows and persistence for
all; step response (0.4 → 0.7, 48 h) from the seven operational start contexts of truth_step.json, relative to the truth; conformal coverage
(calibrated on the first 7 days of test_operational, stride 6 h; tested on the remaining 7 days and on test_in).
Families: arx{2,6,12,24,48}; pem{2,3,4}; grey; gru_sim; gru_1s / tf_1s with and without input noise (suffix _noise); 5 seeds for neural.
Usage: C7_V1_PERIOD=P1 WORKERS=12 python3 c15_v1_models.py [families...]
"""
import os, sys, json, numpy as np
from multiprocessing import Pool
import c7_neural as N
import c13_pem_baselines as PB

P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'; WARM, ROLL = 96, 192; ALPHA = 0.1
BANDS = {'0-6h': (0, 24), '6-24h': (24, 96), '24-48h': (96, 192)}


def windows(L, stride=96):
    for st in range(WARM, len(L) - ROLL + 1, stride): yield L[st - WARM:st + ROLL]


def neural_train(kind, L, seed, noise):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y, u, x = L[:, 0], L[:, 1], L[:, 2:4]
    nz = (y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, x.mean(0), x.std(0) + 1e-6); my, sy, mu, su, mx, sx = nz
    T = lambda a_: torch.tensor(a_, dtype=torch.float32); yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = N.build(kind, 2); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    for it in range(2000 if kind == 'gru_sim' else 6000):
        if kind == 'gru_sim':
            Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        elif kind == 'gru_1s':
            Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :Lw] + noise * torch.randn_like(yb[:, :Lw]), ub, xb, Lw); loss = ((pred - yb[:, 1:]) ** 2).mean()
        else:
            Lw = N.WARM; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.step_all(yb[:, :Lw] + noise * torch.randn_like(yb[:, :Lw]), ub, xb); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    def sim(yh, w):
        with torch.no_grad():
            return m.roll(T((yh - my) / sy)[None], T((w[:, 0] - mu) / su)[None], T((w[:, 1:3] - mx) / sx)[None], len(yh))[0, len(yh) - 1:len(yh) - 1 + (len(w) - len(yh))].numpy() * sy + my
    return sim, m.state_dict()


def evaluate(sim):
    out = dict(test={}, coverage={}, step=None)
    truth = json.load(open(f'{D}/truth_step.json'))
    for name in ('test_in', 'test_out', 'test_operational'):
        L = np.load(f'{D}/{name}.npy'); errs, pers, R, inside = [], [], [], []
        lo, hi = np.load(f'{D}/log_operational.npy')[:, 0].min(), np.load(f'{D}/log_operational.npy')[:, 0].max()
        for seg in windows(L):
            pr = sim(seg[:WARM, 0], seg[:, 1:4]); r = pr - seg[WARM:, 0]; R.append(r); errs.append(np.mean(r ** 2)); pers.append(np.mean((seg[WARM - 1, 0] - seg[WARM:, 0]) ** 2))
            inside.append((seg[WARM:, 0] >= lo) & (seg[WARM:, 0] <= hi))
        R, inside = np.array(R), np.array(inside)
        out['test'][name] = dict(rmse=float(np.sqrt(np.mean(errs))), persistence=float(np.sqrt(np.mean(pers))), per_window=[float(np.sqrt(e)) for e in errs],
                                 rmse_inside=float(np.sqrt(np.mean(R[inside] ** 2))) if inside.any() else None, frac_inside=float(inside.mean()))
    # step response from the operational contexts of the truth protocol
    resp, tr = [], []
    for t in truth:
        pre = np.array(t['pre']); base = np.array(t['base']); interv = np.array(t['interv'])
        wb = np.r_[pre[:, 1:4], base[:, 1:4]]; wi = np.r_[pre[:, 1:4], interv[:, 1:4]]
        resp.append(sim(pre[:, 0], wi) - sim(pre[:, 0], wb)); tr.append(np.array(t['effect']))
    resp, tr = np.array(resp), np.array(tr)
    out['step'] = dict(learned=resp.mean(0).round(3).tolist(), truth=tr.mean(0).round(3).tolist(), per_start_24h=[float(v) for v in resp[:, 95]], truth_per_start_24h=[float(v) for v in tr[:, 95]],
                       ratio_24h=float(resp[:, 95].mean() / tr[:, 95].mean()), ratio_48h=float(resp[:, 191].mean() / tr[:, 191].mean()))
    # conformal: calibrate on first 7 days of test_operational (stride 24 steps), test on the remaining days and on test_in
    op = np.load(f'{D}/test_operational.npy'); cal = op[:7 * 96 + WARM]; rest = op[7 * 96:]
    q = np.quantile(np.abs(np.array([sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0] for s in windows(cal, 24)])), 1 - ALPHA, axis=0)
    for name, L in (('operational', rest), ('test_in', np.load(f'{D}/test_in.npy'))):
        R = np.array([sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0] for s in windows(L, 24)])
        out['coverage'][name] = {b: float(np.mean(np.abs(R[:, lo:hi]) <= q[lo:hi])) for b, (lo, hi) in BANDS.items()}
    return out


def job(a):
    fam, train, seed = a
    L = np.load(f'{D}/{train}.npy'); os.makedirs(f'{D}/models', exist_ok=True); tag = f'{fam}_{train}_s{seed}'
    if fam.startswith('arx'): m = PB.arx_fit(L, int(fam[3:])); sim = lambda yh, w: PB.arx_sim(m, yh, w); np.save(f'{D}/models/{tag}.npy', m['th'])
    elif fam.startswith('pem'): m = PB.ss_fit(L, int(fam[3:]), seed=seed); sim = lambda yh, w: PB.ss_sim(m, yh, w); np.save(f'{D}/models/{tag}.npy', m['theta'])
    elif fam == 'grey': m = PB.ss_fit(L, 2, grey=True, seed=seed); sim = lambda yh, w: PB.ss_sim(m, yh, w); np.save(f'{D}/models/{tag}.npy', m['theta'])
    else:
        import torch
        kind = fam.replace('_noise', ''); noise = 0.1 if fam.endswith('_noise') else 0.0
        sim, sd = neural_train(kind, L, seed, noise); torch.save(sd, f'{D}/models/{tag}.pt')
    r = dict(family=fam, train=train, seed=seed, period=P, **evaluate(sim))
    print(f"{P} {fam:12s} {train:18s} s{seed}: in {r['test']['test_in']['rmse']:.2f} (pers {r['test']['test_in']['persistence']:.2f}) out {r['test']['test_out']['rmse']:.2f} op {r['test']['test_operational']['rmse']:.2f} | step24 ratio {r['step']['ratio_24h']:.2f} | cov op {r['coverage']['operational']['6-24h']:.2f} in {r['coverage']['test_in']['6-24h']:.2f}", flush=True)
    return r


if __name__ == '__main__':
    fams = sys.argv[1:] or ['arx2', 'arx12', 'arx24', 'arx48', 'pem2', 'pem3', 'grey', 'gru_sim', 'gru_1s', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise']
    trains = ['log_operational'] + [f'log_dither_A2.0_s{s}' for s in range(3)] + ['log_designed_s0', 'log_designed_s1'] + ([f'log_dither_A{A}_s0' for A in (0.5, 1.0)] if P == 'P1' else [])
    jobs = []
    for f in fams:
        for t in trains:
            seeds = range(5) if f.startswith(('gru', 'tf')) else ([0] if f.startswith('arx') else [0])
            for s in seeds: jobs.append((f, t, s))
    out = f'{D}/c15_results.json'; done = json.load(open(out)) if os.path.exists(out) else []
    have = {(r['family'], r['train'], r['seed']) for r in done}; jobs = [j for j in jobs if j not in have]
    print(f'{P}: {len(jobs)} jobs', flush=True)
    with Pool(int(os.environ.get('WORKERS', 12))) as p:
        for r in p.imap_unordered(job, jobs):
            done.append(r); json.dump(done, open(out, 'w'))
