#!/usr/bin/env python3
"""C17 — referee MAJOR 7: small hyper-parameter sweep for the clean next-step families (gru_1s, tf_1s), P1: hidden {48, 96}, lr {1e-3, 3e-3},
steps {6000, 10000}, 2 seeds, trained on the designed log (selection) and the operational log (report).  Selection rule (pre-stated here):
the recipe with the lowest in-range designed-test error on designed-data training; its operational-log learned fraction is then the reported one."""
import os, sys, json, numpy as np
from multiprocessing import Pool
import c7_neural as N
import c15_v1_models as V

D = 'c7_v1/P1'


def train(kind, L, seed, h, lr, steps):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y, u, x = L[:, 0], L[:, 1], L[:, 2:4]; nz = (y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, x.mean(0), x.std(0) + 1e-6); my, sy, mu, su, mx, sx = nz
    T = lambda a_: torch.tensor(a_, dtype=torch.float32); yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = N.build(kind, 2, h=h); opt = torch.optim.Adam(m.parameters(), lr); B = 32
    for it in range(steps):
        if kind == 'gru_1s':
            Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :Lw], ub, xb, Lw); loss = ((pred - yb[:, 1:]) ** 2).mean()
        else:
            Lw = N.WARM; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.step_all(yb[:, :Lw], ub, xb); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    def sim(yh, w):
        with torch.no_grad():
            return m.roll(T((yh - my) / sy)[None], T((w[:, 0] - mu) / su)[None], T((w[:, 1:3] - mx) / sx)[None], len(yh))[0, len(yh) - 1:len(yh) - 1 + (len(w) - len(yh))].numpy() * sy + my
    return sim


def job(a):
    kind, tr, seed, h, lr, steps = a
    sim = train(kind, np.load(f'{D}/{tr}.npy'), seed, h, lr, steps); r = V.evaluate(sim)
    out = dict(kind=kind, train=tr, seed=seed, h=h, lr=lr, steps=steps, in_rmse=r['test']['test_in']['rmse'], pred=r['test']['test_operational']['rmse'], ratio=r['step']['ratio_24h'])
    print(f"{kind} {tr:16s} h={h} lr={lr:g} steps={steps} s{seed}: in {out['in_rmse']:.2f} pred {out['pred']:.2f} ratio {out['ratio']:.2f}", flush=True); return out


if __name__ == '__main__':
    V.P, V.D = 'P1', D
    jobs = [(k, tr, s, h, lr, st) for k in ('gru_1s', 'tf_1s') for tr in ('log_designed_s0', 'log_operational') for h in (48, 96) for lr in (1e-3, 3e-3) for st in (6000, 10000) for s in (0, 1)]
    with Pool(int(os.environ.get('WORKERS', 12))) as p: rows = p.map(job, jobs)
    json.dump(rows, open(f'{D}/c17_hp_sweep.json', 'w'), indent=1)
    for k in ('gru_1s', 'tf_1s'):
        best = min([(np.mean([r['in_rmse'] for r in rows if r['kind'] == k and r['train'] == 'log_designed_s0' and (r['h'], r['lr'], r['steps']) == cfg]), cfg) for cfg in {(r['h'], r['lr'], r['steps']) for r in rows}])
        op = [r['ratio'] for r in rows if r['kind'] == k and r['train'] == 'log_operational' and (r['h'], r['lr'], r['steps']) == best[1]]
        print(f"SELECTED {k}: h={best[1][0]} lr={best[1][1]:g} steps={best[1][2]} (designed in-range {best[0]:.2f}) → operational learned fraction {np.mean(op):.2f} (seeds {np.round(op, 2).tolist()})", flush=True)
