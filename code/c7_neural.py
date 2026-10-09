#!/usr/bin/env python3
"""
C7 — stateful neural twins on BOPTEST logs: GRU (simulation-error) and causal Transformer (one-step, c5c recipe), inputs (y, u, x) with
x = (outdoor temperature, solar).  Primary metric: do-RMSE of zone temperature over 48-h rollouts on held-out DESIGNED test schedules
(days 42–56, seeds 10/11), warm-up = the preceding 24 h of true history, weather given.  Training logs: designed (capability control),
operational, dither A ∈ {0.5, 1, 2}.  Also: the twin's response to a +0.3 modulation step (contrast of two rollouts) from the same warm-up.
"""
import os, sys, json, numpy as np
DATA = {'bestest_hydronic_heat_pump': 'c7_data', 'singlezone_commercial_hydronic': 'c7_data_commercial'}[os.environ.get('C7_CASE', 'bestest_hydronic_heat_pump')]
from multiprocessing import Pool

WARM, ROLL = 96, 192                       # 24 h warm-up, 48 h rollout (15-min steps)


def build(kind, nx, h=48, layers=2):
    import torch, torch.nn as nn

    class GRUSim(nn.Module):
        def __init__(s, h=h):
            super().__init__(); s.cell = nn.GRUCell(2 + nx, h); s.out = nn.Linear(h, 1); s.h = h
        def roll(s, y, u, x, warm):                       # y (B, ≥warm); u (B, L); x (B, L, nx) → predictions of y_{t+1}
            B, L = u.shape; hs = torch.zeros(B, s.h, device=u.device); yp = y[:, 0]; outs = []
            for t in range(L):
                yin = y[:, t] if t < warm else yp
                hs = s.cell(torch.cat([yin[:, None], u[:, t:t + 1], x[:, t]], 1), hs); yp = yin + 0.1 * s.out(hs)[:, 0]; outs.append(yp)
            return torch.stack(outs, 1)

    class TF(nn.Module):
        def __init__(s, d=h, ctx=WARM):
            super().__init__(); s.inp = nn.Linear(2 + nx, d); s.pos = nn.Parameter(torch.zeros(ctx, d))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True), layers); s.out = nn.Linear(d, 1); s.ctx = ctx
            s.register_buffer('mask', torch.triu(torch.full((ctx, ctx), float('-inf')), 1))
        def step_all(s, y, u, x):
            L = y.shape[1]; z = s.inp(torch.cat([y[..., None], u[..., None], x], -1)) + s.pos[:L]
            return y + 0.1 * s.out(s.enc(z, mask=s.mask[:L, :L]))[..., 0]
        def roll(s, y, u, x, warm):
            ys = [y[:, t] for t in range(warm)]; outs = []
            for t in range(u.shape[1]):
                lo = max(0, t + 1 - s.ctx); yn = s.step_all(torch.stack(ys[lo:t + 1], 1), u[:, lo:t + 1], x[:, lo:t + 1])[:, -1]; outs.append(yn)
                if t + 1 >= warm: ys.append(yn)
            return torch.stack(outs, 1)
    return {'gru_sim': GRUSim, 'gru_1s': GRUSim, 'tf_1s': TF}[kind]()        # gru_1s: same network, next-step training


def job(a):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    kind, train_name, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    L = np.load(f'{DATA}/log_{train_name}.npy'); y, u, x = L[:, 0], L[:, 1], L[:, 2:4]
    my, sy, mu, su = y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6; mx, sx = x.mean(0), x.std(0) + 1e-6
    T = lambda a_: torch.tensor(a_, dtype=torch.float32)
    yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = build(kind, 2); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    steps = 2000 if kind == 'gru_sim' else 6000
    for it in range(steps):
        if kind == 'gru_sim':
            Lw = 48 + 144; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        else:
            Lw = WARM; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.step_all(yb[:, :Lw] + 0.1 * torch.randn_like(yb[:, :Lw]), ub, xb); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    res = dict(kind=kind, train=train_name, seed=seed, loss=float(loss.item()), test={}, step_resp=[])
    with torch.no_grad():
        for s in (10, 11):
            te = np.load(f'{DATA}/test_designed_s{s}.npy'); errs = []; pers = []; resp = []
            for st in range(WARM, len(te) - ROLL, ROLL // 2):
                seg = te[st - WARM:st + ROLL]
                yy = T((seg[:WARM, 0] - my) / sy)[None]; uu = T((seg[:, 1] - mu) / su)[None]; xx = T((seg[:, 2:4] - mx) / sx)[None]
                pr = m.roll(yy, uu, xx, WARM)[0, WARM - 1:WARM - 1 + ROLL].numpy() * sy + my
                errs.append(np.mean((pr - seg[WARM:, 0]) ** 2)); pers.append(np.mean((seg[WARM - 1, 0] - seg[WARM:, 0]) ** 2))
                # step contrast: same warm-up and weather, modulation held at 0.4 vs 0.7 for the rollout
                u2 = np.stack([np.r_[seg[:WARM, 1], np.full(ROLL, 0.4)], np.r_[seg[:WARM, 1], np.full(ROLL, 0.7)]])
                pr2 = m.roll(yy.repeat(2, 1), T((u2 - mu) / su), xx.repeat(2, 1, 1), WARM)[:, WARM - 1:WARM - 1 + ROLL].numpy() * sy
                resp.append(pr2[1] - pr2[0])
            res['test'][s] = dict(do_rmse=float(np.sqrt(np.mean(errs))), persistence=float(np.sqrt(np.mean(pers))))
            res['step_resp'].append(np.mean(resp, 0).round(3).tolist())
    return res


if __name__ == '__main__':
    trains = ['designed_A0.0_s0', 'designed_A0.0_s1', 'operational_A0.0_s0', 'dither_A0.5_s0', 'dither_A1.0_s0', 'dither_A2.0_s0']
    jobs = [(k, t, s) for k in ('gru_sim', 'tf_1s') for t in trains for s in (0, 1, 2)]
    with Pool(int(os.environ.get('WORKERS', 18))) as p: rows = p.map(job, jobs)
    json.dump(rows, open(f'{DATA}/c7_neural.json', 'w'))
    for r in rows:
        sr = np.mean(r['step_resp'], 0)
        print(f"{r['kind']:8s} {r['train']:20s} s{r['seed']}: do-RMSE {r['test'][10]['do_rmse']:.2f} / {r['test'][11]['do_rmse']:.2f} K "
              f"(persistence {r['test'][10]['persistence']:.2f} / {r['test'][11]['persistence']:.2f})  +0.3 step @6/24/48 h {sr[[23, 95, 191]].round(2)}", flush=True)
