#!/usr/bin/env python3
"""
C16 — referee ML M3 / control M3: larger neural twins on the v1 building data, trained on the GPU.  Families: gru_sim (256 hidden), gru_1s
(256 hidden, no input noise, 10,000 steps), tf_1s (d = 128, 4 layers, no input noise, 10,000 steps).  Training logs: operational, dither ±2 K
seed 0, designed seed 0; 3 seeds.  Same evaluation as c15 (in-support / out-of-range / operational tests, within-support step from the
operational contexts, conformal coverage).  Run: PYTHONPATH=. ~/ndo/thermal_env/bin/python c16_big_models.py   (period via C7_V1_PERIOD)
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c7_neural as N

P = os.environ.get('C7_V1_PERIOD', 'P1'); D = f'c7_v1/{P}'; WARM, ROLL = 96, 192; ALPHA = 0.1; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BANDS = {'0-6h': (0, 24), '6-24h': (24, 96), '24-48h': (96, 192)}
SIZES = {'gru_sim': dict(h=256, layers=2, steps=4000), 'gru_1s': dict(h=256, layers=2, steps=10000), 'tf_1s': dict(h=128, layers=4, steps=10000)}


def windows(L, stride=96):
    for st in range(WARM, len(L) - ROLL + 1, stride): yield L[st - WARM:st + ROLL]


def train(kind, L, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); cfg = SIZES[kind]
    y, u, x = L[:, 0], L[:, 1], L[:, 2:4]
    nz = (y.mean(), y.std() + 1e-6, u.mean(), u.std() + 1e-6, x.mean(0), x.std(0) + 1e-6); my, sy, mu, su, mx, sx = nz
    T = lambda a_: torch.tensor(a_, dtype=torch.float32, device=DEV); yn, un, xn = T((y - my) / sy), T((u - mu) / su), T((x - mx) / sx)
    m = N.build(kind, 2, h=cfg['h'], layers=cfg['layers']).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3); B = 64
    for it in range(cfg['steps']):
        if kind == 'gru_sim':
            Lw = 192; idx = rng.integers(0, len(y) - Lw - 1, B)
            yb = torch.stack([yn[i:i + Lw + 1] for i in idx]); ub = torch.stack([un[i:i + Lw] for i in idx]); xb = torch.stack([xn[i:i + Lw] for i in idx])
            pred = m.roll(yb[:, :48], ub, xb, 48); loss = ((pred[:, 47:] - yb[:, 48:]) ** 2).mean()
        elif kind == 'gru_1s':
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
            return m.roll(T((yh - my) / sy)[None], T((w[:, 0] - mu) / su)[None], T((w[:, 1:3] - mx) / sx)[None], len(yh))[0, len(yh) - 1:len(yh) - 1 + (len(w) - len(yh))].cpu().numpy() * sy + my
    return sim, m.state_dict(), float(loss.item())


def evaluate(sim):
    out = dict(test={}, coverage={}, step=None); truth = json.load(open(f'{D}/truth_step.json')); op0 = np.load(f'{D}/log_operational.npy'); lo, hi = op0[:, 0].min(), op0[:, 0].max()
    for name in ('test_in', 'test_out', 'test_operational'):
        L = np.load(f'{D}/{name}.npy'); errs, pers, R, inside = [], [], [], []
        for seg in windows(L):
            pr = sim(seg[:WARM, 0], seg[:, 1:4]); r = pr - seg[WARM:, 0]; R.append(r); errs.append(np.mean(r ** 2)); pers.append(np.mean((seg[WARM - 1, 0] - seg[WARM:, 0]) ** 2)); inside.append((seg[WARM:, 0] >= lo) & (seg[WARM:, 0] <= hi))
        R, inside = np.array(R), np.array(inside)
        out['test'][name] = dict(rmse=float(np.sqrt(np.mean(errs))), persistence=float(np.sqrt(np.mean(pers))), per_window=[float(np.sqrt(e)) for e in errs], rmse_inside=float(np.sqrt(np.mean(R[inside] ** 2))) if inside.any() else None, frac_inside=float(inside.mean()))
    resp, tr = [], []
    for t in truth:
        pre = np.array(t['pre']); base = np.array(t['base']); interv = np.array(t['interv'])
        resp.append(sim(pre[:, 0], np.r_[pre[:, 1:4], interv[:, 1:4]]) - sim(pre[:, 0], np.r_[pre[:, 1:4], base[:, 1:4]])); tr.append(np.array(t['effect']))
    resp, tr = np.array(resp), np.array(tr)
    out['step'] = dict(learned=resp.mean(0).round(3).tolist(), truth=tr.mean(0).round(3).tolist(), per_start_24h=[float(v) for v in resp[:, 95]], truth_per_start_24h=[float(v) for v in tr[:, 95]], ratio_24h=float(resp[:, 95].mean() / tr[:, 95].mean()), ratio_48h=float(resp[:, 191].mean() / tr[:, 191].mean()))
    op = np.load(f'{D}/test_operational.npy'); cal = op[:7 * 96 + WARM]; rest = op[7 * 96:]
    q = np.quantile(np.abs(np.array([sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0] for s in windows(cal, 24)])), 1 - ALPHA, axis=0)
    for name, L in (('operational', rest), ('test_in', np.load(f'{D}/test_in.npy'))):
        R = np.array([sim(s[:WARM, 0], s[:, 1:4]) - s[WARM:, 0] for s in windows(L, 24)])
        out['coverage'][name] = {b: float(np.mean(np.abs(R[:, lo_:hi_]) <= q[lo_:hi_])) for b, (lo_, hi_) in BANDS.items()}
    return out


if __name__ == '__main__':
    trains = ['log_operational', 'log_dither_A2.0_s0', 'log_designed_s0']; os.makedirs(f'{D}/models', exist_ok=True)
    out = f'{D}/c16_results.json'; rows = json.load(open(out)) if os.path.exists(out) else []
    FAMS = os.environ.get('C16_FAMS', 'gru_sim,gru_1s,tf_1s').split(','); NSEED = int(os.environ.get('C16_SEEDS', 0))
    for kind in FAMS:
        for t in (trains if (kind == 'gru_sim' and not NSEED) else ['log_operational', 'log_designed_s0']):
            for seed in (range(NSEED) if NSEED else (range(3) if kind == 'gru_sim' else range(2))):
                if any(r['family'] == f'{kind}_big' and r['train'] == t and r['seed'] == seed for r in rows): continue
                L = np.load(f'{D}/{t}.npy'); sim, sd, loss = train(kind, L, seed); torch.save(sd, f'{D}/models/{kind}_big_{t}_s{seed}.pt')
                r = dict(family=f'{kind}_big', train=t, seed=seed, period=P, loss=loss, **evaluate(sim)); rows.append(r); json.dump(rows, open(out, 'w'))
                print(f"{P} {kind}_big {t:18s} s{seed}: in {r['test']['test_in']['rmse']:.2f} (pers {r['test']['test_in']['persistence']:.2f}) out {r['test']['test_out']['rmse']:.2f} op {r['test']['test_operational']['rmse']:.2f} | step24 ratio {r['step']['ratio_24h']:.2f} | cov op {r['coverage']['operational']['6-24h']:.2f} in {r['coverage']['test_in']['6-24h']:.2f}", flush=True)
