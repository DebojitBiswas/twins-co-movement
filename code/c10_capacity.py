"""C10 — T3 test: scaling plateau in capacity.  Synthetic plant (c1), closed loop kp 2, d 0.8, no dither, N = 2e5, vs open-loop designed data.
GRU hidden size 8 … 256, simulation-error training; report PREDICTIVE error (held-out one-step RMSE on same-regime data) and
INTERVENTIONAL error (effect rel. error), 3 seeds."""
import json, numpy as np
from multiprocessing import Pool
import c1_synthetic as c
from c1b_gru_controls import openloop


def member(a):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    Y, U, Yte, Ute, h, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    my, sy, mu, su = Y.mean(), Y.std() + 1e-6, U.mean(), U.std() + 1e-6
    yn = torch.tensor((Y - my) / sy, dtype=torch.float32); un = torch.tensor((U - mu) / su, dtype=torch.float32)

    class Sim(nn.Module):
        def __init__(s):
            super().__init__(); s.cell = nn.GRUCell(2, h); s.out = nn.Linear(h, 1)
        def forward(s, y0, u, warm):
            Bz, L = u.shape; hs = torch.zeros(Bz, h); yp = y0[:, 0]; outs = []
            for t in range(L):
                yin = y0[:, t] if t < warm else yp
                hs = s.cell(torch.stack([yin, u[:, t]], 1), hs); yp = yin + 0.1 * s.out(hs)[:, 0]; outs.append(yp)
            return torch.stack(outs, 1)
    m = Sim(); opt = torch.optim.Adam(m.parameters(), 3e-3); warm, roll, B = 50, 150, 32; L = warm + roll
    for it in range(3000):
        idx = rng.integers(0, len(Y) - L - 1, B)
        yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
        pred = m(yb[:, :warm], ub, warm); loss = ((pred[:, warm - 1:] - yb[:, warm:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        # predictive: rollout error on held-out same-regime data (same recipe)
        yt = torch.tensor((Yte - my) / sy, dtype=torch.float32); ut = torch.tensor((Ute - mu) / su, dtype=torch.float32)
        idx = np.arange(0, len(Yte) - L - 1, L); yb = torch.stack([yt[i:i + L + 1] for i in idx]); ub = torch.stack([ut[i:i + L] for i in idx])
        pred = m(yb[:, :warm], ub, warm); pr = float(torch.sqrt(((pred[:, warm - 1:] - yb[:, warm:]) ** 2).mean()) * sy)
        y0 = torch.full((2, warm), (c.TREF - my) / sy)
        useq = np.r_[np.full(warm, c.U_OP), c.U_DES]
        ub = torch.tensor(np.stack([(useq - mu) / su, (np.full(warm + c.H, c.U_OP) - mu) / su]), dtype=torch.float32)
        out = m(y0, ub, warm).numpy() * sy + my
    eff = out[0, warm - 1:warm - 1 + c.H] - out[1, warm - 1:warm - 1 + c.H]
    return dict(h=h, seed=seed, pred_rmse=pr, eff_err=c.rel_err(eff), nparams=sum(p.numel() for p in m.parameters()))


if __name__ == '__main__':
    N = 200000
    Yc, Uc, _, _ = c.simulate(N, 2.0, 0.8, 0.0, 0.01, seed=7); Yct, Uct, _, _ = c.simulate(20000, 2.0, 0.8, 0.0, 0.01, seed=99)
    Yo, Uo = openloop(N, 3); Yot, Uot = openloop(20000, 5)
    jobs, meta = [], []
    for h in (8, 16, 32, 64, 128, 256):
        for s in (0, 1, 2):
            jobs.append((Yc, Uc, Yct, Uct, h, s)); meta.append('closed'); jobs.append((Yo, Uo, Yot, Uot, h, s)); meta.append('open')
    with Pool(12) as p: res = p.map(member, jobs)
    rows = [dict(regime=mt, **r) for mt, r in zip(meta, res)]
    json.dump(rows, open('c10_capacity.json', 'w'), indent=1)
    for reg in ('open', 'closed'):
        for h in (8, 16, 32, 64, 128, 256):
            R = [r for r in rows if r['regime'] == reg and r['h'] == h]
            print(f"{reg:6s} h={h:3d} ({R[0]['nparams']:6d} params): predictive RMSE {np.mean([r['pred_rmse'] for r in R]):.3f} ± {np.std([r['pred_rmse'] for r in R]):.3f}   "
                  f"interventional rel err {np.mean([r['eff_err'] for r in R]):.2f} ± {np.std([r['eff_err'] for r in R]):.2f}", flush=True)
