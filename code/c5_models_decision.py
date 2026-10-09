#!/usr/bin/env python3
"""
C5 — model families and decision consequence.

Model families trained on the same logs (c1 plant):
  gru_sim   GRU, simulation-error training (warm-up 50, free rollout 150)              — the digital-twin recipe
  gru_1s    GRU, teacher-forced one-step training on long windows (200)                  — the sequence-model / world-model recipe
  tf_1s     causal Transformer, teacher-forced next-step prediction, context 128         — the foundation-model recipe
Theory (c2): simulation-error → γ²G + (1−γ²)G_op;  one-step with long history → consistent under pure feedback (d = 0),
omitted-variable bias when the controller anticipates the disturbance (d > 0).

Decision consequence: for every trained model, plan an input sequence (bounded 0..UMAX) that makes the MODEL's predicted temperature
track a +3 K set-point change for 300 steps, then apply the plan open-loop to the TRUE plant (noise-free, q = 0) and score tracking RMSE.

Regimes: open-loop designed data; closed loop kp 2 with d ∈ {0, 0.8} and σ_r ∈ {0, 1}; N = 200,000.  3 seeds per model.
"""
import json, sys, time, numpy as np
from multiprocessing import Pool
import c1_synthetic as c
from c1b_gru_controls import openloop

WARM, HPLAN, TARGET = 128, 300, 3.0
import os
NOISE = float(os.environ.get("TF_NOISE", 0.0))


def build(kind, my, sy, mu, su):
    import torch, torch.nn as nn

    class GRUSim(nn.Module):
        def __init__(s, h=32):
            super().__init__(); s.cell = nn.GRUCell(2, h); s.out = nn.Linear(h, 1); s.h = h
        def roll(s, y_obs, u, warm):            # y_obs (B, warm) normalised; u (B, L)
            B, L = u.shape; hs = torch.zeros(B, s.h); yp = y_obs[:, 0]; outs = []
            for t in range(L):
                yin = y_obs[:, t] if t < warm else yp
                hs = s.cell(torch.stack([yin, u[:, t]], 1), hs); yp = yin + 0.1 * s.out(hs)[:, 0]; outs.append(yp)
            return torch.stack(outs, 1)          # outs[:, t] = prediction of y_{t+1}

    class GRU1s(GRUSim):
        pass                                     # same network; differs only in training (teacher forcing)

    class TF1s(nn.Module):
        def __init__(s, d=32, ctx=WARM):
            super().__init__(); s.inp = nn.Linear(2, d); s.pos = nn.Parameter(torch.zeros(ctx, d))
            layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.0, batch_first=True)
            s.enc = nn.TransformerEncoder(layer, 2); s.out = nn.Linear(d, 1); s.ctx = ctx
            s.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1))
        def step_all(s, y, u):                   # teacher-forced: predict y_{t+1} for every t in the window (length ≤ ctx)
            L = y.shape[1]; z = s.inp(torch.stack([y, u], -1)) + s.pos[:L]
            return y + 0.1 * s.out(s.enc(z, mask=s.mask[:L, :L]))[..., 0]
        def roll(s, y_obs, u, warm):
            ys = [y_obs[:, t] for t in range(warm)]; outs = []
            for t in range(u.shape[1]):
                lo = max(0, t + 1 - s.ctx); yw = torch.stack(ys[lo:t + 1], 1); uw = u[:, lo:t + 1]
                yn = s.step_all(yw, uw)[:, -1]; outs.append(yn)
                if t + 1 >= warm: ys.append(yn)
            return torch.stack(outs, 1)

    return {"gru_sim": GRUSim, "gru_1s": GRU1s, "tf_1s": TF1s}[kind]()


def train_eval(args):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    kind, Y, U, seed, steps = args
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    my, sy, mu, su = Y.mean(), Y.std() + 1e-6, U.mean(), U.std() + 1e-6
    yn = torch.tensor((Y - my) / sy, dtype=torch.float32); un = torch.tensor((U - mu) / su, dtype=torch.float32)
    m = build(kind, my, sy, mu, su); opt = torch.optim.Adam(m.parameters(), 3e-3); B = 32
    for it in range(steps):
        if kind == "gru_sim":
            L = 200; idx = rng.integers(0, len(Y) - L - 1, B)
            yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
            pred = m.roll(yb[:, :50], ub, 50); loss = ((pred[:, 49:] - yb[:, 50:]) ** 2).mean()
        elif kind == "gru_1s":
            L = 200; idx = rng.integers(0, len(Y) - L - 1, B)
            yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
            yin = yb[:, :L] + NOISE * torch.randn_like(yb[:, :L])
            pred = m.roll(yin, ub, L); loss = ((pred - yb[:, 1:]) ** 2).mean()
        else:
            L = WARM; idx = rng.integers(0, len(Y) - L - 1, B)
            yb = torch.stack([yn[i:i + L + 1] for i in idx]); ub = torch.stack([un[i:i + L] for i in idx])
            yin = yb[:, :L] + NOISE * torch.randn_like(yb[:, :L])       # input-noise injection against rollout drift
            pred = m.step_all(yin, ub); loss = ((pred - yb[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    for p in m.parameters(): p.requires_grad_(False)
    y0 = torch.full((1, WARM), (c.TREF - my) / sy)
    nu = lambda u: (torch.as_tensor(u, dtype=torch.float32) - mu) / su
    # 1) effect of the designed sequence
    with torch.no_grad():
        useq = np.stack([np.r_[np.full(WARM, c.U_OP), c.U_DES], np.full(WARM + c.H, c.U_OP)])
        out = m.roll(y0.repeat(2, 1), nu(useq), WARM).numpy() * sy + my
    eff = out[0, WARM - 1:WARM - 1 + c.H] - out[1, WARM - 1:WARM - 1 + c.H]
    # 2) plan: u = UMAX * sigmoid(a), make model track TREF + TARGET
    a = torch.zeros(1, HPLAN, requires_grad=True); popt = torch.optim.Adam([a], 0.05)
    target = (c.TREF + TARGET - my) / sy
    for it in range(150):
        u = c.UMAX * torch.sigmoid(a)
        useq = torch.cat([torch.full((1, WARM), c.U_OP), u], 1)
        yp = m.roll(y0, nu(useq), WARM)[:, WARM - 1:WARM - 1 + HPLAN]
        lossp = ((yp - target) ** 2).mean(); popt.zero_grad(); lossp.backward(); popt.step()
    plan = (c.UMAX * torch.sigmoid(a)).detach().numpy()[0]
    x = np.array([c.TREF, c.XM_OP]); ytrue = np.empty(HPLAN)
    for t in range(HPLAN):
        ytrue[t] = x[0]; x = c.Ad @ x + c.Bd[:, 0] * plan[t]
    track = float(np.sqrt(np.mean((ytrue - c.TREF - TARGET) ** 2)))
    return dict(eff=eff, plan=plan, track=track, model_track=float(np.sqrt(lossp.item()) * sy), loss=float(loss.item()))


def optimal_track():
    """reference: the same planner run on the TRUE plant (noise-free)."""
    from scipy.optimize import minimize
    def f(a):
        u = c.UMAX / (1 + np.exp(-a)); x = np.array([c.TREF, c.XM_OP]); e = 0.0
        for t in range(HPLAN): e += (x[0] - c.TREF - TARGET) ** 2; x = c.Ad @ x + c.Bd[:, 0] * u[t]
        return e / HPLAN
    r = minimize(f, np.zeros(HPLAN), method="L-BFGS-B", options=dict(maxiter=300))
    return float(np.sqrt(r.fun))


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    N = 200000; regimes = {"open": openloop(N, 3)}
    for d in (0.0, 0.8):
        for sr in (0.0, 1.0):
            Y, U, R, Q = c.simulate(N, 2.0, d, sr, 0.01, seed=7); regimes[f"cl_d{d}_sr{sr}"] = (Y, U)
    jobs, meta = [], []
    for reg, (Y, U) in regimes.items():
        for kind in ("gru_sim", "gru_1s", "tf_1s"):
            for s in range(3): jobs.append((kind, Y, U, s, steps)); meta.append((reg, kind, s))
    t0 = time.time()
    with Pool(22) as p: res = p.map(train_eval, jobs)
    sc = np.sqrt(np.mean(c.EFF_TRUE ** 2)); rows = []
    for (reg, kind, s), r in zip(meta, res):
        rows.append(dict(regime=reg, model=kind, seed=s, eff_err=c.rel_err(r["eff"]), track=r["track"], model_track=r["model_track"],
                         loss=r["loss"], eff=r["eff"][::10].round(3).tolist(), plan=r["plan"][::5].round(3).tolist()))
        print(f"{reg:16s} {kind:8s} s{s}: effect err {rows[-1]['eff_err']:.2f}  plan→true-plant tracking RMSE {r['track']:.2f} K "
              f"(model thinks {r['model_track']:.2f})", flush=True)
    print("planner on true plant:", round(optimal_track(), 3), "K;  doing nothing:", TARGET, "K;  elapsed", int(time.time() - t0), "s")
    json.dump(rows, open("c5_results.json", "w"), indent=1)
