#!/usr/bin/env python3
"""ED Fig 11 — full 48-h trajectories: emulator truth, −1/Ĉ (fitted embedded controller), operational-data twins (mean over seeds), per period."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from figures import C, TXT, TXT2, GRID, grid, save
import c19_inverse_controller as I
FAM = [('gru_sim', 'GRU, simulation-error', C['operational'], '-'), ('tf_1s_noise', 'Transformer, next-step + noise', C['fm'], '--'), ('tf_1s', 'Transformer, next-step', C['fm'], '-'), ('arx48', 'ARX(48)', C['designed'], '-'), ('pem2', 'PEM state-space', C['designed'], '--')]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12)); h = np.arange(1, 193) / 4
for ax, P in zip(axs, ('P1', 'P2', 'P3')):
    L = np.load(f'c7_v1/{P}/log_operational.npy'); a, b, r2 = I.fit_controller(L); gop = I.inv_ctrl_step(a, b)
    tr = np.mean([t['effect'] for t in json.load(open(f'c7_v1/{P}/truth_step.json'))], 0); R = json.load(open(f'c7_v1/{P}/c18_results.json'))
    ax.plot(h, tr, color=TXT, lw=1.6, label='emulator (truth)'); ax.plot(h, gop, color=C['ref'], lw=1.4, label=r'$-1/\hat C$ (controller inverse)')
    for fam, lab, col, ls in FAM:
        rows = [r for r in R if r['family'] == fam and r['train'] == 'log_operational']
        if rows: ax.plot(h, np.mean([r['step']['learned'] for r in rows], 0), color=col, ls=ls, lw=1.2, label=lab)
    ax.axhline(0, color=GRID, lw=0.8); ax.set_title(f'period {P[1]}', fontsize=8, loc='left', color=TXT); ax.set_xticks([0, 6, 12, 24, 48]); grid(ax)
axs[0].set_ylabel('zone-temperature response (K)'); axs[1].set_xlabel('hours after the +0.3 modulation step'); axs[-1].legend(frameon=False, fontsize=5.5, loc='upper left')
save(fig, 'v1_figED11_invctrl')
