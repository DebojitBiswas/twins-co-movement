#!/usr/bin/env python3
"""v1 figures from c7_v1/summary.json (analyze_v1.py) and the referee-driven result files.  Palette validated (dataviz, light).
Usage: python3 figures_v1.py [figA figB ...]  -> figs/v1_*.pdf/png"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures import C, TXT, TXT2, GRID, grid, panel_label, save

FAM = [('gru_sim', 'GRU, simulation-error', C['operational'], 'o'), ('gru_1s_noise', 'GRU, next-step + noise', C['operational'], 'v'),
       ('tf_1s_noise', 'Transformer, next-step + noise', C['fm'], 'v'), ('tf_1s', 'Transformer, next-step', C['fm'], 's'),
       ('gru_1s', 'GRU, next-step', C['operational'], '^'), ('arx48', 'ARX(48), classical', C['designed'], 'D'), ('pem2', 'PEM state-space, classical', C['designed'], 'P'),
       ('grey', 'grey-box RC, free gain', C['ref'], 'X')]
S = json.load(open('c7_v1/summary.json'))
PERIODS = sorted({x['period'] for x in S})


def get(fam, reg, P):
    e = [x for x in S if x['period'] == P and x['family'] == fam and (x['regime'] == reg or (reg == 'dither' and x['regime'].startswith('dither A2')))]
    return e


# ---------------------------------------------------------------- Fig A: the accuracy–fidelity trade-off (operational log)
def figA():
    fig, axs = plt.subplots(1, len(PERIODS), figsize=(2.4 * len(PERIODS) + 0.6, 2.6), sharey=True, gridspec_kw=dict(wspace=0.15))
    axs = np.atleast_1d(axs)
    for ax, P in zip(axs, PERIODS):
        pers = None
        for fam, lab, col, mk in FAM:
            e = get(fam, 'operational', P)
            if not e: continue
            e = e[0]; pers = pers or None
            ax.errorbar(e['pred'], e['ratio'], yerr=[[e['ratio'] - e['ratio_ci'][0]], [e['ratio_ci'][1] - e['ratio']]], fmt=mk, ms=6, color=col, ecolor=col, elinewidth=0.8, capsize=2, label=lab if P == PERIODS[0] else None)
        rows = json.load(open(f'c7_v1/{P}/c18_results.json' if os.path.exists(f'c7_v1/{P}/c18_results.json') else f'c7_v1/{P}/c15_results.json')); p = rows[0]['test']['test_operational']['persistence']
        ax.axvline(p, color=TXT2, lw=0.8, ls=':'); ax.text(p + 0.02, 0.05, 'persistence', color=TXT2, fontsize=6, rotation=90, va='bottom')
        ax.axhline(1.0, color=GRID, lw=1.0); ax.set_xlim(0.05, 1.6); ax.set_ylim(0, 2.5); ax.set_xticks([0.1, 0.5, 1.0, 1.5])
        ax.set_xlabel('error on operational test weeks (K)'); ax.set_title(f'period {P[1]}', fontsize=8); grid(ax); panel_label(ax, 'abc'[PERIODS.index(P)])
    axs[0].set_ylabel('learned / true heating effect'); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.24), handlelength=1.0, columnspacing=1.2)
    save(fig, 'v1_figA_tradeoff')


# ---------------------------------------------------------------- Fig B: what the dither buys, by family and period (3 independent logs)
def figB():
    fams = [f for f in FAM if f[0] in ('gru_sim', 'tf_1s', 'tf_1s_noise', 'gru_1s_noise', 'arx48', 'pem2')]
    fig, ax = plt.subplots(figsize=(5.6, 2.6)); x = np.arange(len(fams)); w = 0.8 / (2 * len(PERIODS))
    for i, (fam, lab, col, mk) in enumerate(fams):
        for j, P in enumerate(PERIODS):
            op = get(fam, 'operational', P); di = get(fam, 'dither', P)
            if op: ax.bar(x[i] + (2 * j - len(PERIODS) + 0.5) * w, op[0]['ratio'], w - 0.02, color=col, alpha=0.45, edgecolor='white', lw=0.5, label='operational log' if (i == 0 and j == 0) else None)
            if di: ax.bar(x[i] + (2 * j - len(PERIODS) + 1.5) * w, np.mean([d['ratio'] for d in di]), w - 0.02, color=col, edgecolor='white', lw=0.5, label='operational + ±2 K dither (3 logs)' if (i == 0 and j == 0) else None); ax.plot([x[i] + (2 * j - len(PERIODS) + 1.5) * w] * len(di), [d['ratio'] for d in di], 'o', ms=2, color=TXT, alpha=0.7)
    ax.axhline(1.0, color=GRID, lw=1.0); ax.set_xticks(x); ax.set_xticklabels(['GRU\nsim.-error', 'GRU\nnext-step\n+ noise', 'Transf.\nnext-step\n+ noise', 'Transf.\nnext-step', 'ARX(48)', 'PEM\nstate-space'], fontsize=6); ax.set_ylabel('learned / true heating effect'); ax.set_ylim(0, 2.4); grid(ax)
    ax.legend(frameon=False, fontsize=6, loc='upper right'); ax.text(0.01, 0.97, 'bars left→right: periods ' + ', '.join(p[1] for p in PERIODS), transform=ax.transAxes, va='top', fontsize=6, color=TXT2)
    save(fig, 'v1_figB_dither')


# ---------------------------------------------------------------- Fig C: decisions (MPC + pancreas dosing v2)
def figC():
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5), gridspec_kw=dict(wspace=0.4, width_ratios=[1, 1.15]))
    ax = axs[0]; mpc = json.load(open('c7_data/c12_mpc.json' if os.path.exists('c7_data/c12_mpc.json') else 'c7_data/c12_mpc_v0override.json')); emb = np.mean([e['track'] for e in mpc['embedded']])
    for tr, lab, col in (('operational_A0.0_s0', 'operational twin', C['operational']), ('dither_A2.0_s0', 'dithered twin', C['dither']), ('designed_A0.0_s0', 'designed twin', C['designed'])):
        lams = sorted({r['lam'] for r in mpc['rows']}); v = [np.mean([r['track'] for r in mpc['rows'] if r['train'] == tr and r['lam'] == l]) for l in lams]
        s = [np.std([r['track'] for r in mpc['rows'] if r['train'] == tr and r['lam'] == l]) / np.sqrt(5) for l in lams]
        ax.errorbar(range(len(lams)), v, yerr=s, fmt='o-', ms=4, lw=1.3, color=col, capsize=2, label=lab)
    ax.axhline(emb, color=TXT2, lw=0.8, ls=':'); ax.text(-0.1, emb - 0.05, "building's own controller", color=TXT2, fontsize=6, ha='left', va='top')
    ax.set_xticks(range(4)); ax.set_xticklabels(['0', '1', '3', '10']); ax.set_xlabel('pessimism penalty λ'); ax.set_ylabel('MPC tracking error (K)'); ax.set_ylim(0, 2.0); grid(ax); ax.legend(frameon=False, fontsize=6, loc='upper right'); panel_label(ax, 'a')
    ax = axs[1]; d = json.load(open('c6f_dosing_regen.json'))['dosing']
    labs = [('true effect (at operating point)', 'reference', C['designed']), ('operational twin d=0.0', 'twin\nreactive\nloop', C['operational']), ('operational twin d=0.7', 'twin\nbolused\nloop', C['operational']), ('IV twin d=0.0', 'dither+IV\nreactive', C['dither']), ('IV twin d=0.7', 'dither+IV\nbolused', C['dither'])]
    for i, (k, lab, col) in enumerate(labs):
        S_ = [x for x in d if x['label'] == k and 'error' not in x]; sev = sum(x['severe'] for x in S_)
        ax.bar(i, sev / len(S_), 0.6, color=col); ax.text(i, sev / len(S_) + 0.02, f'{sev}/{len(S_)}', ha='center', fontsize=6, color=TXT2)
    ax.set_xticks(range(5)); ax.set_xticklabels([l[1] for l in labs], fontsize=5.0); ax.set_ylabel('patients with severe hypoglycaemia'); ax.set_ylim(0, 0.85); grid(ax); panel_label(ax, 'b')
    save(fig, 'v1_figC_decisions')


# ---------------------------------------------------------------- Fig D: within-architecture ablation (clean vs noise injection)
def figD():
    """Within-architecture ablation with the referee-requested additions: 256-unit and recipe-selected clean GRUs, operational error under every bar."""
    H = json.load(open('c7_v1/P1/c17_hp_sweep.json'))
    def selected(kind):
        recs = sorted({(r['h'], r['lr'], r['steps']) for r in H})
        des = {rc: np.mean([r['in_rmse'] for r in H if r['kind'] == kind and r['train'].startswith('log_designed') and (r['h'], r['lr'], r['steps']) == rc]) for rc in recs}
        rc = min(des, key=des.get); rs = [r for r in H if r['kind'] == kind and r['train'] == 'log_operational' and (r['h'], r['lr'], r['steps']) == rc]
        v = [r['ratio'] for r in rs]; return dict(ratio=float(np.mean(v)), ratio_ci=[min(v), max(v)], pred=float(np.mean([r['pred'] for r in rs])))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.08), sharey=True)
    groups = [('Transformer', C['fm'], [('tf_1s', 'next-step', 1.0, ''), ('tf_1s_noise', 'next-step + noise', 0.45, '////'), ('tf_1s_big', '4-layer, next-step', 1.0, '..'), ('sel', 'selected recipe', 1.0, 'xx')]),
              ('GRU', C['operational'], [('gru_1s', 'next-step', 1.0, ''), ('gru_1s_noise', 'next-step + noise', 0.45, '////'), ('gru_1s_big', '256 units, next-step', 1.0, '..'), ('sel', 'selected recipe', 1.0, 'xx'), ('gru_sim', 'simulation error', 0.75, '\\\\')])]
    x = np.arange(len(PERIODS)); cap = 1.55
    for ax, (lab, col, bars) in zip(axs, groups):
        w = 0.8 / len(bars)
        for k, (fam, blab, alpha, hatch) in enumerate(bars):
            for j, P in enumerate(PERIODS):
                if fam == 'sel':
                    if P != 'P1': continue
                    e = selected('tf_1s' if lab == 'Transformer' else 'gru_1s')
                else:
                    e = get(fam, 'operational', P)
                    if not e: continue
                    e = e[0]
                xx = x[j] + (k - (len(bars) - 1) / 2) * w; v = min(e['ratio'], cap)
                ax.bar(xx, v, w - 0.02, color=col, alpha=alpha, hatch=hatch, edgecolor='white', lw=0.5, label=blab if j == 0 else None)
                if e['ratio'] > cap: ax.text(xx, cap + 0.02, f"{e['ratio']:.1f}", ha='center', va='bottom', fontsize=5, color=TXT2)
                else: ax.errorbar(xx, e['ratio'], yerr=[[e['ratio'] - e['ratio_ci'][0]], [e['ratio_ci'][1] - e['ratio']]], fmt='none', ecolor=TXT, elinewidth=0.7, capsize=1.5)
                ax.text(xx, -0.06, f"{e['pred']:.2f}", ha='center', va='top', fontsize=4.6, color=TXT2, rotation=90)
        ax.axhline(1.0, color=GRID, lw=1.0); ax.set_xticks(x); ax.set_xticklabels([f'period {P[1]}' for P in PERIODS]); ax.tick_params(axis='x', pad=22)
        ax.set_ylim(-0.02, 1.75); ax.set_title(lab, fontsize=8, color=TXT, loc='left', pad=30); grid(ax)
        ax.legend(frameon=False, fontsize=5.5, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=0.8, handlelength=1.2)
    axs[0].set_ylabel('learned / true heating effect'); axs[0].text(-0.62, -0.06, 'op. error (K)', fontsize=4.6, color=TXT2, ha='right', va='top', rotation=90)
    save(fig, 'v1_figD_ablation')


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['figA', 'figB', 'figC', 'figD']): globals()[name]()
