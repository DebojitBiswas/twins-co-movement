#!/usr/bin/env python3
"""Paper C figures (NMI).  One function per figure; data read from the result JSONs; palette validated (dataviz validator, light mode).
Usage: python3 figures.py [fig1 fig2 fig4 ...]   -> figs/figN.pdf + .png"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = dict(designed='#2a78d6', operational='#eb6834', dither='#1baf7a', fm='#eda100', iv='#4a3aa7', truth='#0b0b0b', ref='#9a9a95')
TXT, TXT2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.edgecolor': TXT2, 'axes.linewidth': 0.6, 'xtick.color': TXT2, 'ytick.color': TXT2, 'text.color': TXT, 'axes.labelcolor': TXT,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
os.makedirs('figs', exist_ok=True)


def save(fig, name):
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight'); fig.savefig(f'figs/{name}.png', bbox_inches='tight'); plt.close(fig); print('saved', name)


def panel_label(ax, s): ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')


def grid(ax): ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1: the failure (synthetic): families, scaling, sign, uncertainty
def fig1():
    import c1_synthetic as c1
    sub = json.load(open('c1_results_gru_sub.json')); ctl = json.load(open('c1b_results.json')); fam = json.load(open('c5c_results.json')) + json.load(open('c5_results.json'))
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.6))
    # a: learned effect trajectories (GRU, closed loop kp2 d0.8? use sub cells) vs truth
    ax = axs[0]; t = np.arange(0, c1.H, 10) / 60
    ax.plot(np.arange(c1.H) / 60, c1.EFF_TRUE, color=C['truth'], lw=1.6, label='true effect')
    for r in [x for x in ctl if x['regime'] == 'open' and x['N'] == 200000][:1]:
        ax.plot(t, r['eff_mean'], color=C['designed'], lw=1.4, label='GRU, designed data')
    for r in [x for x in sub if x['kp'] == 8.0 and x['d'] == 0.0 and x['sigma_r'] == 0.0 and x['N'] == 200000 and x['seed'] == 0]:
        ax.plot(t, r['eff_mean'], color=C['operational'], lw=1.4, label='GRU, operational data')
    ax.axhline(0, color=GRID, lw=0.6); ax.set_xlabel('time after intervention (h)'); ax.set_ylabel('effect on temperature (K)'); grid(ax)
    ax.legend(frameon=False, loc='upper right', handlelength=1.4, fontsize=6, bbox_to_anchor=(1.02, 1.0)); panel_label(ax, 'a')
    # b: scaling with N (GRU closed loop, 4 regimes) + open loop
    ax = axs[1]
    for d, sr, lab, col, ls in ((0.0, 0.0, 'operational, feedback', C['operational'], '-'), (0.8, 0.0, 'operational, anticipat.', C['operational'], '--'), (0.0, 0.2, 'operational, dithered', C['dither'], '-')):
        pts = sorted([(r['N'], r['gru']) for r in sub if r['kp'] == 8.0 and r['d'] == d and r['sigma_r'] == sr])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=3.5, lw=1.4, color=col, ls=ls, label=lab)
    pts = sorted([(r['N'], r['gru']) for r in ctl if r['regime'] == 'open']); ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=3.5, lw=1.4, color=C['designed'], label='designed')
    ax.set_xscale('log'); ax.set_xlabel('training samples'); ax.set_ylabel('interventional error (rel.)'); ax.set_ylim(0, 1.4); grid(ax)
    ax.legend(frameon=False, loc='center left', fontsize=6, handlelength=1.6, bbox_to_anchor=(0.0, 0.62)); panel_label(ax, 'b')
    # c: model families (c5c + c5 gru_sim) closed vs open
    ax = axs[2]; fams = [('gru_sim', 'GRU\nsim.'), ('gru_1s', 'GRU\n1-step'), ('tf_1s', 'Transf.')]
    w = 0.36; x = np.arange(3)
    for i, (k, lab) in enumerate(fams):
        src = [r for r in fam if r['model'] == k]
        for j, (reg, col) in enumerate((('open', C['designed']), ('cl_d0.8_sr0.0', C['operational']))):
            v = [r['eff_err'] for r in src if r['regime'] == reg]
            if k != 'gru_sim': v = [r['eff_err'] for r in json.load(open('c5c_results.json')) if r['model'] == k and r['regime'] == reg]
            ax.bar(x[i] + (j - 0.5) * w, np.median(v), w - 0.04, color=col, label=('designed data' if j == 0 else 'operational data') if i == 0 else None)
            ax.plot([x[i] + (j - 0.5) * w] * len(v), v, 'o', ms=2.5, color=TXT, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f[1] for f in fams]); ax.set_ylabel('interventional error (rel.)'); ax.set_ylim(0, 1.6); grid(ax); ax.legend(frameon=False, loc='upper left', fontsize=6); panel_label(ax, 'c')
    # d: ensemble spread vs error
    ax = axs[3]
    for r in sub: ax.plot(r['gru_spread'], r['gru'], 'o', ms=3.5, color=C['operational'], alpha=0.8)
    for r in [x for x in ctl if x['regime'] == 'open']: ax.plot(r['spread'], r['gru'], 'o', ms=3.5, color=C['designed'])
    lim = 1.4; ax.plot([0, lim], [0, lim], color=GRID, lw=0.8); ax.set_xlim(0, 0.3); ax.set_ylim(0, lim)
    ax.set_xlabel('ensemble spread (rel.)'); ax.set_ylabel('interventional error (rel.)'); grid(ax); panel_label(ax, 'd')
    ax.text(0.29, 0.95, 'operational', color=C['operational'], ha='right', fontsize=6.5); ax.text(0.29, 0.5, 'designed', color=C['designed'], ha='right', fontsize=6.5)
    save(fig, 'fig1')


# ---------------------------------------------------------------- Fig 2: the law and its collapse
def fig2():
    import c1_synthetic as c1
    col = json.load(open('c2_collapse.json'))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]; sc = np.sqrt(np.mean(c1.EFF_TRUE ** 2)); t = np.arange(0, c1.H, 10) / 60
    r = [x for x in col if x['kp'] == 8.0 and x['d'] == 0.0 and x['sr'] == 0.0 and x['N'] == 200000][0]
    ax.plot(np.arange(c1.H) / 60, c1.EFF_TRUE, color=C['truth'], lw=1.6, label='true effect  G')
    ax.plot(t, r['pred'], color=C['ref'], lw=2.2, label='law  γ²G + (1−γ²)G$_{op}$')
    ax.plot(t, r['gru'], color=C['operational'], lw=1.3, label='GRU, operational data')
    ax.plot(t, r['oe'], color=C['iv'], lw=1.1, ls='--', label='linear sim-error twin')
    ax.axhline(0, color=GRID, lw=0.6); ax.set_xlabel('time after intervention (h)'); ax.set_ylabel('effect (K)'); grid(ax); ax.legend(frameon=False, loc='upper right', handlelength=1.4, fontsize=6, bbox_to_anchor=(1.02, 1.0)); panel_label(ax, 'a')
    ax = axs[1]
    for r in col:
        ax.plot(np.array(r['pred'])[::6] / sc, np.array(r['gru'])[::6] / sc, 'o', ms=2.2, color=C['operational'], alpha=0.5)
    ax.plot([-1.5, 1.5], [-1.5, 1.5], color=GRID, lw=0.8); ax.set_xlim(-1.5, 1.2); ax.set_ylim(-1.5, 1.2)
    ax.set_xlabel('law-predicted effect (rel.)'); ax.set_ylabel('GRU-learned effect (rel.)'); grid(ax); panel_label(ax, 'b')
    ax.text(0.03, 0.92, fontsize=6.5, s=f"28 cells: error vs truth {np.median([r['gru_vs_truth'] for r in col]):.2f}\n              vs law   {np.median([r['gru_vs_law'] for r in col]):.2f}", transform=ax.transAxes, va='top', color=TXT2)
    ax = axs[2]
    gm = np.array([[0.19, 0.46, 0.54, 0.80], [0.49, 0.65, 0.67, 0.91]]); g2 = np.array([0, 0.43, 0.62, 0.75])
    ax.plot(g2, gm[0], 'o-', color=C['dither'], lw=1.4, ms=4, label='GRU (simulation)'); ax.plot(g2, gm[1], 's-', color=C['fm'], lw=1.4, ms=4, label='Transformer')
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8); ax.set_xlim(-0.03, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('coherence γ² with logged dither'); ax.set_ylabel('learned / true heating effect'); grid(ax); ax.legend(frameon=False, loc='lower right'); panel_label(ax, 'c')
    ax.text(0.02, 0.97, 'building emulator,\ndither 0 / 0.5 / 1 / 2 K', transform=ax.transAxes, va='top', color=TXT2)
    save(fig, 'fig2')


# ---------------------------------------------------------------- Fig 4: pancreas
def fig4():
    rows = json.load(open('c6_results.json')); dos = json.load(open('c6c_dosing.json'))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]; t = np.arange(72) / 12
    R = [r for r in rows if r['d'] == 0.7 and r['A'] == 0.0]
    for i, r in enumerate(R):
        ax.plot(t, r['true'], color=C['truth'], lw=0.8, alpha=0.6, label='true (10 patients)' if i == 0 else None)
        ax.plot(t, r['lin'], color=C['operational'], lw=0.8, alpha=0.7, label='twin, pump log' if i == 0 else None)
    ax.axhline(0, color=GRID, lw=0.6); ax.set_xlabel('hours after 1-U bolus'); ax.set_ylabel('effect on glucose (mg/dL)'); grid(ax); ax.legend(frameon=False, loc='lower left'); panel_label(ax, 'a')
    ax = axs[1]
    for d, col, lab in ((0.0, C['dither'], 'reactive loop'), (0.7, C['operational'], 'meal-bolused loop')):
        R = [r for r in rows if r['d'] == d and r['A'] == 0.0 and r['hypo'] < 0.5]
        ax.plot([r['true'][47] for r in R], [r['lin'][47] for r in R], 'o', ms=4, color=col, label=lab)
    ax.plot([-45, 5], [-45, 5], color=GRID, lw=0.8); ax.axhline(0, color=GRID, lw=0.6); ax.set_xlim(-45, 5); ax.set_ylim(-20, 15)
    ax.set_xlabel('true 4-h effect (mg/dL)'); ax.set_ylabel('twin 4-h effect (mg/dL)'); grid(ax); ax.legend(frameon=False, loc='upper left'); panel_label(ax, 'b')
    ax = axs[2]; labs = [('true (designed-data reference)', 'ref.', C['designed']), ('operational twin d=0.0', 'reactive\ntwin', C['dither']), ('operational twin d=0.7', 'bolused\ntwin', C['operational']), ('IV twin (dither) d=0.7', 'bolused\nIV twin', C['iv'])]
    for i, (k, lab, col) in enumerate(labs):
        rr = [r['results'][k] for r in dos if k in r['results']]
        ax.bar(i, np.mean([x['dose'] for x in rr]), 0.6, color=col); ax.plot([i] * len(rr), [x['dose'] for x in rr], 'o', ms=2.5, color=TXT, alpha=0.6)
        ax.text(i, 10.6, f"{sum(x['severe'] for x in rr)}/{len(rr)}", ha='center', va='bottom', fontsize=6.5, color=TXT2)
    ax.text(1.5, 12.0, 'severe hypoglycaemia:', ha='center', va='bottom', fontsize=6.5, color=TXT2)
    ax.set_xticks(range(4)); ax.set_xticklabels([l[1] for l in labs], fontsize=6); ax.set_ylabel('correction dose (U)'); ax.set_ylim(0, 13.5); grid(ax); panel_label(ax, 'c')
    save(fig, 'fig4')




# ---------------------------------------------------------------- Fig 3: uncertainty and decisions
def fig3():
    cf = json.load(open('c7_data/c8_conformal.json')); pl = json.load(open('c7_data/c7b_results.json')); nr = json.load(open('c7_data/c7_neural.json'))
    fam = json.load(open('c5_results.json')); fam2 = json.load(open('c5c_results.json'))
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.55)); axs = axs.ravel()
    # a: conformal coverage (nominal 0.9): operational test vs interventional test, operational-data vs designed-data twins (GRU+TF pooled)
    ax = axs[0]; bands = ['0-6h', '6-24h', '24-48h']; x = np.arange(3); w = 0.2
    for j, (reg, test, col, lab, hatch) in enumerate((('operational', 'operational', C['operational'], 'operational twin — operational test', ''), ('operational', 'designed', C['operational'], 'operational twin — intervention', '////'),
                                                     ('designed', 'operational', C['designed'], 'designed twin — operational test', ''), ('designed', 'designed', C['designed'], 'designed twin — intervention', '////'))):
        v = []
        for b in bands:
            if test == 'operational': v.append(np.mean([r['coverage']['operational'][b] for r in cf if r['regime'] == reg]))
            else: v.append(np.mean([r['coverage'][k][b] for r in cf if r['regime'] == reg for k in ('designed_s10', 'designed_s11')]))
        ax.bar(x + (j - 1.5) * w, v, w - 0.03, color=col, alpha=1.0 if not hatch else 0.55, hatch=hatch, edgecolor='white', lw=0.5, label=lab)
    ax.axhline(0.9, color=TXT2, lw=0.8, ls=':'); ax.text(2.65, 0.92, 'nominal', color=TXT2, fontsize=6, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(['0–6 h', '6–24 h', '24–48 h']); ax.set_xlabel('forecast horizon'); ax.set_ylabel('coverage of 90% intervals'); ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); grid(ax)
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1, handlelength=1.2, bbox_to_anchor=(-0.02, 1.03), labelspacing=0.25); panel_label(ax, 'a')
    # b: planning in the heat-pump building
    ax = axs[1]; emb = np.mean(pl['embedded'])
    order = [('designed_A0.0_s0', 'designed'), ('dither_A2.0_s0', 'dither 2 K'), ('dither_A1.0_s0', 'dither 1 K'), ('dither_A0.5_s0', 'dither 0.5 K'), ('operational_A0.0_s0', 'operational')]
    cols = [C['designed'], C['dither'], C['dither'], C['dither'], C['operational']]
    for i, (tr, lab) in enumerate(order):
        for j, kind in enumerate(('gru_sim', 'tf_1s')):
            R = [r for r in pl['rows'] if r['train'] == tr and r['kind'] == kind and not r['iv']]
            t = [np.mean([q['track'] for q in r['plans']]) for r in R]; b = [np.mean([q['believed'] for q in r['plans']]) for r in R]
            xx = i + (j - 0.5) * 0.36
            ax.bar(xx, np.mean(t), 0.32, color=cols[i], alpha=1.0 if j == 0 else 0.6, edgecolor='white', lw=0.5)
            ax.plot([xx], [np.mean(b)], marker='_', ms=7, mew=1.2, color=TXT)
    ax.axhline(emb, color=TXT2, lw=0.8, ls=':'); ax.text(-0.45, emb + 0.06, "building's own controller", color=TXT2, fontsize=5.5, ha='left')
    ax.set_xticks(range(5)); ax.set_xticklabels([o[1] for o in order], rotation=25, ha='right', fontsize=6); ax.set_ylabel('tracking error of plan (K)'); ax.set_ylim(0, 3.6); grid(ax); panel_label(ax, 'b')
    ax.text(0.02, 0.98, 'bars: plan applied to the building\n(GRU dark, Transformer light)\nticks: error the twin believes', transform=ax.transAxes, va='top', fontsize=5.3, color=TXT2)
    # c: overshoot law
    ax = axs[2]
    def step(kind, train, h=191): return np.mean([np.mean(r['step_resp'], 0)[h] for r in nr if r['kind'] == kind and r['train'] == train])
    def meanu(kind, train): return np.mean([q['energy'] for r in pl['rows'] if r['kind'] == kind and r['train'] == train and not r['iv'] for q in r['plans']])
    for kind, mk, lab in (('gru_sim', 'o', 'GRU'), ('tf_1s', 's', 'Transformer')):
        gd = np.mean([step(kind, 'designed_A0.0_s0'), step(kind, 'designed_A0.0_s1')]); ud = meanu(kind, 'designed_A0.0_s0')
        for tr, col in (('operational_A0.0_s0', C['operational']), ('dither_A0.5_s0', C['dither']), ('dither_A1.0_s0', C['dither']), ('dither_A2.0_s0', C['dither'])):
            ax.plot(gd / step(kind, tr), meanu(kind, tr) / ud, marker=mk, ms=5, color=col, ls='none', label=lab if tr == 'operational_A0.0_s0' else None)
    ax.plot([1, 8], [1, 8], color=GRID, lw=0.8); ax.set_xlim(1, 8.2); ax.set_ylim(1, 3)
    ax.set_xlabel('overshoot predicted by the identity'); ax.set_ylabel('observed heating ratio of the plan'); grid(ax); ax.legend(frameon=False, fontsize=6, loc='lower right'); panel_label(ax, 'c')
    ax.text(0.03, 0.97, 'plans saturate at full power\n(orange: operational; green: dithered)', transform=ax.transAxes, va='top', fontsize=5.5, color=TXT2)
    # d: synthetic planning: plan tracking on true plant vs believed, closed loop, 3 families
    ax = axs[3]
    for k, mk, lab in (('gru_sim', 'o', 'GRU sim.'), ('gru_1s', '^', 'GRU 1-step'), ('tf_1s', 's', 'Transformer')):
        src = fam if k == 'gru_sim' else fam2
        R = [r for r in src if r['model'] == k and r['regime'] != 'open']
        ax.plot([r['model_track'] for r in R], [r['track'] for r in R], marker=mk, ms=4, color=C['operational'], ls='none', alpha=0.8, label=lab)
        R = [r for r in src if r['model'] == k and r['regime'] == 'open']
        ax.plot([r['model_track'] for r in R], [r['track'] for r in R], marker=mk, ms=4, color=C['designed'], ls='none', alpha=0.8)
    ax.axhline(3.0, color=TXT2, lw=0.8, ls=':'); ax.text(0.12, 3.4, 'doing nothing', color=TXT2, fontsize=6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.1, 10); ax.set_ylim(0.2, 40); ax.plot([0.1, 40], [0.1, 40], color=GRID, lw=0.8)
    ax.set_xlabel('tracking error the twin believes (K)'); ax.set_ylabel('tracking error on the true plant (K)'); grid(ax); ax.legend(frameon=False, fontsize=6, loc='lower right'); panel_label(ax, 'd')
    ax.text(0.03, 0.97, 'synthetic plant; orange: operational\ndata, blue: designed data', transform=ax.transAxes, va='top', fontsize=5.5, color=TXT2)
    save(fig, 'fig3')


# ---------------------------------------------------------------- Fig 5: what helps
def fig5():
    ud = json.load(open('c7_data/c9_updown.json')); pl = json.load(open('c7_data/c7b_results.json')); fm = json.load(open('c7_data/c11_foundation.json')); cf = json.load(open('c7_data/c8_conformal.json'))
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.55)); axs = axs.ravel()
    # a: upstream vs downstream relative effect error
    ax = axs[0]; items = [('A', 'CL', 'closed-loop twin', C['designed']), ('A', 'AC+CT', 'chained twins', C['iv']), ('B', 'AC', 'actuator twin', C['operational']), ('B', 'AC+CT', 'chained twins', C['iv']), ('B', 'CL', 'closed-loop twin', C['designed'])]
    for i, (q, m, lab, col) in enumerate(items):
        eff = np.mean([np.mean(r[f'eff{q}_rms']) for r in ud]); v = [np.mean(r[f'eff{q}'][m]) / eff for r in ud]
        ax.bar(i, np.mean(v), 0.6, color=col); ax.plot([i] * len(v), v, 'o', ms=2.5, color=TXT, alpha=0.6)
    ax.axvline(1.5, color=GRID, lw=0.8); ax.text(0.5, 1.12, 'set-point what-if', ha='center', fontsize=6.5, color=TXT2); ax.text(3, 1.12, 'actuator what-if', ha='center', fontsize=6.5, color=TXT2)
    ax.set_xticks(range(5)); ax.set_xticklabels(['closed-\nloop twin', 'chained\ntwins', 'actuator\ntwin', 'chained\ntwins', 'closed-\nloop twin'], fontsize=5.8); ax.set_ylabel('relative error of predicted effect'); ax.set_ylim(0, 1.25); grid(ax); panel_label(ax, 'a')
    # b: excitation budget
    ax = axs[1]; emb = np.mean(pl['embedded']); op = np.load('c7_data/log_operational_A0.0_s0.npy')
    for kind, mk, lab in (('gru_sim', 'o', 'GRU'), ('tf_1s', 's', 'Transformer')):
        xs, ys = [0.0], [np.mean([np.mean([q['track'] for q in r['plans']]) for r in pl['rows'] if r['kind'] == kind and r['train'] == 'operational_A0.0_s0' and not r['iv']])]
        for A in (0.5, 1.0, 2.0):
            L = np.load(f'c7_data/log_dither_A{A}_s0.npy'); xs.append(np.sqrt(np.mean((L[:, 0] - op[:, 0]) ** 2)))
            ys.append(np.mean([np.mean([q['track'] for q in r['plans']]) for r in pl['rows'] if r['kind'] == kind and r['train'] == f'dither_A{A}_s0' and not r['iv']]))
        ax.plot(xs, ys, marker=mk, ms=4.5, lw=1.4, color=C['dither'], label=lab, alpha=1.0 if kind == 'gru_sim' else 0.6)
        for xx, yy, A in zip(xs, ys, (0, 0.5, 1, 2)):
            if kind == 'gru_sim': ax.text(xx + 0.03, yy + 0.12, f'dither ±{A} K' if A else 'no dither', fontsize=5.5, color=TXT2)
    ax.axhline(emb, color=TXT2, lw=0.8, ls=':'); ax.text(0.01, emb - 0.1, "building's own controller", color=TXT2, fontsize=5.5, ha='left', va='top')
    ax.set_xlabel('comfort cost of dither (K rms)'); ax.set_ylabel('plan tracking error (K)'); ax.set_ylim(0, 3.3); grid(ax); ax.legend(frameon=False, fontsize=6, loc='upper right'); panel_label(ax, 'b')
    # c: Chronos-2
    ax = axs[2]; labs = [('zero-shot', 'zero-\nshot', C['fm']), ('ft:operational_A0.0_s0', 'fine-tuned\noperational', C['operational']), ('ft:dither_A2.0_s0', 'fine-tuned\ndither 2 K', C['dither']), ('ft:designed_A0.0_s0', 'fine-tuned\ndesigned', C['designed'])]
    pers = np.mean([fm[0]['persist_10'], fm[0]['persist_11']])
    for i, (k, lab, col) in enumerate(labs):
        R = [r for r in fm if r['model'] == k]; v = [np.mean([r['do_rmse_10'], r['do_rmse_11']]) for r in R]; p = [r['pred_rmse_operational'] for r in R]
        ax.bar(i - 0.18, np.mean(v), 0.34, color=col, label='intervention' if i == 0 else None); ax.bar(i + 0.18, np.mean(p), 0.34, color=col, alpha=0.45, label='operation' if i == 0 else None)
        ax.text(i - 0.18, np.mean(v) + 0.06, f"step\n{np.mean([np.array(r['step_resp'])[95] for r in R]):.2f} K", ha='center', va='bottom', fontsize=5.0, color=TXT2)
    ax.axhline(pers, color=TXT2, lw=0.8, ls=':'); ax.text(3.45, pers - 0.06, 'persistence', color=TXT2, fontsize=6, ha='right', va='top')
    ax.set_xticks(range(4)); ax.set_xticklabels([l[1] for l in labs], fontsize=5.8); ax.set_ylabel('48-h forecast error (K)'); ax.set_ylim(0, 3.2); grid(ax); ax.legend(frameon=False, fontsize=6, loc='upper right'); panel_label(ax, 'c')
    ax.text(0.02, 0.98, 'Chronos-2, heat-pump building\n(true 24-h step ≈ 2 K)', transform=ax.transAxes, va='top', fontsize=5.3, color=TXT2)
    # d: the validation that lies — operational hold-out error vs interventional error, all heat-pump twins
    ax = axs[3]
    for r in cf:
        ax.plot(r['rmse']['operational'], np.mean([r['rmse']['designed_s10'], r['rmse']['designed_s11']]), 'o' if r['kind'] == 'gru_sim' else 's', ms=4, color=C['operational'] if r['regime'] == 'operational' else C['designed'], alpha=0.8)
    for r in fm:
        col = C['fm'] if r['model'] == 'zero-shot' else C['operational'] if 'operational' in r['model'] else C['designed'] if 'designed' in r['model'] else C['dither']
        ax.plot(r['pred_rmse_operational'], np.mean([r['do_rmse_10'], r['do_rmse_11']]), 'D', ms=3.5, color=col, alpha=0.8)
    ax.plot([0.01, 6], [0.01, 6], color=GRID, lw=0.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.01, 1.5); ax.set_ylim(0.3, 8)
    ax.set_xlabel('error on held-out operational data (K)'); ax.set_ylabel('error on interventions (K)'); grid(ax); panel_label(ax, 'd')
    ax.text(0.03, 0.97, 'heat-pump building; ● GRU, ■ Transformer, ◆ Chronos-2', transform=ax.transAxes, va='top', fontsize=5.3, color=TXT2)
    ax.text(0.03, 0.30, 'operational', transform=ax.transAxes, va='top', fontsize=5.5, color=C['operational']); ax.text(0.03, 0.22, 'dithered', transform=ax.transAxes, va='top', fontsize=5.5, color=C['dither']); ax.text(0.03, 0.14, 'designed', transform=ax.transAxes, va='top', fontsize=5.5, color=C['designed'])
    save(fig, 'fig5')


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5']): globals()[name]()
