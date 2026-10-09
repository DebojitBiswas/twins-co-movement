#!/usr/bin/env python3
"""Numbers audit: every headline number of paper C, recomputed from the result JSONs (QUALITY.md item 6).  Run before each draft."""
import json, os, numpy as np
from collections import defaultdict
import c1_synthetic as c1

def med(x): return float(np.median(x))
out = []
P_ = lambda *a: out.append(' '.join(str(x) for x in a))

# --- synthetic (C1/C2/C5)
sub = json.load(open('c1_results_gru_sub.json')); ctl = json.load(open('c1b_results.json')); col = json.load(open('c2_collapse.json'))
P_('SYNTHETIC GRU-sim closed loop (24 cells): rel. error range', round(min(r['gru'] for r in sub), 2), '-', round(max(r['gru'] for r in sub), 2),
  '| spread range', round(min(r['gru_spread'] for r in sub), 2), '-', round(max(r['gru_spread'] for r in sub), 2),
  '| error/spread', round(min(r['gru'] / r['gru_spread'] for r in sub)), '-', round(max(r['gru'] / r['gru_spread'] for r in sub)))
P_('SYNTHETIC GRU-sim designed data N=2e3/2e4/2e5:', [round(r['gru'], 2) for r in ctl if r['regime'] == 'open'])
P_('LAW collapse (28 cells): GRU vs truth median', round(med([r['gru_vs_truth'] for r in col]), 2), 'vs law', round(med([r['gru_vs_law'] for r in col]), 2),
  '| OE vs truth', round(med([r['oe_vs_truth'] for r in col]), 2), 'vs law', round(med([r['oe_vs_law'] for r in col]), 2))
fam = json.load(open('c5_results.json')); fam2 = json.load(open('c5c_results.json'))
for k in ('gru_sim',):
    P_(f'FAMILIES {k}: designed', round(med([r['eff_err'] for r in fam if r['model'] == k and r['regime'] == 'open']), 2), 'closed(d0.8)', round(med([r['eff_err'] for r in fam if r['model'] == k and r['regime'] == 'cl_d0.8_sr0.0']), 2))
for k in ('gru_1s', 'tf_1s'):
    P_(f'FAMILIES {k}: designed', round(med([r['eff_err'] for r in fam2 if r['model'] == k and r['regime'] == 'open']), 2), 'closed(d0.8)', round(med([r['eff_err'] for r in fam2 if r['model'] == k and r['regime'] == 'cl_d0.8_sr0.0']), 2))
cl = [r for r in fam if r['regime'] != 'open'] ; cl2 = [r for r in fam2 if r['regime'] != 'open']
P_('PLANNING synthetic: closed-loop plans worse than doing nothing (3.0 K):', sum(r['track'] > 3.0 for r in cl), '/', len(cl), '(sim-GRU)', sum(r['track'] > 3.0 for r in cl2), '/', len(cl2), '(1-step families); planner on true plant 0.17 K')
if os.path.exists('c10_capacity.json'):
    cap = json.load(open('c10_capacity.json'))
    for reg in ('open', 'closed'):
        P_(f'CAPACITY {reg}:', {h: (round(np.mean([r['pred_rmse'] for r in cap if r['regime'] == reg and r['h'] == h]), 3), round(np.mean([r['eff_err'] for r in cap if r['regime'] == reg and r['h'] == h]), 2)) for h in sorted({r['h'] for r in cap})})

# --- pancreas
rows = json.load(open('c6_results.json'))
for d in (0.0, 0.7):
    R = [r for r in rows if r['d'] == d and r['A'] == 0.0 and r['hypo'] < 0.5]
    t = np.array([r['true'][23] for r in R]); w = np.array([r['lin'][23] for r in R])
    P_(f'PANCREAS d={d} n={len(R)}: twin 2-h effect positive in', int((w > 0).sum()), '/', len(R), '| true mean', round(t.mean(), 1), 'twin mean', round(w.mean(), 1),
      '| 4-h CV true', round(np.std([r['true'][47] for r in R]) / abs(np.mean([r['true'][47] for r in R])), 2), 'twin', round(np.std([r['lin'][47] for r in R]) / abs(np.mean([r['lin'][47] for r in R])), 2))
dos = json.load(open('c6c_dosing.json'))
for lab in sorted({k for r in dos for k in r['results']}):
    rr = [r['results'][lab] for r in dos if lab in r['results']]
    P_('DOSING', lab, ': mean dose', round(np.mean([x['dose'] for x in rr]), 1), 'severe', sum(x['severe'] for x in rr), '/', len(rr), 'no-dose', sum(x['dose'] == 0 for x in rr))

# --- buildings
for case, D in (('heat pump', 'c7_data'), ('commercial', 'c7_data_commercial')):
    if not os.path.exists(f'{D}/c7_neural.json'): continue
    nr = json.load(open(f'{D}/c7_neural.json'))
    g = defaultdict(list)
    for r in nr: g[(r['kind'], r['train'])].append((r['test']['10']['do_rmse'], r['test']['11']['do_rmse'], np.mean(r['step_resp'], 0)[95]))
    P_(f'BUILDING {case}: persistence', round(nr[0]['test']['10']['persistence'], 2), '/', round(nr[0]['test']['11']['persistence'], 2))
    for k, v in sorted(g.items()):
        v = np.array(v); P_(f'  {k[0]:8s} {k[1]:20s} do-RMSE {v[:,0].mean():.2f}±{v[:,0].std():.2f} / {v[:,1].mean():.2f}±{v[:,1].std():.2f}  step24h {v[:,2].mean():.2f}')
    if os.path.exists(f'{D}/c7b_results.json'):
        pl = json.load(open(f'{D}/c7b_results.json')); P_(f'  PLANNING embedded controller {np.mean(pl["embedded"]):.2f} K')
        g = defaultdict(list)
        for r in pl['rows']: g[(r['kind'], r['train'], r['iv'])].append((np.mean([q['track'] for q in r['plans']]), np.mean([q['believed'] for q in r['plans']])))
        for k, v in sorted(g.items()):
            v = np.array(v); P_(f'  plan {k[0]:8s} {k[1]:20s} {"IV" if k[2] else "  "} -> emulator {v[:,0].mean():.2f}±{v[:,0].std():.2f} K, believed {v[:,1].mean():.2f}')
    if os.path.exists(f'{D}/c11_foundation.json'):
        fm = json.load(open(f'{D}/c11_foundation.json')); g = defaultdict(list)
        for r in fm: g[r['model']].append((r['do_rmse_10'], r['do_rmse_11'], r['pred_rmse_operational'], np.array(r['step_resp'])[95]))
        for k, v in g.items():
            v = np.array(v); P_(f'  CHRONOS {k:24s} do-RMSE {v[:,0].mean():.2f} / {v[:,1].mean():.2f}  pred-op {v[:,2].mean():.2f}  step24h {v[:,3].mean():.2f}')
    if os.path.exists(f'{D}/c8_conformal.json'):
        cf = json.load(open(f'{D}/c8_conformal.json'))
        for reg in ('operational', 'designed'):
            R = [r for r in cf if r['regime'] == reg]
            P_(f'  CONFORMAL {reg}: coverage op-test', round(min(min(r['coverage']['operational'].values()) for r in R), 2), '-', round(max(max(r['coverage']['operational'].values()) for r in R), 2),
              '| designed test', round(min(min(r['coverage'][k].values()) for r in R for k in ('designed_s10', 'designed_s11')), 2), '-', round(max(max(r['coverage'][k].values()) for r in R for k in ('designed_s10', 'designed_s11')), 2))
    if os.path.exists(f'{D}/c9_updown.json'):
        ud = json.load(open(f'{D}/c9_updown.json'))
        for q in ('A', 'B'):
            eff = np.mean([np.mean(r[f'eff{q}_rms']) for r in ud])
            P_(f'  WHAT-IF {q} (effect {eff:.2f} K):', {m: (round(np.mean([np.mean(r[q][m]) for r in ud]), 2), round(np.mean([np.mean(r[f"eff{q}"][m]) for r in ud]) / eff, 2) if m in ud[0][f'eff{q}'] else None) for m in ud[0][q]})
open('numbers_audit.txt', 'w').write('\n'.join(out)); print('\n'.join(out))

# --- v1 buildings (c7_v1/summary.json from analyze_v1.py)
if os.path.exists('c7_v1/summary.json'):
    S = json.load(open('c7_v1/summary.json'))
    for P in ('P1', 'P2', 'P3'):
        for fam in ('gru_sim', 'gru_sim_big', 'gru_1s', 'tf_1s', 'arx48', 'pem2', 'grey'):
            for reg in ('operational', 'designed'):
                e = [x for x in S if x['period'] == P and x['family'] == fam and x['regime'] == reg]
                if e: e = e[0]; P_(f'V1 {P} {fam:12s} {reg:12s}: in-RMSE {e["in_rmse"]:.2f} (pers from summary) step ratio {e["ratio"]:.2f} [{e["ratio_ci"][0]:.2f},{e["ratio_ci"][1]:.2f}] n={e["n"]} cov op/in {e["cov_op"]:.2f}/{e["cov_in"]:.2f}')
            d = [x for x in S if x['period'] == P and x['family'] == fam and x['regime'].startswith('dither A2')]
            if d: P_(f'V1 {P} {fam:12s} dither ±2 K (3 logs): step ratio {min(x["ratio"] for x in d):.2f}–{max(x["ratio"] for x in d):.2f}, in-RMSE {min(x["in_rmse"] for x in d):.2f}–{max(x["in_rmse"] for x in d):.2f}')
    open('numbers_audit.txt', 'a').write('\n' + '\n'.join(out[-60:]))

# --- v1.1 referee-driven experiments (every main-text number should be traceable here)
def _ld(f):
    return json.load(open(f)) if os.path.exists(f) else None
ex = []
c5d = _ld('c5d_onestep_clean.json')
if c5d: ex.append('C5d clean one-step (d=0): ' + ', '.join(f"{r['model']} s{r['seed']} {r['eff_err']:.2f}" for r in c5d if r['regime'] == 'cl_d0.0'))
c11b = _ld('c7_data/c11b_sweep.json')
if c11b: ex.append('C11b Chronos designed sweep step24h: ' + ', '.join(f"lr{r['lr']:g}/{r['steps']} {np.array(r['step_resp'])[95]:.2f}K" for r in c11b if r['model'] == 'ft:designed') + f"; passes: {sum(r.get('pass', False) for r in c11b)}")
c7c = _ld('c7_data/c7c_control_function.json')
if c7c: ex.append('C7c control-function GRU step24h: ' + ', '.join(f"{r['train']} s{r['seed']} {np.mean(r['step_resp'], 0)[95]:.2f}" for r in c7c if r['kind'] == 'gru_sim'))
for P in ('P1', 'P2', 'P3'):
    c16 = _ld(f'c7_v1/{P}/c16_results.json')
    if c16: ex.append(f'C16 {P} big models operational (pre-rescore, superseded by c18): ' + ', '.join(f"{r['family']} s{r['seed']} {r['step']['ratio_24h']:.2f}" for r in c16 if r['train'] == 'log_operational'))
    c18 = _ld(f'c7_v1/{P}/c18_results.json')
    if c18:
        tr = np.mean([t['effect'] for t in json.load(open(f'c7_v1/{P}/truth_step.json'))], 0); HZ = {'3h': 11, '6h': 23, '12h': 47, '24h': 95, '48h': 191}
        ex.append(f"C18 {P} big models operational (corrected truth): " + ', '.join(f"{r['family']} s{r['seed']} {r['step']['ratio_24h']:.2f}" for r in c18 if r['train'] == 'log_operational' and r['family'].endswith('_big')))
        ex.append(f"C18 {P} step in-range fraction (first 24 h) base {c18[0]['step']['frac_inside_base']:.2f} interv {c18[0]['step']['frac_inside_interv']:.2f}")
        for fam in ('gru_sim', 'gru_sim_big', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
            rows = [r for r in c18 if r['family'] == fam and r['train'] == 'log_operational']
            if rows:
                Lm = np.mean([r['step']['learned'] for r in rows], 0)
                ex.append(f"C18 {P} {fam} operational fraction by horizon: " + ', '.join(f"{h} {Lm[i] / tr[i]:.2f}" for h, i in HZ.items()) + f" | conformal 6-24h op/sched/in-range-masked {np.mean([r['coverage']['operational']['6-24h'] for r in rows]):.2f}/{np.mean([r['coverage']['test_in']['6-24h'] for r in rows]):.2f}/{np.mean([r['coverage']['test_in_inrange']['6-24h'] for r in rows]):.2f}")
        for tag in (['dither_A0.5_s0', 'dither_A1.0_s0'] if P == 'P1' else []) + [f'dither_A2.0_s{s}' for s in range(3)]:
            op = np.load(f'c7_v1/{P}/log_operational.npy'); d = np.load(f'c7_v1/{P}/log_{tag}.npy')
            from scipy.signal import coherence as _coh
            f_, C_ = _coh(d[:, 1], d[:, 4] - op[:, 4], fs=1.0, nperseg=192); g2 = C_[(f_ > 1 / 192) & (f_ < 1 / 12)].mean()
            rows = [r for r in c18 if r['family'] == 'gru_sim' and r['train'] == 'log_' + tag]
            ex.append(f"C18 {P} {tag}: comfort rms {np.sqrt(np.mean((d[:, 0] - op[:, 0]) ** 2)):.2f} K, energy {100 * (d[:, 1].sum() / op[:, 1].sum() - 1):+.0f}%, gamma2(u, offset) {g2:.2f}, gru_sim frac24 {np.mean([r['step']['ratio_24h'] for r in rows]):.2f}, frac6 {np.mean([r['step']['learned'] for r in rows], 0)[23] / tr[23]:.2f}")
    c22 = _ld(f'c7_v1/{P}/c22_small_step.json')
    if c22 and len(c22) > 5:
        ex.append(f"C22 {P} +0.1 step: truth 6/24/48 h {c22[0]['small']['truth']['6h']:.2f}/{c22[0]['small']['truth']['24h']:.2f}/{c22[0]['small']['truth']['48h']:.2f} K, interv arm in range {c22[0]['small']['frac_inside_interv']:.2f} (+0.3: {c22[0]['big']['frac_inside_interv']:.2f})")
        for fam in ('gru_sim', 'gru_sim_big', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
            rows = [r for r in c22 if r['family'] == fam and r['train'] == 'log_operational']
            if rows: ex.append(f"C22 {P} {fam} operational +0.1 fraction by horizon: " + ', '.join(f"{h} {np.mean([r['small']['ratio'][h] for r in rows]):.2f}" for h in ('3h', '6h', '12h', '24h', '48h')) + f"; +0.3 24h {np.mean([r['big']['ratio']['24h'] for r in rows]):.2f}")
    c21 = _ld(f'c7_v1/{P}/c21_greybox_oe.json')
    if c21: ex.append(f'C21 {P} grey-box OE: ' + ', '.join(f"{r['train']} {r['step']['ratio_24h']:.2f}" for r in c21))
c20b = _ld('c20b_pancreas_arx_v1.json')
if c20b:
    for d in (0.0, 0.7):
        S_ = [x for x in c20b if x['d'] == d and 'error' not in x and x['hypo'] < 0.5]
        pats = sorted({x['patient'] for x in S_})
        ex.append(f"C20b pancreas ARX on regenerated v1 logs d={d} ({len(S_)} cells, {len(pats)} patients): " + '; '.join(f"{k} per-patient median ratio {np.nanmedian([np.nanmedian([x[k] / x['true4'] for x in S_ if x['patient'] == p]) for p in pats]):.2f}, positive {sum(np.nanmedian([x[k] for x in S_ if x['patient'] == p]) > 0 for p in pats)}/{len(pats)} patients, finite cells {sum(np.isfinite(x[k]) for x in S_)}/{len(S_)}" for k in ('lin4', 'arx6_4h', 'arx12_4h', 'arx24_4h')))
c20c = _ld('c20c_pancreas.json')
if c20c:
    S_ = [x for x in c20c if x['kind'] == 'designed' and 'error' not in x]
    ex.append(f"C20c pancreas ARX on DESIGNED logs ({len(S_)} patients): " + '; '.join(f"{k} median ratio {np.nanmedian([x[k] / x['true4'] for x in S_]):.2f}, finite {sum(np.isfinite(x[k]) for x in S_)}" for k in ('lin4', 'arx6_4h', 'arx12_4h', 'arx24_4h')))
c6f = _ld('c6f_dosing_regen.json')
if c6f:
    dos = c6f['dosing']
    for lab in sorted({x['label'] for x in dos}):
        S_ = [x for x in dos if x['label'] == lab and 'error' not in x]
        ex.append(f"C6f dosing on regenerated logs {lab}: n={len(S_)} dose {np.mean([x['dose'] for x in S_]):.1f} severe {sum(x['severe'] for x in S_)}/{len(S_)} no-dose {sum(x['dose'] == 0 for x in S_)}")
for P in ('P1', 'P2', 'P3'):
    for MODE in ('shuffled', 'strat'):
        c23 = _ld(f'c7_v1/{P}/c23_{MODE}.json')
        if c23:
            tr23 = np.mean([t['effect'] for t in json.load(open(f'c7_v1/{P}/truth_step.json'))], 0)
            for fam in sorted({r['family'] for r in c23}):
                ex.append(f"C23 {P} {MODE}-day replay {fam}: " + ', '.join(f"{t} 24h {np.mean([r['step']['ratio_24h'] for r in c23 if r['family'] == fam and r['train'] == t]):.2f} 6h {np.mean([r['step']['learned'] for r in c23 if r['family'] == fam and r['train'] == t], 0)[23] / tr23[23]:.2f} (op err {np.mean([r['test']['test_operational']['rmse'] for r in c23 if r['family'] == fam and r['train'] == t]):.2f})" for t in (f'log_{MODE}_s0', f'log_{MODE}_s1')))
c17 = _ld('c7_v1/P1/c17_hp_sweep.json')
if c17: ex.append('C17 HP sweep operational fractions by recipe (tf_1s): ' + ', '.join(f"h{r['h']}/lr{r['lr']:g}/{r['steps']} s{r['seed']} {r['ratio']:.2f}" for r in c17 if r['kind'] == 'tf_1s' and r['train'] == 'log_operational'))
ic = _ld('c7_v1/inverse_controller.json')
if ic: ex.append('C19 inverse controller: ' + '; '.join(f"{P} -1/C {v['inv_ctrl_24h']:+.2f} truth {v['truth_24h']:.2f} law {v['law_pred_24h']:+.2f} learned " + ','.join(f"{k} {x:+.2f}" for k, x in v['learned_24h'].items()) for P, v in ic.items()))
c20 = _ld('c20_pancreas_arx.json')
if c20:
    for d in (0.0, 0.7):
        S = [x for x in c20 if x['d'] == d]
        ex.append(f'C20 pancreas ARX d={d}: ' + ', '.join(f"{k} median ratio {np.nanmedian([x[k] / x['true4'] for x in S]):.2f} positive {np.mean([x[k] > 0 for x in S]):.2f}" for k in ('arx6_4h', 'arx12_4h', 'arx24_4h')))
c6e = _ld('c6e_dosing_v2.json')
if c6e:
    for lab in sorted({x['label'] for x in c6e['dosing']}):
        S = [x for x in c6e['dosing'] if x['label'] == lab and 'error' not in x]
        ex.append(f"C6e dosing {lab}: n={len(S)} dose {np.mean([x['dose'] for x in S]):.1f} severe {sum(x['severe'] for x in S)}/{len(S)} no-dose {sum(x['dose']==0 for x in S)} BG0 {min(x['bg0'] for x in S):.0f}-{max(x['bg0'] for x in S):.0f}")
for f in ('c7_data/c12_mpc.json', 'c7_data/c12_mpc_t21.8.json'):
    m = _ld(f)
    if m:
        ex.append(f"C12 {f}: embedded {np.mean([e['track'] for e in m['embedded']]):.2f}; " + '; '.join(f"{t} λ{l:g} {np.mean([r['track'] for r in m['rows'] if r['train']==t and r['lam']==l]):.2f}±{np.std([r['track'] for r in m['rows'] if r['train']==t and r['lam']==l]):.2f}" for t in ('operational_A0.0_s0', 'dither_A2.0_s0', 'designed_A0.0_s0') for l in (0.0, 10.0)))
open('numbers_audit.txt', 'a').write('\n' + '\n'.join(ex)); print('\n'.join(ex))
