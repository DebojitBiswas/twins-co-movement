#!/usr/bin/env python3
"""Generate paper/ed_tables.tex (ED Tables 3-6) from c7_v1/summary.json, c17_hp_sweep.json and the v1 logs, so the tex carries the audited numbers
(round-3 ML referee MAJOR 5, 6, 7, 9): two-clause capability table; HP sweep; conformal coverage (operational / in-range schedule /
in-range-masked); dither amplitude series with comfort and energy cost."""
import json, numpy as np
S = json.load(open('c7_v1/summary.json')); PER = ['P1', 'P2', 'P3']; PERS = {'P1': 1.10, 'P2': 1.74, 'P3': 1.40}
NAME = dict(gru_sim='GRU, simulation error', gru_sim_big='GRU 256, simulation error', gru_1s='GRU, next-step', gru_1s_big='GRU 256, next-step',
            gru_1s_noise='GRU, next-step + noise', tf_1s='Transformer, next-step', tf_1s_big='Transformer 4-layer, next-step',
            tf_1s_noise='Transformer, next-step + noise', arx12='ARX(12)', arx48='ARX(48)', pem2='PEM-SS(2)', grey='Grey-box 2R2C (innovations gain)')
FAMS = list(NAME)
def get(f, r, P):
    for e in S:
        if e['family'] == f and e['regime'] == r and e['period'] == P: return e
L = []
# ED Table 3: two-clause capability
L.append(r'\paragraph{ED Table 3 --- Two-clause capability criterion, designed-data twins, applied as pre-stated.} Clause 1: 24-h learned fraction within $\pm50\%$ of the truth (a ratio rule, asymmetric on the log scale: 0.50 fails, 1.50 passes). Clause 2: rollout error on the in-range designed test schedule, all samples, below persistence on the same schedule (1.10, 1.74, 1.40~K; restricted to in-range samples the pair is twin in-range error against persistence 0.98, 0.80 and 1.62~K, which fails most families in period~2, ARX(48) included, because the in-range samples there are the quiet ones, so that version is informative about the mask rather than the twins). Entries: fraction / in-range designed-test error (K); bold where both clauses pass; superscripts mark the failed clause.')
L.append(r'\begin{center}\small\begin{tabular}{lccc}\toprule family & period 1 & period 2 & period 3\\\midrule')
MPERS = {'P1': 0.98, 'P2': 0.80, 'P3': 1.62}
for f in FAMS:
    row = [NAME[f]]
    for P in PER:
        e = get(f, 'designed', P)
        if not e: row.append('--'); continue
        c1 = abs(e['ratio'] - 1) <= 0.5; c2 = e['in_rmse'] < PERS[P]; s_ = f"{e['ratio']:.2f} / {e['in_rmse']:.2f}"
        row.append(r'\textbf{' + s_ + '}' if (c1 and c2) else s_ + ('$^{1}$' if not c1 else '') + ('$^{2}$' if not c2 else ''))
        print(f'[masked pair] {P} {f}: twin in-range {e["in_range"]:.2f} vs masked persistence {MPERS[P]:.2f} -> {"pass" if e["in_range"] < MPERS[P] else "FAIL"}')
    L.append(' & '.join(row) + r'\\')
L.append(r'\bottomrule\end{tabular}\end{center} The GPU families have 2--3 seeds and the 256-unit next-step GRU was run in period~1 only. Under the pre-stated rule the simulation-error GRU, its 256-unit version and the clean next-step GRU fail clause~2 in period~1 (designed logs lie mostly outside the operating range there) and the noise-injected transformer fails clause~1 in period~2 by 0.001; the headline comparison therefore rests on periods~2 and~3 for the simulation-error GRU, and no headline range depends on the noise-injected transformer\'s period-2 cell.')
# ED Table 4: HP sweep
H = json.load(open('c7_v1/P1/c17_hp_sweep.json'))
L.append(r'\paragraph{ED Table 4 --- Recipe sweep for the clean next-step families (period 1).} Hidden size $\{48, 96\}$, learning rate $\{10^{-3}, 3\cdot10^{-3}\}$, steps $\{6000, 10000\}$, two seeds, trained on the designed log (selection) and on the operational log (report). Selection rule, fixed before the sweep: lowest in-range designed-test error of the designed-data twin. Entries: operational-twin learned fraction (seed 0 / seed 1) and its in-range error on the designed test schedule, mean of two seeds.')
L.append(r'\begin{center}\small\begin{tabular}{lcccccc}\toprule recipe & TF selection error (K) & TF fraction & TF in-range (K) & GRU selection error (K) & GRU fraction & GRU in-range (K)\\\midrule')
recs = sorted({(r['h'], r['lr'], r['steps']) for r in H})
sel = {}
for kind in ('tf_1s', 'gru_1s'):
    des = {rc: np.mean([r['in_rmse'] for r in H if r['kind'] == kind and r['train'].startswith('log_designed') and (r['h'], r['lr'], r['steps']) == rc]) for rc in recs}
    sel[kind] = min(des, key=des.get)
for rc in recs:
    row = [f"h{rc[0]}, lr {rc[1]:g}, {rc[2]} steps"]
    for kind in ('tf_1s', 'gru_1s'):
        rs = sorted([r for r in H if r['kind'] == kind and r['train'] == 'log_operational' and (r['h'], r['lr'], r['steps']) == rc], key=lambda r: r['seed'])
        ds = [r for r in H if r['kind'] == kind and r['train'].startswith('log_designed') and (r['h'], r['lr'], r['steps']) == rc]
        fr = ' / '.join(f"{r['ratio']:.2f}" for r in rs); ir = f"{np.mean([r['in_rmse'] for r in rs]):.2f}"; se = f"{np.mean([r['in_rmse'] for r in ds]):.2f}"
        if sel[kind] == rc: fr = r'\textbf{' + fr + '}'; se = r'\textbf{' + se + '}'
        row += [se, fr, ir]
    L.append(' & '.join(row) + r'\\')
allt = [r['ratio'] for r in H if r['kind'] == 'tf_1s' and r['train'] == 'log_operational']; allg = [r['ratio'] for r in H if r['kind'] == 'gru_1s' and r['train'] == 'log_operational']
irg = [r['in_rmse'] for r in H if r['kind'] == 'gru_1s' and r['train'] == 'log_operational']
L.append(r'\bottomrule\end{tabular}\end{center} Bold, the recipe selected on designed data. Across the sixteen operational transformer runs the fraction spans %.2f--%.2f: fifteen lie above the noise-injected transformer\'s interval (0.22--0.50) and one (0.49) touches its upper limit. The clean GRU spans %.2f--%.2f with in-range errors of %.1f--%.1f~K against persistence 1.10~K, i.e.\ it is unstable rather than merely recipe-dependent.' % (min(allt), max(allt), min(allg), max(allg), min(irg), max(irg)))
# ED Table 5: conformal
L.append(r'\paragraph{ED Table 5 --- Conformal coverage of 90\% intervals, operational-data twins, 6--24-h horizon band.} Columns: held-out operational days; the in-range designed schedule, all samples; the same schedule restricted to samples whose temperature lies inside the training log\'s range (the quantity quoted in Results). Mean over seeds.')
L.append(r'\begin{center}\small\begin{tabular}{l' + 'ccc' * 3 + r'}\toprule & \multicolumn{3}{c}{period 1} & \multicolumn{3}{c}{period 2} & \multicolumn{3}{c}{period 3}\\ family & op. & sched. & in-range & op. & sched. & in-range & op. & sched. & in-range\\\midrule')
for f in ('gru_sim', 'gru_sim_big', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
    row = [NAME[f]]
    for P in PER:
        e = get(f, 'operational', P); row += ([f"{100*e['cov_op']:.0f}", f"{100*e['cov_in']:.0f}", f"{100*e['cov_in_range']:.0f}"] if e else ['--'] * 3)
    L.append(' & '.join(row) + r'\\')
L.append(r'\bottomrule\end{tabular}\end{center}')
# ED Table 6: amplitude series
L.append(r'\paragraph{ED Table 6 --- Dither amplitude series (period 1) and the cost of the $\pm2$-K dither in every period.} Comfort cost, rms difference between the dithered and the operational zone temperature over the training month; energy, change in summed heat-pump modulation. Learned fraction of operational-twin families trained on the dithered log (mean over seeds; for $\pm2$~K the range over the three logs).')
L.append(r'\begin{center}\small\begin{tabular}{lcccccc}\toprule log & comfort (K) & energy & GRU sim. & TF next-step & ARX(48) & PEM-SS(2)\\\midrule')
def cost(P, tag):
    op = np.load(f'c7_v1/{P}/log_operational.npy'); d = np.load(f'c7_v1/{P}/{tag}.npy')
    return np.sqrt(np.mean((d[:, 0] - op[:, 0]) ** 2)), 100 * (d[:, 1].sum() / op[:, 1].sum() - 1)
for a in ('0.5', '1.0'):
    c, en = cost('P1', f'log_dither_A{a}_s0'); row = [f"period 1, $\\pm{a}$~K", f"{c:.2f}", f"{en:+.0f}\\%"]
    for f in ('gru_sim', 'tf_1s', 'arx48', 'pem2'):
        e = get(f, f'dither A{a} s0', 'P1'); row.append(f"{e['ratio']:.2f}" if e else '--')
    L.append(' & '.join(row) + r'\\')
for P in PER:
    cs = [cost(P, f'log_dither_A2.0_s{s}') for s in range(3)]; row = [f"period {P[1]}, $\\pm2$~K (3 logs)", f"{min(c for c, _ in cs):.2f}--{max(c for c, _ in cs):.2f}", f"{min(e for _, e in cs):+.0f} to {max(e for _, e in cs):+.0f}\\%"]
    for f in ('gru_sim', 'tf_1s', 'arx48', 'pem2'):
        es = [get(f, f'dither A2.0 s{s}', P) for s in range(3)]; es = [e for e in es if e]
        row.append(f"{min(e['ratio'] for e in es):.2f}--{max(e['ratio'] for e in es):.2f}" if es else '--')
    L.append(' & '.join(row) + r'\\')
L.append(r'\bottomrule\end{tabular}\end{center}')
open('paper/ed_tables.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

# ED Table 7: dither-log law test; ED Table 8: horizons and the +0.1 step
from scipy.signal import coherence
R18 = {P: json.load(open(f'c7_v1/{P}/c18_results.json')) for P in PER}; TR = {P: np.mean([t['effect'] for t in json.load(open(f'c7_v1/{P}/truth_step.json'))], 0) for P in PER}
IC = json.load(open('c7_v1/inverse_controller.json'))
L = [r'\paragraph{ED Table 7 --- The linear law compared with the dithered logs, where its excitation is exogenous.} Squared coherence $\gamma^2$ between the applied modulation and the logged set-point offset, averaged over the 14 Welch bins with periods between 3.2 and 24~h (band chosen after the fact; Extended Data Table~2 item~19), the law\'s 24-h prediction $\gamma^2 + (1-\gamma^2)\,G_{\mathrm{op}}(24\,\mathrm{h})/G(24\,\mathrm{h})$ with the fitted controller\'s inverse ($-0.07$, $-0.29$ and $-0.22$ of the truth in the three periods), and the simulation-error GRU\'s learned fraction at 24 and 6~h on the same log (mean over five seeds). This is a comparison, not a test with a null: the logs share weather within a period, and the law\'s prediction is a scalar for one horizon.']
L.append(r'\begin{center}\small\begin{tabular}{lccccc}\toprule log & $\gamma^2$ & law, 24 h & GRU, 24 h & GRU $-$ law & GRU, 6 h\\\midrule')
G2, F24, LAW = [], [], []
for P in PER:
    op = np.load(f'c7_v1/{P}/log_operational.npy'); gop = IC[P]['inv_ctrl_24h'] / IC[P]['truth_24h']
    for tag in (['dither_A0.5_s0', 'dither_A1.0_s0'] if P == 'P1' else []) + [f'dither_A2.0_s{s}' for s in range(3)]:
        d = np.load(f'c7_v1/{P}/log_{tag}.npy'); f, Cc = coherence(d[:, 1], d[:, 4] - op[:, 4], fs=1.0, nperseg=192); g2 = Cc[(f > 1 / 192) & (f < 1 / 12)].mean()
        rows = [r for r in R18[P] if r['family'] == 'gru_sim' and r['train'] == 'log_' + tag]; f24 = np.mean([r['step']['ratio_24h'] for r in rows]); f6 = np.mean([r['step']['learned'] for r in rows], 0)[23] / TR[P][23]
        law = g2 + (1 - g2) * gop; G2.append(g2); F24.append(f24); LAW.append(law)
        L.append(f"period {P[1]}, {tag.replace('dither_A', '$\\pm$').replace('_s', '~K, log ')} & {g2:.2f} & {law:.2f} & {f24:.2f} & {f24 - law:+.2f} & {f6:.2f}\\\\")
from scipy.stats import spearmanr
rho, pv = spearmanr(G2, F24); rms_law = np.sqrt(np.mean((np.array(F24) - np.array(LAW)) ** 2)); rms_const = np.sqrt(np.mean((np.array(F24) - np.mean(F24)) ** 2))
L.append(r'\bottomrule\end{tabular}\end{center} Across the eleven logs the GRU exceeds the law in every period-3 log by 0.2--0.3 and agrees within 0.1 in six; Spearman $\rho(\gamma^2, \text{fraction}) = %.2f$ ($p = %.2f$), and the rms deviation from the law (%.3f) is larger than that from the constant mean fraction (%.3f): the scalar law does not predict the fraction across these logs. The 6-h fractions (0.94--1.17) are not described by the scalar law at all.' % (rho, pv, rms_law, rms_const))
print(f'[law] rho {rho:.2f} p {pv:.2f} rms law {rms_law:.3f} const {rms_const:.3f}; within 0.1: {sum(abs(a-b)<=0.1 for a,b in zip(F24,LAW))}/11')
try:
    C22 = {P: json.load(open(f'c7_v1/{P}/c22_small_step.json')) for P in PER}; assert all(len(v) > 5 for v in C22.values())
    HZ = ['3h', '6h', '12h', '24h', '48h']
    L.append(r'\paragraph{ED Table 8 --- Learned fraction by horizon, $+0.3$ and $+0.1$ steps from the same seven operational start states (operational-data twins, mean over seeds; the $+0.1$ 24-h entry carries the hierarchical bootstrap interval over seeds and starts).} In-range fraction of the first 24~h of each arm (base 0.4; intervention 0.7 or 0.5) and the ratio of the operational-data to the designed-data 24-h estimate (op./des.). True $+0.1$ effects at 24~h: %.2f, %.2f, %.2f~K.' % tuple(C22[P][0]['small']['truth']['24h'] for P in PER))
    L.append(r'\begin{center}\scriptsize\begin{tabular}{ll' + 'c' * 11 + r'}\toprule period & family & \multicolumn{5}{c}{$+0.3$ step: 3, 6, 12, 24, 48 h} & \multicolumn{5}{c}{$+0.1$ step: 3, 6, 12, 24, 48 h} & op./des.\\\midrule')
    for P in PER:
        fb, fi, fs = C22[P][0]['big']['frac_inside_interv'], C22[P][0]['small']['frac_inside_interv'], R18[P][0]['step']['frac_inside_base']
        L.append(r'\multicolumn{13}{l}{period %s: in range, base arm %.2f, $+0.3$ arm %.2f, $+0.1$ arm %.2f}\\' % (P[1], fs, fb, fi))
        for f in ('gru_sim', 'gru_sim_big', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
            rows = [r for r in C22[P] if r['family'] == f and r['train'] == 'log_operational']
            if not rows: continue
            e, ed = get(f, 'operational', P), get(f, 'designed', P)
            import hboot
            big = [f"{np.mean([r['big']['ratio'][h] for r in rows]):.2f}" for h in HZ]; small = [f"{np.mean([r['small']['ratio'][h] for r in rows]):.2f}" for h in HZ]
            if 'per_start' in rows[0]['big']:
                m, lo, hi = hboot.ratio_ci(rows, 'small', '24h'); small[3] = f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
            L.append(' & '.join([P[1], NAME[f]] + big + small + [f"{e['ratio'] / ed['ratio']:.2f}" if e and ed else '--']) + r'\\')
    L.append(r'\bottomrule\end{tabular}\end{center}')
except (FileNotFoundError, AssertionError): print('c22 not complete; ED Table 8 skipped')
open('paper/ed_tables.tex', 'a').write('\n'.join(L) + '\n'); print('\n'.join(L))

# ED Table 9: operational/designed ratio by horizon (same architecture, same horizon), +0.3 step
try:
    C22 = {P: json.load(open(f'c7_v1/{P}/c22_small_step.json')) for P in PER}; HZ = ['3h', '6h', '12h', '24h', '48h']
    L = [r'\paragraph{ED Table 9 --- Operational-data response normalised by the designed-data response of the same architecture at the same horizon ($+0.3$ step).} A ratio of 1 means the operational twin learned what its designed-data counterpart learned; decay with horizon that is specific to the operational twins, and absent for the direct methods, is the signature of the operating regime rather than of the model class.']
    L.append(r'\begin{center}\small\begin{tabular}{ll' + 'c' * 5 + r'}\toprule period & family & 3 h & 6 h & 12 h & 24 h & 48 h\\\midrule')
    for P in PER:
        for f in ('gru_sim', 'gru_sim_big', 'gru_1s_noise', 'tf_1s', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
            o = [r for r in C22[P] if r['family'] == f and r['train'] == 'log_operational']; d = [r for r in C22[P] if r['family'] == f and r['train'].startswith('log_designed')]
            if not o or not d: continue
            L.append(' & '.join([P[1], NAME[f]] + [f"{np.mean([r['big']['ratio'][h] for r in o]) / np.mean([r['big']['ratio'][h] for r in d]):.2f}" for h in HZ]) + r'\\')
    L.append(r'\bottomrule\end{tabular}\end{center}')
    open('paper/ed_tables.tex', 'a').write('\n'.join(L) + '\n')
except FileNotFoundError: pass

# ED Table 10: open-loop replays of the operational input (c23: shuffled days; stratified = permuted within quartiles of daily mean outdoor temperature)
def r2_drivers(Lg):
    hour = (np.arange(len(Lg)) % 96) / 96.0; X = np.column_stack([Lg[:, 2], Lg[:, 3], Lg[:, 4], np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour), np.ones(len(Lg))])
    b = np.linalg.lstsq(X, Lg[:, 1], rcond=None)[0]; return 1 - np.var(Lg[:, 1] - X @ b) / np.var(Lg[:, 1])
def ci_rows(rows):
    S = np.array([r['step']['per_start_24h'] for r in rows]); t = np.array(rows[0]['step']['truth_per_start_24h']); ns, nk = S.shape
    rng = np.random.default_rng(sum(map(ord, rows[0]['train'] + rows[0]['family'] + rows[0]['period'])))
    rb = [np.sum(S[i][:, j]) / (ns * np.sum(t[j])) for i, j in ((rng.choice(ns, ns), rng.choice(nk, nk)) for _ in range(1000))]
    return float(np.mean(S.mean(1) / t.mean())), float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))
from scipy.signal import welch
for MODE, title in (('shuffled', 'ED Table 10 --- Shuffled-day replay: the operational modulation sequence with its days permuted, replayed open-loop.'), ('strat', 'ED Table 11 --- Weather-stratified replay: days permuted only within quartiles of the daily mean outdoor temperature, replayed open-loop.')):
    try:
        C23 = {P: json.load(open(f'c7_v1/{P}/c23_{MODE}.json')) for P in PER}
    except FileNotFoundError: print(f'c23 {MODE} incomplete; table skipped'); continue
    L = [r'\paragraph{%s} Same weather, occupancy schedule, multiset of input values and within-day profile as the operational log. Per log: correlation of the replayed with the operational sequence; correlation of the modulation with the outdoor temperature and $R^2$ of the modulation on the logged drivers (outdoor temperature, irradiance, set-point, hour of day) against the operational log\'s; zone-temperature range and s.d.; fraction of replay samples inside the operational range; replay-to-operational ratio of input power in the 1--2-day and $>$2-day bands (7-day Welch segments; the daily line is within 4\%% in every log). Per family: 24-h learned fraction with a hierarchical bootstrap interval over seeds and start states, the 6-h fraction, the twin\'s rollout error on the operational test weeks, and the operational-log values. The same two permutation seeds are used in every period; the spread between the two logs exceeds the spread between seeds and is not characterised by two draws.' % title]
    for P in PER:
        op = np.load(f'c7_v1/{P}/log_operational.npy'); lo, hi = op[:, 0].min(), op[:, 0].max()
        dth = [np.load(f'c7_v1/{P}/log_dither_A2.0_s{s}.npy') for s in range(3)]; des = [np.load(f'c7_v1/{P}/log_designed_s{s}.npy') for s in range(2)]
        L.append(r'\emph{Period %s.} Operational log: corr$(u,T_{\mathrm{out}})$ %.2f, $R^2$ %.2f, zone temperature %.1f--%.1f~$^\circ$C (s.d.\ %.2f~K); dithered logs: corr %.2f to %.2f, $R^2$ %.2f--%.2f; designed logs: $R^2$ %.2f--%.2f; persistence on the operational weeks %.2f~K.' % (P[1], np.corrcoef(op[:, 1], op[:, 2])[0, 1], r2_drivers(op), lo, hi, op[:, 0].std(), min(np.corrcoef(d[:, 1], d[:, 2])[0, 1] for d in dth), max(np.corrcoef(d[:, 1], d[:, 2])[0, 1] for d in dth), min(r2_drivers(d) for d in dth), max(r2_drivers(d) for d in dth), min(r2_drivers(d) for d in des), max(r2_drivers(d) for d in des), C23[P][0]['test']['test_operational']['persistence']))
        L.append(r'\begin{center}\scriptsize\begin{tabular}{lcccccccc}\toprule log & corr.\ with op.\ $u$ & corr$(u,T_{\mathrm{out}})$ & $R^2$ & range ($^\circ$C) & s.d.\ (K) & in range & power 1--2 d & power $>$2 d\\\midrule')
        for s in (0, 1):
            d = np.load(f'c7_v1/{P}/log_{MODE}_s{s}.npy'); f, Po = welch(op[:, 1], fs=1.0, nperseg=672); f, Pr = welch(d[:, 1], fs=1.0, nperseg=672)
            band = lambda a, b: Pr[(f >= a) & (f < b)].sum() / Po[(f >= a) & (f < b)].sum()
            L.append(f"replay {s} & {np.corrcoef(d[:, 1], op[:, 1])[0, 1]:.2f} & {np.corrcoef(d[:, 1], d[:, 2])[0, 1]:.2f} & {r2_drivers(d):.2f} & {d[:, 0].min():.1f}--{d[:, 0].max():.1f} & {d[:, 0].std():.2f} & {np.mean((d[:, 0] >= lo) & (d[:, 0] <= hi)):.2f} & {band(1 / 192, 1 / 96):.2f} & {band(0, 1 / 192):.2f}\\\\")
        L.append(r'\bottomrule\end{tabular}\end{center}')
        L.append(r'\begin{center}\scriptsize\begin{tabular}{lcccccccc}\toprule & \multicolumn{3}{c}{replay 0} & \multicolumn{3}{c}{replay 1} & \multicolumn{2}{c}{operational log}\\ family & 24 h [interval] & 6 h & op.\ error & 24 h [interval] & 6 h & op.\ error & 24 h & 6 h\\\midrule')
        for fam in ('gru_sim', 'tf_1s', 'gru_1s_noise', 'tf_1s_noise', 'arx12', 'arx48', 'pem2'):
            cells = []
            for t in (f'log_{MODE}_s0', f'log_{MODE}_s1'):
                rows = [r for r in C23[P] if r['family'] == fam and r['train'] == t]
                if not rows: cells += ['--', '--', '--']; continue
                Lm = np.mean([r['step']['learned'] for r in rows], 0); m, lo_, hi_ = ci_rows(rows)
                cells += [f"{m:.2f} [{lo_:.2f}, {hi_:.2f}]", f"{Lm[23] / TR[P][23]:.2f}", f"{np.mean([r['test']['test_operational']['rmse'] for r in rows]):.2f}"]
            o = [r for r in R18[P] if r['family'] == fam and r['train'] == 'log_operational']; Lo = np.mean([r['step']['learned'] for r in o], 0)
            L.append(' & '.join([NAME[fam]] + cells + [f"{Lo[95] / TR[P][95]:.2f}", f"{Lo[23] / TR[P][23]:.2f}"]) + r'\\')
        L.append(r'\bottomrule\end{tabular}\end{center}')
    open('paper/ed_tables.tex', 'a').write('\n'.join(L) + '\n')

# post-process: raw-string apostrophes were escaped as \\' for Python; LaTeX wants a plain apostrophe
_s = open('paper/ed_tables.tex').read().replace("\\'", "'")
# wide tables (more than six columns) are scaled to the text width
import re as _re
def _wrap(m):
    spec = m.group(2); ncol = len(_re.sub(r'[^lcr]', '', spec))
    return m.group(0) if ncol <= 6 else m.group(1) + r'\resizebox{\textwidth}{!}{' + m.group(3) + '}' + m.group(4)
_s = _re.sub(r'(\\begin\{center\}(?:\\small|\\scriptsize)?)(?=\\begin\{tabular\}\{([^}]*)\})(\\begin\{tabular\}.*?\\end\{tabular\})(\\end\{center\})', _wrap, _s, flags=_re.S)
open('paper/ed_tables.tex', 'w').write(_s)
