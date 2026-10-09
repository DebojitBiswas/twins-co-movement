#!/usr/bin/env python3
"""Table 1 from c7_v1/summary.json + c18_results.json (6-h fractions): operational 24 h [CI], operational 6 h, dithered range, designed mean."""
import json, numpy as np
S = json.load(open('c7_v1/summary.json')); PER = ['P1', 'P2', 'P3']
R = {P: json.load(open(f'c7_v1/{P}/c18_results.json')) for P in PER}; T = {P: np.mean([t['effect'] for t in json.load(open(f'c7_v1/{P}/truth_step.json'))], 0) for P in PER}
ROWS = [('gru_sim', 'GRU, simulation-error (5 seeds)'), ('gru_sim_big', 'GRU 256 units, simulation-error (3, 2, 2)'), ('gru_1s_noise', 'GRU, next-step + noise (5)'), ('tf_1s_noise', 'Transformer, next-step + noise (5)'),
        ('tf_1s', 'Transformer, next-step (5)'), ('tf_1s_big', 'Transformer 4 layers, next-step (2)'), ('gru_1s', 'GRU, next-step (5; unstable)'), ('arx2', 'ARX(2)'), ('arx12', 'ARX(12)'), ('arx24', 'ARX(24)'), ('arx48', 'ARX(48)'), ('pem2', 'PEM state-space (2)'), ('grey', 'grey-box 2R2C, free gain')]
def get(f, r, P):
    for e in S:
        if e['family'] == f and e['regime'] == r and e['period'] == P: return e
import hboot
C22 = {P: hboot.load(P) for P in PER}
def frac6(f, P):
    if C22[P]:
        rows = [r for r in C22[P] if r['family'] == f and r['train'] == 'log_operational']
        if rows: m, lo, hi = hboot.ratio_ci(rows, 'big', '6h'); return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
    rows = [r for r in R[P] if r['family'] == f and r['train'] == 'log_operational']
    return f"{np.mean([r['step']['learned'] for r in rows], 0)[23] / T[P][23]:.2f}" if rows else None
L = [r'\begin{table}[t]\centering\footnotesize',
     r'\caption{\textbf{Learned fraction of the true heating effect, BOPTEST heat-pump building, from operational start states.} Twins trained on the operational log (24-h fraction, mean over seeds with 95\% hierarchical bootstrap interval over seeds and the seven 48-h-spaced start states; 6-h fraction, mean and the same interval), on three $\pm2$-K dithered logs (24-h, range over logs) and on designed logs (24-h, mean), for three heating-season periods. True 24-h effects 2.55, 2.97 and 2.70~K (s.d.\ over starts 0.33, 0.37, 0.40); 6-h effects 0.92, 1.13, 0.96~K. Seeds in parentheses; for the single-fit linear models the interval is a bootstrap over the seven starts only. The interval statistic is the ratio of the mean learned to the mean true effect over resampled seeds and starts; the point estimate is the mean of per-seed ratios. The grey-box model with a free innovations gain fails the designed-data capability criterion and is shown for completeness; the 48-unit clean next-step GRU is shown although its rollouts are worse than persistence. Extended Data Table~8 gives every horizon and the $+0.1$ step.}\label{tab:v1}',
     r'\resizebox{\textwidth}{!}{\begin{tabular}{lccc|ccc|ccc|ccc}\toprule',
     r' & \multicolumn{3}{c|}{operational log, 24 h} & \multicolumn{3}{c|}{operational log, 6 h} & \multicolumn{3}{c|}{operational + $\pm2$-K dither (3 logs)} & \multicolumn{3}{c}{designed data}\\',
     r'family & P1 & P2 & P3 & P1 & P2 & P3 & P1 & P2 & P3 & P1 & P2 & P3\\\midrule']
for f, name in ROWS:
    row = [name]
    for P in PER:
        e = get(f, 'operational', P); row.append(f"{e['ratio']:.2f} [{e['ratio_ci'][0]:.2f}, {e['ratio_ci'][1]:.2f}]" if e else '--')
    for P in PER:
        v = frac6(f, P); row.append(v if v is not None else '--')
    for P in PER:
        es = [x for x in S if x['family'] == f and x['period'] == P and x['regime'].startswith('dither A2')]
        row.append('--' if not es else (f"{es[0]['ratio']:.2f} (one log)" if len(es) == 1 else f"{min(x['ratio'] for x in es):.2f}--{max(x['ratio'] for x in es):.2f}"))
    for P in PER:
        e = get(f, 'designed', P); row.append(f"{e['ratio']:.2f}" if e else '--')
    L.append(' & '.join(row) + r'\\')
L.append(r'\bottomrule\end{tabular}}\end{table}')
open('paper/table_v1.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[5:]))
