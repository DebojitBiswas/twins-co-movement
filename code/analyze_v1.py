#!/usr/bin/env python3
"""Aggregate v1 results (c7_v1/P*/c15_results.json): per period × family × training regime — in-support do-RMSE (mean ± s.d. over seeds; block
bootstrap 95 % CI over test windows), in-range-only RMSE, out-of-range RMSE, predictive RMSE, step ratio (24 h; bootstrap over the 10 starts),
coverage (6–24 h band, operational vs in-support test).  Prints a table and writes c7_v1/summary.json."""
import json, os, sys, numpy as np
rng = np.random.default_rng(0)

def boot_ci(vals, n=1000):
    vals = np.asarray(vals); 
    if len(vals) < 2: return (float('nan'), float('nan'))
    b = [np.mean(rng.choice(vals, len(vals))) for _ in range(n)]; return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))

def regime(train):
    return 'operational' if train == 'log_operational' else 'designed' if train.startswith('log_designed') else train.replace('log_dither_', 'dither ').replace('_s', ' s')

summary = []
for P in ('P1', 'P2', 'P3'):
    f = f'c7_v1/{P}/c18_results.json' if os.path.exists(f'c7_v1/{P}/c18_results.json') else f'c7_v1/{P}/c15_results.json'
    if not os.path.exists(f): continue
    print(f'[{P}: using {f}]')
    rows = json.load(open(f)); fams = sorted({r['family'] for r in rows}); regs = ['operational', 'dither A2.0 s0', 'dither A2.0 s1', 'dither A2.0 s2', 'dither A0.5 s0', 'dither A1.0 s0', 'designed']
    pers = rows[0]['test']['test_in']['persistence']; pers_out = rows[0]['test']['test_out']['persistence']
    print(f'\n== {P}: in-support persistence {pers:.2f} K, out-of-range persistence {pers_out:.2f} K, step truth {np.mean(rows[0]["step"]["truth_per_start_24h"]):.2f} K')
    print(f"{'family':12s} {'regime':14s} {'in-RMSE':>16s} {'in-range':>9s} {'out':>6s} {'pred':>6s} {'step24 ratio':>18s} {'cov op/in':>10s}  n")
    for fam in fams:
        for reg in regs:
            R = [r for r in rows if r['family'] == fam and (regime(r['train']) == reg or (reg == 'designed' and regime(r['train']) == 'designed'))]
            if not R: continue
            ins = [r['test']['test_in']['rmse'] for r in R]; win = np.concatenate([r['test']['test_in']['per_window'] for r in R])
            ci = boot_ci(np.array(win) ** 2); ratio = [r['step']['ratio_24h'] for r in R]
            S_ = np.array([r['step']['per_start_24h'] for r in R]); tstart = np.array(R[0]['step']['truth_per_start_24h']); ns, nk = S_.shape
            rb = []                                                       # hierarchical bootstrap: resample seeds (rows) and start states (columns)
            for _ in range(1000):
                i = rng.choice(ns, ns); j = rng.choice(nk, nk); rb.append(np.sum(S_[i][:, j]) / (ns * np.sum(tstart[j])))
            entry = dict(period=P, family=fam, regime=reg, n=len(R), in_rmse=float(np.mean(ins)), in_sd=float(np.std(ins)), in_ci=(np.sqrt(ci[0]), np.sqrt(ci[1])),
                         in_range=float(np.mean([r['test']['test_in']['rmse_inside'] or np.nan for r in R])), out=float(np.mean([r['test']['test_out']['rmse'] for r in R])),
                         pred=float(np.mean([r['test']['test_operational']['rmse'] for r in R])), ratio=float(np.mean(ratio)), ratio_sd=float(np.std(ratio)),
                         ratio_ci=(float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))),
                         cov_op=float(np.mean([r['coverage']['operational']['6-24h'] for r in R])), cov_in=float(np.mean([r['coverage']['test_in']['6-24h'] for r in R])),
                         cov_in_range=float(np.mean([r['coverage'].get('test_in_inrange', {}).get('6-24h') or np.nan for r in R])), pred_sd=float(np.std([r['test']['test_operational']['rmse'] for r in R])))
            summary.append(entry)
            print(f"{fam:12s} {reg:14s} {entry['in_rmse']:5.2f}±{entry['in_sd']:4.2f} [{entry['in_ci'][0]:.2f},{entry['in_ci'][1]:.2f}] {entry['in_range']:8.2f} {entry['out']:6.2f} {entry['pred']:6.2f} "
                  f"{entry['ratio']:5.2f}±{entry['ratio_sd']:4.2f} [{entry['ratio_ci'][0]:.2f},{entry['ratio_ci'][1]:.2f}] {entry['cov_op']:4.2f}/{entry['cov_in']:4.2f}  {len(R)}")
json.dump(summary, open('c7_v1/summary.json', 'w'), indent=1)
