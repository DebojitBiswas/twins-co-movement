# Round 4 — statistics referee (2026-10-09). Verdict: MINOR REVISION, not acceptable as it stands. Most damaging: MAJOR 1 — the dither-log
"law test" drops the inverse-controller term the paper uses elsewhere; with it, agreement 6/11 within 0.1, P3 off by 0.22–0.31; across 11 logs
the law predicts the fraction no better than a constant (ρ=0.51, p=0.11; RMS 0.100 vs 0.096); n is nearer 3 than 11; band as coded is
3.2–24 h not 3 h–2 d. Present as a comparison; delete "holds to within 0.2".
MAJOR 2 capability: ED Table 3 marks tf_noise P2 0.50 as fail, Results calls it borderline pass; PEM 1.47 is outright pass; unregistered rule →
apply as generated; floor 0.65. MAJOR 3 ED Table 4 "never inside" false (0.49); selection column (designed-twin error) missing. MAJOR 4 pancreas
comparator: NaN exclusions undisclosed (10/5/7 of 68); unit of replication = 23 patients, not 68 cells; bolused medians ≈0 → "estimates ≈0",
not "wrong sign"; "not a feedback bias" untested → ARX on designed pancreas logs. MAJOR 5 abstract ranges (0.45–1.12; 0.32–0.78 / 0.14–0.29;
0.9–1.0 K; "orders 12–48 and state-space"). MAJOR 6 ARX(48) not flat in P3. MAJOR 7 Fig 2 caption persistence mismatch (operational
persistence 0.66/0.65/0.51). MAJOR 8 CIs for +0.1 (free from per_start_24h: 0.26 [0.17,0.37], 0.15 [0.09,0.20], 0.36 [0.23,0.61]; paired
+0.1−+0.3: +0.04, 0.00, +0.10 → "same fraction", not "smaller shortfall") and for the 6-h column (needs per-start rescoring).
MINOR m1 ARX/PEM CIs are 7-start bootstraps; m2 "energy" = summed modulation; m3 band; m4 "matches"; m5 controller energy 0.42; m6 two
pancreas numbers; m7 dosing n 22/23; m8 1.35/1.01/0.82 is ARX(48); m9 masked vs unmasked persistence; m10 operational coverage GRU-only;
m11 "within 24%"; m12 Fig 4 s.d.; m13 bootstrap statistic.
