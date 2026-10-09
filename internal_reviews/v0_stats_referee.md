# Red-team v0 — statistics / methods referee (2026-10-08). Verdict: MAJOR REVISION. Most damaging: M1.
MAJOR
M1 Building headline metrics evaluated OUT OF OUTPUT SUPPORT: operational log 20.99±0.46 °C (max 23.38); designed tests 24.7±2.0 / 26.6±2.8 °C —
   74 % / 80 % of samples above the operational maximum → do-RMSE, conformal 0–4 % and Fig 5d confound extrapolation with confounding; metric itself
   post hoc (register 3). Planning target 22.5 °C above the operational 99th percentile. FIX: in-support step query (truth.json protocol) primary with
   per-start CIs; designed test schedule constrained to the operational range; stratify; planning target inside support.
M2 "29 of 36" pools v0 families that failed capability control; sim-GRU alone 7/12; under pure feedback sim-GRU plans are BETTER than doing nothing
   (2.42–2.74 vs 3.0) and the belief is roughly right (2.8) → re-examine "does not know" for that regime; "up to 23 K" from superseded v0 gru_1s.
M3 Fig 2c hard-coded (no script computes building coherence); normalised by designed-twin step not truth (→ 0.26/0.63/0.74/1.09 GRU); steps from
   out-of-support designed contexts vs truth from operational states; band post hoc; partial coherence never computed; n=1 log per amplitude.
M4 Real n is 1 everywhere: synthetic one noise realisation per cell (seed 7) with nested N; building one weather period, one operational log, one dither log
   per amplitude, overlapping starts; pancreas one log per patient-config, deterministic linear twin; ± reported is the SMALL one (seeds), s.d. over
   starts 0.3–0.9 K. FIX: unit of replication for every number; mean±s.d. over seeds AND block-bootstrap CI over natural units; replicates.
M5 Conformal not finite-sample (n=8 overlapping calibration windows; needs ≥30 non-overlapping), temporal shift not separated (op-test adjacent to
   calibration; interventions 2–5 weeks later and out of support); designed twins cover 16–60 % (T2 "keep coverage" FAILED; ED scores T2 met); 0–4 vs 0–5.
M6 Minimal replication: ≥3 operational periods, ≥3 dither seeds/amplitude, non-overlapping starts; synthetic ≥5 log seeds; pancreas ≥3 meal seeds, add
   adolescent/child (n≈30); CIs everywhere; finish or drop the commercial building.
M7 Pancreas 3/9 vs 0/10 severe: Fisher p=0.087; BG0 175–359; drip continues (reference ends 215; adult#003 739); 7/9 doses at the 10-U cap; "designed-
   data reference" is the TRUE effect; no designed-data capability control for the linear twin.
M8 10/10 is at 2 h; 4 h 5/10; Spearman reactive 0.20 (p=0.61, n=9) — "uncorrelated" unsupported; bolused loop 0.89; reactive training logs 8–30 % hypo.
M9 Chronos "better than any model" contradicted (TF 0.02–0.03 on its hold-out); hold-out sets not comparable; two persistence baselines for one test set;
   commercial designed-Chronos fails control (step 0.23/−0.09/0.05 vs truth 0.20); dither does not help Chronos — say so in main text.
M10 Upstream/downstream: queries hand-picked after a failed v1 (AC still beats persistence 0.41 vs 0.93); absolute effect errors reversed (CL/A 0.38 >
    AC/B 0.32); contrast p≈0.05 before overlap correction; r ⟂ v doubtful (schedule and gains both functions of time of day); audit stale.
M11 "Not model class" contradicted: TF learns 55 %, GRU 25 % on the same log; commercial GRU over 4–6×, TF under; restrict to simulation-error
    population limit; T4 ordinal.
M12 Pre-registered P1 scored on the post-hoc metric; on the original step query the TF is within ~45 % → ambiguous; commercial not scored.
M13 ED Fig 6 pancreas neural: 2 seeds, enormous divergence (adult#010 −118 vs +260); no pass criterion defined anywhere.
MINOR m1 untraceable numbers (coherence, TF fractions, comfort/energy, C9, 23 K, −1/Ĉ 0.1 K, Chronos persistence); m2 0–4/0–5, 4.4/5.4×, 1.9±0.7; m3 init-only
ensembles; m4 adult#009 kept at 43 % hypo in d=0.7 → sensitivity; m5 post-hoc items missing from register (dosing rule/cap, plan target, Chronos HPs,
capacity steps, conformal split); m6 TF noise 0.1 not in sweep; m7 caption counts; m8 commercial normalise by truth 0.20±0.11; m9 designed-data rollout
error 1.6–1.8 in capacity sweep; m10 Twin Houses analyst d.o.f.; m11 QUALITY.md promises unmet; m12 seed counts.
