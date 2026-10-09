# Paper C — pre-registration C3: the mixing law on the GPU thermal plant (dither sweep)

Written 2026-10-06, **before any of the runs below exist**. Extends paper B's run-4 pre-registration (paperB/preregistration_run4.md),
whose protocol and predictions P1–P5 stand unchanged; run 4 is one cell of this sweep. Nothing below may be edited after the first run
starts; deviations go in a dated addendum.

## Hypothesis (from c1/c2, synthetic)
A simulator trained by simulation error on closed-loop data converges, frequency by frequency, to
    G_hat(ω) = γ²(ω) G(ω) + (1 − γ²(ω)) (−1/C(ω)),   γ²(ω) = |coherence(u, r)|²,
where G is the plant (power → die temperature), C the controller (known: the script's PI, k_p 0.1, k_i 0.003 per s, in duty units, mapped
to power by the calibration curve), and r the logged set-point. With r constant (γ² = 0) the twin is the negative inverse controller.

## Runs (fixed)
Plant, power cap (130 W), PI gains, hidden CPU schedule (seed 1) as in run 2 / run 4. Each regulated run 4 h. Set-point r: random steps,
hold 120–600 s, uniform in [74 − A, 74 + A]; the step sequence uses the script's seed + 200 = 201 in every run (same timing, different amplitude):
- R0: A = 0 — existing run 2 (regulated_20261005_065649.csv; no new run)
- R1: A = 1.5 K
- R2: A = 4 K
- R3: paper B run 4 exactly as pre-registered (uniform 64–78, i.e. A = 7 about 71)
Designed test sequences D1 (existing), D2 (seed 2), D3 (seed 3), as in run 4.
Run order: R3 (= run 4), D2, D3 (already running, see addendum), then R1, R2. Commands: see bottom.

## Analysis (fixed)
2-s resampling. Plant reference G: linear output-error (2-node) fit on D1 (the paper B designed-data model; DT 2 s).
γ²: Welch coherence of (duty, r) on each regulated run, nperseg 1024 samples (≈34 min), averaged over the band 1/600–1/60 Hz.
Twins fitted on each regulated run only: (a) linear simulation-error twin (filter-bank, as c4_house_lintwin, inputs: power only),
(b) GRU and (c) neural ODE from expB10 (unchanged settings), 3 seeds; (d) two-stage IV linear twin (duty projected on lagged r).
Primary quantity: each twin's 60-s and 300-s response to a +50-W step, and its do-RMSE on D1–D3.

## Predictions
- Q1: In R0 the linear twin's 60-s step response is closer to the −1/C prediction than to G (sign: ≤ 0 or < 25% of G).
- Q2: Across R0–R3 the linear twin's 300-s step response increases monotonically with γ², and lies within ±30% of G of the law's
  prediction γ²G + (1 − γ²)(−1/C) in at least 3 of the 4 runs.
- Q3: GRU/NODE ensemble-mean responses follow the same ordering (Spearman ρ ≥ 0.8 with γ² over R0–R3).
- Q4: The two-stage IV twin is within ±30% of G at 300 s in every run with γ² ≥ 0.2.
- Q5: Ensemble spread of GRU/NODE responses is smaller than their error vs G in R0 and R1 (overconfidence).
Every outcome will be reported.

## Commands
R3 (= paper B run 4), D2 and D3 were started by the author at 2026-10-06 07:55 as one chain (regulated 64–78 → rolbs seed 2 → rolbs seed 3,
130-W cap). R1 and R2 are queued to start when that chain exits, same cap, same environment:
    cd ~/ndo/ndo_dyn/paperB/thermal
    nohup sh -c "while kill -0 970409 2>/dev/null; do sleep 60; done; \
                 python3 thermal_experiment.py --phase regulated --hours 4 --sp-min 72.5 --sp-max 75.5 --seed 1 && \
                 python3 thermal_experiment.py --phase regulated --hours 4 --sp-min 70 --sp-max 78 --seed 1" > runsC.log 2>&1 &
    # afterwards: sudo nvidia-smi -pl 280

## Addenda
- 2026-10-06 09:25 (at writing): this document was written ~1.5 h AFTER run R3 (= paper B run 4) started (07:55) and before any of its data
  were analysed or plotted; R3's predictions here are therefore not strictly pre-data. R1, R2 and the R0 (existing) analysis are. Note also
  that paperB/preregistration_run4.md is headed "Written 2026-10-07" but its file was last modified 2026-10-06 08:03:59, ~8 min after run 4
  started; the order of that pre-registration and run 4 cannot be established from file timestamps.
- R0 file: regulated_20261005_065649.csv (PI, 74.00 ± 0.32 °C, 14,399 rows).
- 2026-10-06 16:30: paper B run-4 P1 scored (only needs run 4): 5-min-smoothed die s.d. 3.26 K (threshold ≥ 1.0) — MET. (Same estimator on
  run 2 / run 3: 0.069 / 0.056 K; paperB notes quote 0.044 / 0.035 with a different smoothing — to reconcile.)
- 2026-10-06 16:30: CPU-load disclosure. The hidden disturbance of this testbed is case/CPU heat. CPU-heavy paper-C simulations (c1, c2,
  c5, c6, up to 22 cores) ran during parts of run 4 (R3), D2 and D3. Under the protocol the latent is "the CPU schedule plus whatever else
  happens on the machine", so this is within the definition, but it is an unscheduled addition; CPU temperatures are logged (k10temp
  columns) and its timing can be reconstructed. No CPU-heavy jobs will be started during R1/R2 (≈19:55–03:55); scoring
  (paperC/score_run4.sh) is queued to start only after the whole chain has exited.
- 2026-10-06 16:30: code change before scoring, file selection only: expB7_thermal.py gained REG_FILE / ROL_FILE / B7_OUT / B7_MODEL_DIR,
  expB10_learned_baselines.py gained run 4 (regulated_20261006_075544.csv, W 300 / PRE 90 at 2 s as run 3), ROL_FILE and B10_OUT. Model
  classes, training settings and seeds unchanged. D1 = rolbs_20261005_105650.csv, D2 = rolbs_20261006_115546.csv, D3 = rolbs_20261006_155548.csv.
- 2026-10-06 16:50: WITHDRAWN for paper C by the author's decision: the GPU testbed stays with paper B (avoids data overlap between papers).
  R1 and R2 were cancelled before they started (no R1/R2 data exist). Paper B's own run-4 pre-registration is unaffected; its scoring
  (score_run4.sh) now starts after D3 (≈19:55). Paper C's realistic testbed: BOPTEST (IBPSA building emulators).
