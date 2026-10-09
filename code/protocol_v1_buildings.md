# Building protocol v1 — written 2026-10-08 after red-team v0, BEFORE any v1 data or model exists

## Why a v1
Red-team v0 established: (a) the designed test schedules lie largely outside the operational output range (74–80 % of samples above the
operational maximum), so v0's do-RMSE / conformal / Fig 5d results confound extrapolation with the confounding mechanism; (b) every v0 building
number rests on one operational log, one dither log per amplitude and three training seeds; (c) a plain ARX(12) trained on the operational log
beats persistence on the v0 interventions (1.11 vs 1.70 K), so the classical direct method is a mandatory comparator and the claim must be about
the model class / recipe, not about operational data as such.

## Fixed design (bestest_hydronic_heat_pump; commercial building only if time allows)
Periods (train 4 weeks → test 2 weeks, embedded controller throughout unless stated):
  P1 train days 7–35, test 42–56 (January–February, as v0) · P2 train 45–73, test 80–94 · P3 train 300–328, test 335–349.
Training logs per period: operational (1; deterministic given weather); dither ±2 K seeds 0,1,2; dither ±0.5 and ±1 K seed 0 (P1 only, budget curve);
designed (actuator random levels around the operating region) seeds 0,1.
Test material per period (days as above):
  T-op   operational log of the test weeks (predictive error; conformal test with matched weather).
  T-in   IN-SUPPORT designed schedule: levels chosen so the zone stays within the operational range (target: ≥ 90 % of samples inside the
         training log's [min, max]; levels tuned on the emulator once, before any model is fitted, and then frozen — recorded in an addendum).
  T-out  the v0-style out-of-range designed schedule (reported as secondary; labelled extrapolation).
  Step   within-support step query from embedded operation: 10 start times spaced ≥ 48 h apart (non-overlapping), actuator held at 0.4 vs 0.7
         for 48 h; truth = emulator difference; the twins' responses computed from the SAME operational contexts.
Models (all with saved weights): ARX(p), p ∈ {2, 6, 12, 24, 48}; innovations-form PEM state space n ∈ {2, 3, 4}; sign-constrained grey-box 2R2C
(PEM); GRU simulation-error; GRU and transformer next-step WITHOUT noise injection (and with, as a labelled variant); Chronos-2 with the best
recipe from the sweep c11b (recipe fixed before v1 runs). Five training seeds for the neural families; PEM models from 6 random inits (best).
Capability criterion (pre-stated): on designed data a family's within-support step response is within ±50 % of the truth at 24 h AND its T-in
do-RMSE is below persistence; families failing it are reported but excluded from headline comparisons.
Metrics: do-RMSE on T-in (primary), T-out (secondary), predictive RMSE on T-op (same windows and persistence as the others), step response from
operational contexts (primary effect metric; relative to the truth), conformal coverage calibrated on operational days of the SAME period's test
weeks (first 7 days, stride 6 h; finite-sample caveat stated) and tested on the remaining days and on T-in.
Replication and uncertainty: unit = (period, start window); report mean ± s.d. over training seeds AND a block bootstrap (over start windows,
1,000 resamples) 95 % CI; cross-period consistency reported as the range over P1–P3.
Decision test: receding-horizon MPC (c12 protocol) with target inside the operational range (21.0 → 21.8 °C), bootstrap ensembles, λ ∈ {0, 1, 3, 10}.
Pre-stated predictions (v1):
  V1 simulation-error GRU and Chronos-2 trained on operational logs: step response < 50 % of truth in all three periods.
  V2 ARX(≥12) / PEM-SS trained on operational logs: step response within ±50 % of truth in ≥ 2 of 3 periods, T-in do-RMSE below persistence.
  V3 next-step neural families without noise injection: between V1 and V2 (we do not know; this is the open question).
  V4 dither ±2 K: GRU step response fraction increases in every period and every seed.
  V5 MPC with λ > 0 on operational-twin ensembles: tracking not better than λ = 0 by more than 20 % (pessimism does not repair bias).
Every outcome reported.

## Addenda
- 2026-10-08 (before any v1 model was fitted): the in-support test could not reach ≥ 90 % of samples inside the operational range with any
  open-loop actuator schedule (drift without feedback); the closest recipe — levels {0.1, 0.2, 0.3, 0.4}, holds 15 min–1.5 h — gives 85 % and
  is adopted as T-in. All T-in results are also reported restricted to in-range samples (rmse_inside). Tried: {0.1,0.25,0.4}/1–12 h 72 %;
  {0.15,0.25,0.35}/1–12 h 74 %; {0.2,0.3}/1–12 h 76 %; {0.1,0.2,0.3}/1–12 h 56 %; short holds 70–85 %.
- 2026-10-08 (after P2 data, before any P2 model): with P1's absolute levels the milder period P2 (mean actuator 0.15 vs 0.30) overheated
  (test_in 61 % inside, designed logs 12–15 %). Levels are now defined RELATIVE to each period's operational mean actuator ū: in-support
  {1/3, 2/3, 1, 4/3} ū, designed/out {1/3, 5/6, 4/3, 11/6, 7/3} ū (identical to P1's absolute values). P2 designed logs and tests regenerated.
- 2026-10-08 P2 regenerated with relative levels: test_in 70 % inside (P1 85 %); in-range stratification applies.
- 2026-10-08 (after P3 data, before any P3 model): P3's test weeks (days 335–349) are colder than its training weeks (mean actuator 0.34 vs 0.23),
  so levels scaled by the TRAINING mean under-heated the in-support test (37 % inside). The in-support and out-of-range test levels are now scaled
  by the test period's own operational mean actuator ū_test (the embedded controller's mean over the test weeks; P1 0.30, P2 0.15, P3 0.34),
  i.e. "around what the controller does in those weeks". Training-log levels keep the training-period scaling. P3 tests regenerated.
- 2026-10-08 P3 tests regenerated with ū_test scaling: test_in 53 % inside (narrow operational range 20.2–22.1 °C); stratified reporting applies.
- 2026-10-08 19:30 (v1.1): the step-query start times did not meet the stated ≥48-h spacing (an interleaving error gave 18/30-h gaps) and the
  emulator's Boolean override-activation inputs had been set as Real, which failed at some starts (zero effect in both arms). Both corrected:
  seven starts exactly 48 h apart per period; setBoolean. The models are unchanged; every evaluation is recomputed against the corrected truth.
