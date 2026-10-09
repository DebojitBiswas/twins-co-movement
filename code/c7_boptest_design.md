# C7 — BOPTEST design (written 2026-10-06 17:10, before any BOPTEST data exist)

## Why BOPTEST
IBPSA Project 1 BOPTEST (commit 0f8a467, cloned 2026-10-06): community benchmark of Modelica building emulators (FMUs) with embedded
baseline controllers, used across the MPC / RL-for-buildings literature. Independent of papers A/B. Deterministic emulator ⇒ the TRUE
interventional effect is computable exactly: re-initialise at the same start time, same weather, run with and without the intervention.

## Test cases (primary → secondary)
1. bestest_hydronic_heat_pump — residential, air-source heat pump, embedded PI on zone temperature; the most used RL/MPC case.
2. singlezone_commercial_hydronic — commercial, hydronic, occupancy-driven internal gains.
3. twozone_apartment_hydronic — two zones (coupling = a second hidden state).
Exact input/measurement point names to be read from GET inputs/measurements before any run, and recorded here (addendum).

## Regimes
- OPERATIONAL: embedded baseline controller, no overrides; heating season, 4 weeks per case; log at 15 min: zone temperature (output),
  heat-pump modulation / valve / supply-water command (input), outdoor temperature and global solar (logged drivers). NOT logged:
  internal gains / occupancy, wall/slab temperatures (hidden disturbances / states), as in a real building's BMS log.
- DITHERED (repair): embedded controller, plus a small exogenous offset on the zone set-point (if overwritable) or on the control signal,
  random steps held 1–6 h, amplitude A ∈ {0.5, 1, 2} K (or ±10/20/40 % of modulation), logged.
- DESIGNED (capability control): input overridden by a random binary/multilevel sequence (pulses 1–24 h), 4 weeks.

## Query (truth exact)
Effect of the input on zone temperature: from N_s = 20 start times in the test weeks, run the emulator twice from the same state — with the
operational controller replaced by (a) a constant input u0, (b) u0 + Δ step (and a designed multilevel sequence) for 24 h — difference
= true effect trajectory. Twins predict the same difference by rollout given the logged drivers.

## Models (each with designed-data capability control; ≥ 3 seeds)
Linear simulation-error twin (filter bank); grey-box RC (2R2C, sign-constrained); GRU (simulation error); Transformer (one-step, c5c
recipe); neural ODE. IV / two-stage versions on the dithered logs.

## Pre-specified predictions
- P1: in the OPERATIONAL regime, every model family whose designed-data control passes has a heat-input effect error ≥ 50 % of the true
  effect's RMS at 6 h, for every test case.
- P2: the learned effect is closer to the controller-implied response G_op (law; C fitted from the log) than to the true G.
- P3: planning (pre-heat / set-back schedule optimised on the twin) applied to the emulator does worse than the embedded controller
  for ≥ 2/3 of operational-data twins and better or equal for designed-data twins.
- P4: sign-constrained grey-box twins are less affected than unconstrained ones (structure blocks the controller solution).
- P5: IV twins on dithered logs reduce the 6-h effect error by ≥ 30 % relative to plain twins at the largest dither amplitude.
Every outcome reported.

## Addenda
- 2026-10-06 ~21:00 (after the LINEAR-twin results only; before any neural twin): (1) reference signal corrected to oveTSet_y (the
  set-point the embedded PI tracks, 21.2 / 20.5 °C), not reaTSetHea_y (KPI band); dither added to that schedule. (2) Designed levels
  narrowed to {0.1 … 0.7} (v0 {0 … 1} overheated the zone to 27 ± 4 °C). (3) PRIMARY METRIC CHANGED: the 24-h step query branched from
  embedded operation contains an override-engagement transient (dead time ≥ 1 h, zero effect at 1 h) and disagrees with a designed-data
  twin that predicts a held-out designed log to 0.38 K (zone s.d. 2.09 K). Primary metric is now do-RMSE on held-out DESIGNED test
  schedules in the test weeks (days 42–56, seeds 10/11), given weather; the step query is reported as secondary. This change was made
  after seeing linear results (operational twin: 0.24 K in-regime, 3.53 K on a designed log) and is disclosed as such.
