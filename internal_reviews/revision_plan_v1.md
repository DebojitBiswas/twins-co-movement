# Revision plan v1 (2026-10-08) — every red-team v0 point mapped to an action and a status

Legend: DONE / RUNNING / QUEUED / TEXT (fix in writing) / DECLINE (with reason)

## Thesis and theory
- Headline false for the direct method (control M2; stats M11; ML M2). RUNNING c13 (ARX/PEM/grey on v0 tests: ARX(24) 1.03/0.90 K beats
  persistence; grey-box 0.4× step; ARX 2× step) and c15 v1 (all families, 3 periods, in-support). TEXT: reframe — the failure belongs to the
  simulation-error recipe and modern neural one-step recipes, not to operational data; classical direct method as comparator; title changed.
- Identity classical (control M1; ML M1). TEXT: state as Pearl/OVB and Forssell–Ljung's OE bias applied to learned simulators; drop "we derive";
  contribution = empirical demonstration on modern families + decision consequences + protocol. Conditional identity (warm-up on y, x) stated.
- One-step evidence confounded by noise injection (control M3). RUNNING c5d (no noise, d=0 and d=0.8, ARX(20) yardstick); v1 includes both variants.
- Out-of-support evaluation (stats M1; control M8). DONE protocol v1: within-support step from operational contexts primary; in-support test
  schedule (85 % / 70 % in range, stratified); planning target inside support (c12 target 21.0 → 21.8).
- Certificate overclaimed, hard-coded, wrong denominator (control M5; stats M3; ML M7c). TEXT: downgrade to an ordinal check requiring a declared
  exogenous excitation; QUEUED script computing plain and partial coherence with CIs from the v1 dither logs (3 seeds × 3 periods); normalise by
  the truth from the same contexts.
- Upstream/downstream (control M6; stats M10). TEXT: "closer", not "answerable"; disclose v1→v2 query change; QUEUED family of queries with
  designed-data reference on v1 data (after c15).
- Overshoot law (control M7f): TEXT ordinal; formula caveat (integral action, non-steady state).

## Baselines and recipes
- Direct PEM with noise model, grey-box PEM (control M2/M10; ML M3): RUNNING c13, c15.
- HP search / larger models / 5 seeds (ML M3; stats M4): v1 uses 5 seeds; QUEUED a small HP sweep on designed validation for GRU/TF (hidden
  48/96, lr 1e-3/3e-3, steps) — after c15 P1.
- Chronos capability (ML M3; control M7b; stats M9): RUNNING c11b sweep with a pre-stated pass criterion (≥ 70 % of the step); commercial
  Chronos fails the control → ED only. TEXT: "better than any model" removed; dither does not repair Chronos stated in main text.
- IV done properly (control M4; ML M7b): RUNNING c7c control-function; QUEUED DeepIV-style moment loss if CF also fails.
- Pessimistic MBRL planner + feedback (ML M4/M6): RUNNING c12 (bootstrap ensembles, MOPO-style λ, receding-horizon MPC).
- Conformal (stats M5; ML M5): v1 calibrates on matched-weather operational days of the test period; designed-twin control reported next to
  the operational number; finite-sample caveat; QUEUED ACI/EnbPI variant.

## Replication and statistics
- n = 1 everywhere (stats M4/M6): v1 = 3 periods × (1 operational + 3 dither-seed + 2 designed logs) × 5 seeds; block bootstrap over
  non-overlapping starts; cross-period ranges. Synthetic: QUEUED 5 log seeds per cell for Fig 1. Pancreas: QUEUED 3 meal seeds per patient +
  adolescent/child cohorts (n ≈ 30), exact tests and bootstrap CIs.
- Planning counts (stats M2): TEXT: sim-GRU 7/12; v0 families removed; pure-feedback regime discussed honestly (plans marginally better than
  doing nothing, belief roughly right).
- Pancreas (stats M7/M8/M13; ML M9; control M7c): TEXT: horizons stated (10/10 at 2 h, 5/10 at 4 h), Spearman with CI, TIR of training logs,
  reference relabelled "true effect"; QUEUED: drip protocol fixed (stop at dose), designed-data linear-twin control, validated transformer
  twins in the calculator, published controller (simglucose basal-bolus) as a second loop, cohorts.

## Reporting
- Post-hoc register in main text (stats M9/M12; ML M10): TEXT: one paragraph in Methods; P1 scored on both metrics.
- Numbers audit stale/untraceable (stats m1): numbers_audit.py extended to every number; rerun before each draft.
- Lineage paragraph (ML M1; control minor 2): TEXT: causal imitation (de Haan 2019; Swamy 2022), delusions (Ortega 2021), confounded OPE
  (Bruns-Smith 2021; Namkoong 2020; Tennenholtz 2020), TS treatment effects (Bica 2020; Melnychuk 2022), anchor regression, DeepIV/DFIV,
  MOPO/MOReL/COMBO, least-costly design (Bombois 2006), building ID (Bacher & Madsen 2011; Privara 2013).
- Title: "Learned simulators trained on operational data learn the co-movement the controller induces, and the classical remedy was
  forgotten" (working).
- DECLINE: real-data quantitative test (no suitable public dataset with a logged actuator intervention identified yet) — stated as a limitation;
  search continues.
