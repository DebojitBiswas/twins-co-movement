# Revision plan v2 (2026-10-08, after red-team round 2) — thesis, evidence and text changes

## Thesis (final form)
"Both estimators are biased by the operational regime, differently and by different amounts: simulation-error training (and input-noise
injection) loses most of the actuator effect — toward the controller's co-movement; the classical direct method over-estimates it from these
logs and fails under anticipatory control (pancreas); a small logged excitation repairs the feedback-dominated case for every family."
Title candidate: "Simulation-error training teaches digital twins the controller's co-movement, not the system's response".

## Evidence changes (compute)
- DONE Boolean override fix; step truth v1.1 (7 starts, 48 h apart; 2.55/2.97/2.70 K). RUNNING c18 re-evaluation of all saved models (adds
  in-range-masked coverage, step in-range fractions). QUEUED MPC rerun with the fix, both targets. RUNNING HP sweep for next-step families.
- DONE −1/Ĉ test (c19): −0.19/−0.85/−0.59 K; law with schedule coherence 1.58/1.17/1.07 K. DONE pancreas ARX (c20): fails (4–13× over; sign wrong).
- TODO after c18: operational/designed ratio column; pass/fail table under the TWO-clause capability criterion (restored, with addendum);
  ARX all orders in Table 1 (already computed); excitation per period (coherence 0.64/0.53/0.50) next to transformer fractions — demote
  "informativity" to "consistent with"; bootstrap: hierarchical over starts and seeds (fix analyze_v1); in-range-only RMSE primary in Fig A.
- TODO MPC with ARX/PEM twins (torch port of ARX recurrence) — if time; otherwise state as a limitation.
- TODO grey-box fitted by sign-constrained simulation error (OE) — c13 variant; if it passes capability, add to the table.
- DECLINE for now: DeepIV, ACI/EnbPI, commercial building under v1 — stated as limitations.

## Text changes
Abstract (≤200 words): new thesis; ranges from the regenerated audit only; "insulin raises glucose" qualified (2 h, bolused loop); both biases;
dither cost and benefit at the same amplitude; Chronos narrowed (one model, full fine-tuning, our budget). Intro: "regression relative to practice" →
"replaced rather than extended"; cite recipe lineages (PETS/MBPO/MOPO = one-step ensembles; simulation-error = neural system ID / neural ODE /
physics ML; noise injection = GNS/MeshGraphNets/scheduled sampling). Results: Fig A x-axis = operational test-week error (pred), in-range designed
error to ED; lead with the within-architecture ablation (clean vs noise, same TF / same GRU); ARX P3 under-estimate stated; PEM designed-data
over-estimation subtracted; designed-data recovery 0.54–1.30 with borderline cells; conformal in-range-masked numbers with ARX/designed controls;
MPC paragraph: in-range target first, normalised to the designed-twin MPC, paired per-start differences, no "only"; pancreas: v1 Methods, exclusions,
BG0 range, one-sided Fisher, ARX comparator fails, cap-hit fraction. Methods: conditional error formula; displayed identity as eq. (1); figure
numbering; pre-stated orders note. ED: pass/fail table; register items 16–18 (override bug, start spacing, capability clause).
