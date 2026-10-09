# Paper C — plan (started 2026-10-06)

## Working title
"Operational data cannot teach a digital twin what an intervention will do — and a certificate computed from the log says when it can"

## Why this, and not a third version of A/B
Paper A: exact theory of the bias in a linear–Gaussian class (an estimator hierarchy). Paper B: exact inference + structure; its own
red-team reversed the controller-row thesis (B9). What survived **every** domain (B10, B11) is a statement about DATA, not about estimators:
every model class that is accurate when trained on designed data (NODE 0.28 K house, 0.75 K GPU) fails when trained on regulated data from
the same plant (3.0 K house, 3.8–4.5 K GPU); flexible models with equal training loss disagree about the intervention (ring: 0.40 vs 3.01).
That is a general claim about learned simulators / digital twins / world models trained on closed-loop logs — which is how almost all of them
are trained (buildings, data centres, grids, vehicles, insulin pumps, process plants). It is the broadest claim our evidence can support.

## Thesis (three parts — each must survive or the paper changes)
1. **Failure that scale does not fix.** Trained on operational (closed-loop) logs, learned simulators of every capacity are confidently wrong
   about interventions: interventional error plateaus (or grows) with data size while predictive error goes to zero, and ensemble/epistemic
   uncertainty shrinks — the error is invisible to the model's own uncertainty.
2. **A log-only certificate.** Interventional error is governed by two quantities computable from the log *before any intervention*:
   (a) the fraction of input variation not explained by the observed history (exogenous excitation share), and (b) the regulated output's
   excursion at the intervened mode's time-scale relative to sensor+process noise (excitation number Ξ). Predicted error collapses onto one
   curve across plants, controllers and model classes.
3. **A cheap remedy, tested prospectively.** A small randomised reference/input dither, *logged and used as an instrument* (instrumented
   simulation-error training), restores interventional accuracy at a quantified operating cost (e.g. ≤ x % comfort/energy). Pre-registered.

## Evidence plan
| phase | what | kill / pivot criterion |
|---|---|---|
| C1 synthetic | 2-node thermal-type plant, PI controller with tightness K, hidden disturbance, sensor quantisation q, dither σ_r; data size N 10^3–10^6; models: ARX/PEM, grey-box Kalman PEM, GRU, NODE, ensembles; query = designed open-loop sequence | if interventional error of flexible models decreases ~1/√N without dither → thesis 1 dead |
| C2 theory | bias/variance decomposition for closed-loop learned simulators; certificate (a),(b); IV-consistency of instrumented training | if certificate does not rank C1 cells (Spearman < 0.8) → rework |
| C3 GPU testbed | sweep controller tightness × dither amplitude (≥ 8 regimes, 4 h each), designed test sequences D1–D3; run 4 is pre-registered already | pre-registered predictions; report whatever happens |
| C4 real data | Twin Houses (O5 excited / N2 frozen: certificate must predict the 2×2), ring road (AV intervention), Exp 2 house | certificate must predict which direction works |
| C5 high-stakes domain | artificial pancreas (UVA/Padova-type simulator, e.g. simglucose): meal = hidden confounder, insulin controller = policy, query = changed basal/bolus | — |
| C6 write-up | Nature-format: ≤ 3,000 words main, 4 figures, Methods, Extended Data | — |

## Honest venue assessment
Nature main rarely takes ML-methods papers; a cross-domain empirical law + prospective test + practical remedy is the shape that can.
Realistic targets in order: Nature → Nature Machine Intelligence / Nature Communications → PNAS. Classical persistent-excitation /
closed-loop identification (Ljung; Gustavsson–Ljung–Söderström 1977; Van den Hof) is the prior art we must clearly go beyond: learned
simulators, scaling, uncertainty blindness, log-only certificate, prospective cross-domain validation.

## Log
- 2026-10-06: plan written. Starting C1.
- 2026-10-06 C1 linear grid (432 cells, c1_results_lin_full.json): thesis 1 as first written is FALSE for one-step high-order ARX under pure
  feedback (d=0: ARX20 → 0.08 at N=2e5 — classical: direct method with flexible noise model is consistent). It HOLDS when the controller
  anticipates the disturbance (d=0.8: ARX20 ≈ 1.0 for all N without dither) and for simulation-error (OE) training in EVERY cell (0.9–1.1).
  Two-stage with logged reference: 0.11 at N=2e5, sr=1.
- 2026-10-06 C2: mixing law G_hat = γ²G + (1−γ²)(−1/C) predicts the OE-learned effect to 0.09–0.18 (relative to the true effect), incl.
  the wrong sign at kp=0.5. Thesis reframed: "a twin trained on operational data learns the (negative inverse) controller, not the plant".
- 2026-10-06 C1 GRU sub-grid (24 cells × 5 members): rel. error 1.03–1.27 in every cell, independent of N (2e3–2e5), d, kp, sr=0.2; learned
  effect has the WRONG SIGN (+0.2…+2.2 vs −4.6…−6.0); ensemble spread 0.01–0.17 ⇒ 7–80× overconfident. Controls running (c1b).
- 2026-10-06 C1b controls (c1b_results.json): GRU on open-loop designed data converges (0.44→0.23→0.10, N 2e3→2e5) → capacity fine.
  Closed loop with large dither (sr=1): 0.89–0.94 (mixing law: γ²≈0.18 at slow band). IV-GRU (trained on u projected on lagged r): 0.30 for
  d=0 and d=0.8, N 2e4 and 2e5 (3× better; plateau at 0.3 to investigate — likely support/extrapolation of û).
- 2026-10-06 C4 house (c4_house_invctrl.py): O5 thermostat fitted from regulated log (R²≈1.00, ARX(3,3)/(6,6)); −1/Ĉ 500-W step response
  −0.5…−1.9 K at 1 h, −0.7…−0.8 K at 24 h. B10 learned: GRU s0 −0.13→−1.15, s1 ≈−0.12, s2 ≈0; NODE +0.12…+0.52. Truth (B9, N2 designed)
  +1.7…+2.2 at 24 h. Qualitative: all learned responses between −1/C and 0, none near the plant. Quantitative mismatch → need the law
  WITH measured disturbance inputs (weather) and causal/finite-memory projection. TODO.
- 2026-10-06 C4c house: linear simulation-error twin on regulated data — 24-h heater step varies −0.8…+1.6 K with the choice of regulated
  phases (O5 pooled −0.44/−0.83, O5 const +1.57, N2 pooled +1.42, N2 const +0.35); on designed data stable +2.0…+2.2 (truth). Naive
  certificate (variance share of u explained by logged drivers, 0.85–0.97) FAILS on O5. Correct quantity is the partial share given the
  model's other inputs (≈0 with r constant ⇒ −1/C). Reading: real regulated house data leave the heater effect unidentified (arbitrary),
  quantitative law cannot be validated on 13 days of uncontrolled data → GPU pre-registration C3 written (preregistration_C3_gpu.md).
- 2026-10-06 C2 general law: G_hat = γ²G + (1−γ²)G_op, G_op = (G F + H)/(F − C H) = ratio of disturbance-driven output to
  disturbance-driven input (−1/C for pure feedback). "A learned simulator learns how inputs and outputs co-move; in operational data the
  co-movement is set by the controller's response to disturbances." Collapse over 28 synthetic cells (c2_collapse.py): GRU error vs truth
  median 1.13, vs law 0.20; OE 0.95 vs 0.16. N=2000 dithered cells failed due to coherence nperseg > N (fixed, rerunning).
- 2026-10-06 GPU: author started run 4 (= C3 R3) + D2 + D3 at 07:55 (130 W). R1, R2 queued (waits on PID 970409), ends ≈04:00 10-07.
  NOTE prereg_run4.md mtime 08:03:59 > run start 07:55:44 — recorded in C3 addendum; author to add addendum to run-4 prereg.

## TARGET: Nature Machine Intelligence (decided 2026-10-06)
NMI Article: ~3,000–5,000 words main, ≤ 6–8 display items, Methods, Extended Data ≤ 10, code + data availability, reporting summary.
What NMI needs beyond what we have:
1. ML-community relevance: the result must be shown for the model families ML people use — GRU/NODE (have), **transformer sequence
   simulators**, and **world models of offline model-based RL** (behaviour policy reacting to unobserved state = our setting). Show the law
   predicts what a world model learns and that the planner exploits the wrong sign (decision consequence, not just RMSE).
2. Lineage stated up front: classical closed-loop identification (Ljung; Gustavsson–Ljung–Söderström 1977; Van den Hof); our claims are
   the learned-simulator generalisation, the scaling/overconfidence failure, the log-only check, the prospective test.
3. ≥ 2 real domains with a quantitative test: GPU (pre-registered C3) + one of {artificial pancreas simulator (FDA-accepted UVA/Padova
   class), building at scale, data-centre}. Twin Houses as qualitative illustration only.
4. Decision-level consequence: a controller/planner designed on the twin performs badly / unsafely (e.g. insulin over-dosing, heating
   policy that overshoots) — this is what makes it matter.
- 2026-10-06 C6 pancreas v0 (simglucose adults 1–10, 14 d hybrid closed loop, meals unlogged; c6_results_v0_multdither.json):
  meal-bolused loop (d=0.7): linear twin's 1-U bolus effect at 2 h POSITIVE in 10/10 patients (mean +8 mg/dL; truth −3…−27).
  Reactive loop (d=0): twin ≈ −6 mg/dL at 4 h for every patient (CV 0.23) vs truth −8…−40 (CV 0.61), Spearman 0.15–0.28 →
  "the twin of the patient is a twin of the pump's controller". adult#009 d=0 controller failure (92 % hypo) excluded.
  v0 dither multiplied the whole rate (not exogenous) → IV erratic; v1 rerun with additive basal dither A=0.5.
- 2026-10-06 C6 v1 (additive basal dither A=0.5, c6_results.json): d=0.7 plain twin @2h +6.3 (9/9 positive) vs truth −10.9; IV −5.5
  (7/9 correct sign), median |err| 5.4 vs 13.7 (2 h), 6.7 vs 11.9 (4 h). d=0: IV −6.1 / −12.0 vs plain −4.0 / −6.7 (truth −10.9 / −18.7),
  median |err| not better (8.4 vs 6.2; 9.0 vs 8.5). Repair partial: 14 d of ±50 % basal dither with a linear twin of a nonlinear patient
  is a weak instrument. One dithered d=0.7 patient excluded (hypo ≥ 50 %). TODO: GRU/NODE twins on c6_logs; nonlinear IV; longer logs.
- 2026-10-06 C5 (c5_results.json; 200k steps, 3 seeds): DECISION: planner (+3 K target) on true plant 0.17 K, do-nothing 3.0 K.
  Designed-data GRU-sim plans 0.3–1.1 K. Closed-loop-trained twins: 30/36 plans WORSE than doing nothing (up to 23.5 K), best 1.66 K;
  the twins believe their plans track to 0.2–0.9 K where truth is 10–20 K (confidently catastrophic).
  MODEL FAMILIES: gru_1s under pure feedback 0.97 (theory's asymptotic consistency of long-history one-step learners NOT seen in
  practice; ARX20 needed N=2e5 to reach 0.18 at kp 2). Transformer (tf_1s) FAILS ON OPEN-LOOP DATA TOO (effect err 2.2–9.0) → its
  recipe is broken, closed-loop TF numbers uninterpretable. Fix in c5b_tf_check.py (steps 1500/6000 × input-noise 0/0.05/0.2) — launch
  pending (shell classifier outage).
- 2026-10-06 C5b/C5c: transformer fixed (6,000 steps + input-noise 0.1): open-loop effect err 0.09–0.33 (plan 0.4–1.3 K); gru_1s same recipe
  0.07–0.35 (0.3–1.2 K). Closed loop (c5c_results.json): gru_1s 0.43–13.3, tf_1s 0.70–36; plans 1.0–8.9 K, 22/24 worse than doing
  nothing, models believe 0.2–2.9 K. All three families (sim-GRU, 1-step GRU, Transformer): competent on designed data, fail on regulated.
- 2026-10-06 C6b neural pancreas twins (c6b_results.json, 2 seeds): designed-data control (5 pts): TF passes (med|err| 5.5/8.6 @2/4 h),
  GRU-sim FAILS (16/40, overestimates ×3) → GRU-sim pancreas numbers uninterpretable until control passes (retraining 5,000 steps).
  Reactive loop d=0: TF +155/+234 mg/dL, POSITIVE in 9/10 ("insulin raises glucose", huge); d=0.7: TF −5.8/−4.1 (truth −10/−18),
  wrong sign 3–4/10. Claim for paper: linear twin 10/10 positive at 2 h with boluses; validated Transformer 9/10 positive reactive.
- 2026-10-06 16:50 DECISION (author): GPU testbed stays with paper B. C3 GPU pre-registration withdrawn, R1/R2 cancelled before starting.
  Paper C realistic testbed → BOPTEST (IBPSA Modelica building emulators via Docker) + simglucose pancreas. Run-4 scoring for paper B
  queued after D3.
- 2026-10-06 17:10: QUALITY.md (NMI bar) written. BOPTEST cloned (commit 0f8a467), ports bound to 127.0.0.1, build queued after D3.
  C7 design + pre-specified predictions in c7_boptest_design.md (before any BOPTEST data).
- 2026-10-06 21:00 C7 BOPTEST (bestest_hydronic_heat_pump via fmpy; c7_neural.json, 3 seeds): capability control PASSES (designed-trained
  GRU 0.86–1.18 K, TF 0.48–0.97 K do-RMSE on held-out designed test schedules; persistence 1.70/2.15). OPERATIONAL twins: GRU 3.5/5.7 K,
  TF 2.0–2.7/2.6–3.7 K — worse than persistence; learned 24-h effect of +0.3 modulation: GRU 0.44–0.54 K, TF 0.6–1.3 vs designed 2.1–2.7.
  Dither monotone: GRU learned/G 0.19 → 0.46 → 0.54 → 0.80 for A = 0/0.5/1/2 K; γ² of u with the DITHER COMPONENT (r − schedule; schedule
  is synchronous with hidden occupancy, so not a valid instrument — anticipated in design) in the 12 h–2 d band 0/0.43/0.62/0.75 (band
  choice POST HOC). TF same ordering, higher (0.49 → 0.91). Linear stateless twins uninformative here (fail cross-week even on designed).
- 2026-10-07 C7b BOPTEST planning + IV (c7b_results.json, 3 seeds, 10 starts): embedded controller (target as set-point) 0.55 K.
  Operational twins plan → emulator 2.43 (GRU) / 2.98 (TF) K while believing 0.46–0.65 (overheat, mean u 0.63–0.68). Designed twins
  0.96–1.01 K. Plain dithered twins improve monotonically: A=0.5 1.21–1.23, A=1 0.79–0.87, A=2 0.67–0.70 K. IV (two-stage on dither)
  twins WORSE: 1.8–3.1 K (û has little variance → extrapolation). Design predictions: P3 first half MET (operational worse than embedded),
  second half NOT MET (designed twins 1.0 vs embedded 0.55 — open loop cannot correct disturbances; badly posed); P5 NOT MET (IV hurts).
  Practical message: log a small set-point dither and train normally; naive neural IV is harmful.
- 2026-10-07 paper B run 4 scored (preregistration_run4.md Results): P1, P2, P3, P5 MET; P4 NOT MET at 1 min (−27…−39 %), agrees at 5–10 min.
- 2026-10-07 C7 second building (singlezone_commercial_hydronic; c7_data_commercial/c7_neural.json, 3 seeds): persistence 1.36/0.90 K;
  designed twins 0.37–0.53 K; OPERATIONAL twins 2.2–2.6 K (2–3× persistence); dither monotone (GRU 1.87→1.47→1.39; TF 1.59→1.36→1.38).
  Direction of the learned-effect error differs from the heat-pump house: operational GRU OVER-estimates the +0.3 valve step (1.2 K vs
  designed 0.27) because valve and zone T are both driven by the set-point schedule (corr(u,y) +0.61 in the log) — the controller reveals
  the schedule/occupancy state (T1: E[v|u] > 0); TF under-estimates (0.11). My pre-run guess (chat, 2026-10-07) that this building's
  schedule steps would make operational twins "noticeably better" was WRONG: the schedule is synchronous with hidden occupancy gains, so it
  is not an exogenous instrument. Record as a negative for the naive "scheduled set-point = excitation" reading; supports T5/T6 wording.
- 2026-10-07 T4 overshoot law (heat pump, existing data): ordering 8/8 cells; TF magnitudes match (pred 2.27/1.96/1.77/1.22 vs obs
  2.17/1.56/1.39/1.26); GRU predicted ratios inflated (plans saturate at u ≤ 1) → ordinal law with saturation caveat.
  T7 excitation budget (heat pump): comfort cost 0.31/0.56/0.89 K, energy +4/+12/+20 %, plan error 1.22/0.83/0.69 K, learned/G 0.34/0.41/0.60.
- 2026-10-07 C11 Chronos-2 zero-shot (heat pump, known-future covariates u, weather): predictive 0.37 K on operational test weeks;
  designed tests 1.55/2.26 ≈ persistence 1.70/2.43; +0.3 step response ≈ 0.02 K (truth ≈ 2 K): in-context learning from operational history
  inherits the blindness. Fine-tunes (operational / designed / dither A1, A2; 3 seeds) running.
- 2026-10-08 C8 (T2, heat pump, c7_data/c8_conformal.json): split-conformal 90 % intervals of operational-data twins cover 80–100 % of
  operational test outcomes and 0–4 % of interventional ones (predictive RMSE 0.02–0.13 K vs interventional 2.7–4.5 K). Designed-data twins
  cover 16–60 % on interventions — also below nominal (weather shift between calibration days and test weeks) but an order of magnitude
  better; report both. T2(b) confirmed.
- 2026-10-08 C9 v1 (T6, heat pump): query A (schedule +1 K, 2-h earlier start; effect 0.83 K rms): closed-loop twin 0.59 K, chain 0.63,
  persistence 1.04 → T6(i) supported, chain consistent. Query B v1 (heat pump at 0.2 from 17–21 h) had a 0.13-K effect — too weak to
  discriminate (all twins ≈ 0.4 vs persistence 0.7, weather-dominated). v2 (decided after seeing v1 A/B numbers): actuator OFF 05–11 h
  (morning rise), and an EFFECT metric (prediction difference query−base vs truth difference) alongside trajectory do-RMSE. Prediction
  for v2 B: AC twin effect-RMSE ≥ 50 % of the effect rms; CL twin cannot see the override (≈ 100 %); chain does not rescue.
- 2026-10-08 C11 Chronos-2 (heat pump; c7_data/c11_foundation.json; 3 fine-tune seeds, 1,000 steps, lr 1e-5): persistence 1.70/2.43.
  zero-shot 1.55/2.26, pred 0.37, step 0.02; ft:operational pred 0.15 (best of all) but do-RMSE 1.72/2.42 = persistence, step 0.02 K
  (truth ≈ 2 K at 24 h); ft:designed 0.73/1.00, step 0.44/0.49 (under-responds but 20× the operational); ft:dither A1/A2 1.6–1.8/2.7–2.8
  (no interventional gain, unlike GRU/TF), step 0.28–0.31. HEADLINE for T3: a foundation model fine-tuned on operational data is the best
  predictor of operation and learns that the heater does nothing. Commercial-building Chronos run launched.
- 2026-10-08 T5 on the pancreas (A=0.5 additive dither logs): γ²(insulin, dither; 1–12 h) 0.76 for the reactive loop (d=0) vs 0.07 with meal
  boluses (d=0.7; boluses dominate insulin variance); plain-twin recovered fraction of the 4-h effect 0.46 vs 0.10 — ordering as the law
  predicts, magnitudes not quantitative (nonlinear patient, linear twin; two points only). IV 0.61±0.66 / 0.55±0.24.
- 2026-10-08 C6c twin-derived correction-bolus calculator (c6c_dosing.json; 10 adults; dose = 60 / |twin effect at 4 h|, cap 10 U):
  designed-data reference: 4.8 U, 0/10 hypo. Operational twin, reactive loop (d=0): under-estimated effect → 9.3 U (2× the reference),
  3/9 SEVERE hypoglycaemia (< 54 mg/dL). Operational twin, meal-bolused loop (d=0.7): "insulin raises glucose" → NO dose in 5/10 (left
  hyperglycaemic), overdose in the rest → 3/10 severe. IV twins: 1/9 mild hypo, 0 severe. Caveats to report: BG0 reached 175–359 (drip
  controller overshoot in adult#006), drip continues during the test so even the reference does not reach target (BG@4h 215).
- 2026-10-08 C10 capacity sweep (c10_capacity.json; GRU hidden 8–256, 3 seeds, N=2e5): closed loop (kp 2, d 0.8): held-out rollout
  RMSE 0.16–0.17 K for every size; interventional rel. error 1.05–1.07 for every size — flat from 8 to 256 hidden units (T3 plateau in
  capacity). Designed data: interventional error 5.65 → 1.16 → 0.15 → 0.11 → 0.09 → 0.09 (capacity matters when the data are informative).
- 2026-10-08 Commercial-building Chronos-2 (partial): zero-shot 0.81/1.00 (its persistence 1.58/1.15), ft:operational 0.74/0.93, pred-op 0.21,
  step 0.58 (designed GRU/TF twins 0.24–0.38): here the operational-fine-tuned foundation model beats persistence on the TRAJECTORY metric
  while over-estimating the actuator EFFECT 2× — the trajectory metric mixes weather-driven prediction (which operational twins do well)
  with the effect; report the effect metric alongside (as in C9 v2). Caveat for the headline wording.
- 2026-10-08 C9 v2 (T6, heat pump; c7_data/c9_updown.json, 3 seeds, 10 starts): query A (schedule; effect 0.83 K rms): closed-loop twin
  effect-RMSE 0.38 (relative 0.46), chain 0.49 (0.59), trajectory 0.59/0.63 vs persistence 1.04. Query B (actuator off 05–11 h; effect
  0.50 K rms): actuator twin relative effect error 0.64 (pre-stated ≥ 0.5: MET), closed-loop twin 1.00 (cannot see the override: MET),
  chain 0.66 (no rescue: MET); trajectory 0.40–0.59 vs persistence 0.93 (weather-dominated — the effect metric discriminates). Contrast
  upstream 0.46 vs downstream 0.64 is modest; report as such.
- 2026-10-08 commercial planning run killed before completion (CPU freed for referee baselines); its do-RMSE/Chronos results remain in ED only.
- 2026-10-08 C13 classical baselines (heat pump, v0 tests; c13.log): ARX(p) trained on the OPERATIONAL log: p=2 2.12/3.02 (worse than
  persistence 1.70/2.15), p=6 1.23/1.42, p=12 1.11/1.11, p=24 1.03/0.90, p=48 1.04/1.39 K — the direct method with an AR noise model BEATS
  persistence on the (out-of-support) interventions where sim-GRU (3.5/5.7) and TF (2.3/3.2) do not; its 24-h step response 3.6–4.1 K
  OVER-estimates the truth (1.92) ~2× (designed-trained ARX 2.7–2.8, also over); dither-trained ARX worse (2.1–4.9). PEM-SS and grey-box jobs
  crashed (traceback to fix). Thesis reframed: failure belongs to the simulation-error / modern neural recipe, not to operational data.
- 2026-10-08 Commercial-building Chronos-2 complete (c7_data_commercial/c11_foundation.json): designed fine-tune step 0.23/−0.09/0.05 K vs truth
  0.20±0.11 → fails the step capability criterion (passes do-RMSE 0.51–0.55 vs persistence 1.58/1.15); ft:operational 0.74/0.93, step 0.58–0.64
  (3× truth); dither 0.62–0.66, step 0.60–0.76. Inadmissible for effect claims (stats M9); ED only, with the caveat.
- 2026-10-08 v1 P1 data generated (c7_v1/P1): step truth from 10 non-overlapping operational starts 1.84±0.82 K at 24 h; test_in with levels
  {0.1,0.25,0.4} only 72 % inside the operational range (target ≥ 90 %) → levels being re-tuned on the emulator before any model is fitted (protocol
  allows; addendum). Grey-box 2R2C PEM smoke test on operational log: do-RMSE 1.39/1.75, step 0.74 K (0.4× truth): a sign-constrained physical model
  with a free innovations gain ALSO under-estimates (K absorbs the co-movement); ARX(24) 1.03/0.90, step 4.1 K (2.2×).
- 2026-10-08 C13 PEM/grey (v0 tests): PEM-SS(2) operational 1.13/0.98 K (beats persistence), step 4.18 K (2.2× truth 1.92); designed 0.69/0.60,
  step 3.49 (1.8×); dither 2.72/4.05 (!), step 2.58. GREY-2R2C operational 1.39/1.75, step 0.74 (0.4×); dither 1.18/1.81, step 0.28 (0.15×);
  designed 0.83/0.93, step 1.01 (0.5×) → the sign-constrained grey-box fails the capability criterion even on designed data (one-step PEM with a
  free innovations gain K: K absorbs the dynamics). Picture: linear direct methods over-estimate the actuator effect ~2× from operational data
  but predict interventions better than persistence; sim-GRU 0.25×, TF 0.55×, Chronos 0.01×. "Factor two vs factor four–hundred".
- 2026-10-08 v1 P1 interim (linear families; within-support step from operational contexts, truth 1.84±0.82 K at 24 h; test_in persistence 1.10):
  ARX(12/24/48) from the OPERATIONAL log: step ratio 1.54/1.58/1.40 (over by 40–60 %, not 2× as on v0's out-of-range contexts); in-support
  do-RMSE 1.04/1.09/0.75. Dither ±2 K (3 seeds): ARX ratios 0.78–0.91, in-support 0.42–0.80 — dither brings the linear direct method close to
  the truth. Designed: ratios 1.02–1.30. Grey-box 2R2C: ratios 0.11–0.57 everywhere (fails capability). Neural families pending (49/256).
- 2026-10-08 C11b Chronos-2 fine-tuning sweep (heat pump, pre-stated pass: designed fine-tune step ≥ 70 % of 2.4 K and do-RMSE < persistence):
  lr 1e-5/1e-4/1e-3 × 1k/5k steps on the DESIGNED log: step 0.13–0.30 K at 24 h in every recipe (more training or higher lr makes it worse;
  do-RMSE 0.79–1.65) → NO recipe passes. Chronos-2 with the actuator as a known future covariate cannot represent the actuator effect in this
  setting even from designed data (covariate handling / instance normalisation). Consequence: its operational-log result is NOT evidence of
  confounding and must be reported as a capability failure (control referee M7b, ML M3, stats M9). Headline for the foundation model removed.
- 2026-10-08 04:00 paused c7c (control-function IV) and c13 (PEM-SS 3/4) with SIGSTOP to prioritise the v1 fits, MPC and the clean one-step rerun; resume with SIGCONT.
- 2026-10-08 C11b final: lr 1e-3 × 5k steps → step −0.16 K; "no recipe passes the capability criterion" (6/6 fail). Chronos-2 excluded from
  effect claims; reported as a capability failure of covariate-conditioned foundation forecasting for actuator what-ifs. GPU free.
- 2026-10-08 C5d clean one-step rerun (no input noise, 10k steps, N=2e5, 3 seeds): pure feedback d=0 — ARX(20) 0.17 on the same log; GRU-1s
  0.96–0.98; Transformer-1s 1.00/1.00/14.1 (unstable); plans 3.8–23 K. d=0.8 — ARX(20) unstable (inf); GRU-1s 1.02–3.68; TF 1.01–4.83.
  So the neural next-step learners do not recover the plant where a 20th-order ARX does, with or without the noise trick: the gap is the
  neural recipe (optimisation / inductive bias), not the data regime. Supports the reframed thesis (control M3 answered).
- 2026-10-08 c7c and c13 resumed (SIGCONT) after c5d finished.
- 2026-10-08 C13 PEM-SS(3/4) (v0 tests): operational 1.80/2.53 (n=3) and 1.10/0.97 (n=4) vs persistence 1.70/2.15; steps 3.3 / 4.2 K (1.7–2.2×);
  designed 0.71/0.65 and 0.60/0.50, steps 3.2 K; dither 2.0/2.9. All six PEM/ARX direct models over-estimate the actuator effect 1.7–2.2× on
  v0's out-of-range contexts and ≈1.4–1.6× within support (v1 P1), and predict interventions at or below persistence; c13 complete.
- 2026-10-08 v1 P1 interim (sim-GRU, 5 seeds, within support): OPERATIONAL twins: in-support do-RMSE 0.79–0.86 K (BELOW persistence 1.10) yet
  step ratio 0.23–0.34 — a good in-support predictor that has learned a quarter of the actuator effect ("validation lies", now within support);
  dither ±2 K (3 logs × 5 seeds): in 0.33–0.61, ratio 0.70–0.86; designed: in 1.03–1.79 (worse than persistence in support — designed logs
  are mostly out of range), ratio 0.86–1.15. Conformal 6–24 h: operational twins 0.79–0.93 on operational test, 0.39–0.72 on test_in.
  BUG: c15 crashed at family gru_1s (KeyError in c7_neural.build) at 87/256 — fixed (gru_1s → GRUSim); relaunch needed; P2/P3 queues to check.
- 2026-10-08 C6d pancreas v1: linear-twin designed-data capability control — 25/30 patients pass (rel. 4-h error < 0.5; median 0.14). C16 (GPU,
  256-unit sim-GRU, 3 seeds, v1 P1 within support): operational step ratio 0.23–0.26, in-support 0.89–0.92 (< persistence 1.10) — five times the
  capacity changes nothing (T3 in the building, in support).
- 2026-10-08 P2 study crashed at gru_1s (pre-fix import) after 61 jobs; remaining jobs queued (queue_P2b.sh) after the P1 remainder.
- 2026-10-08 C7c control-function IV (heat pump, v0 tests; c7_data/c7c_control_function.json; 3 seeds): vs plain twins on the same dither logs —
  GRU step@24h: A0.5 1.14–1.51 (plain 1.20), A1 1.67–1.73 (plain 1.42), A2 1.75–2.48 (plain 2.09; truth 1.92); do-RMSE ≈ plain (2.1–3.2 /
  3.5–5.0); plans A2 0.68–0.79 (plain 0.67), A1 0.75–0.90 (0.79), A0.5 1.13–1.48 (1.21). TF similar (step 0.8–2.1; plans 0.83–2.54).
  Reading: a proper nonlinear instrumental repair (control function) is neutral to slightly positive on the learned effect and neutral on
  planning; the v0 "IV made it worse" (two-stage, plans 1.8–3.1 K) was an estimator artefact (forbidden regression) — control M4 / ML M7b
  answered; the honest statement is "dither helps; IV-style corrections add little on top".
- 2026-10-08 C12 pessimistic MPC (heat pump v0 target 21→22.5 °C; bootstrap ensembles of 5; receding horizon 24 h, re-plan 2 h; 5 starts):
  embedded controller 0.55 K. Operational-twin ensemble: λ=0 1.44±0.44 K (2.6× embedded even WITH feedback), λ=1 1.41, λ=3 1.06, λ=10 0.74
  (energy 0.56→0.45, out-of-support fraction 0.48→0.31): pessimism helps by shrinking the plan toward the behaviour policy (less heating), not by
  correcting the bias; ensemble spread 0.28 K ≈ designed twins' 0.21 (disagreement does not flag the bias). Dither-twin ensemble 0.35 K at λ=0
  (better than the embedded controller), degraded by pessimism (0.62 at λ=10); designed 0.60–0.70. ML referee M4/M6 answered: feedback and
  pessimism reduce the damage but do not reach the dithered twin; the bias is not disagreement. NOTE target 22.5 °C is above the operational
  99th percentile (v0 target); rerun with an in-range target (21.8 °C) queued.
- 2026-10-08 C16 big sim-GRU (256 units, GPU, v1 P1, 3 seeds): operational step ratio 0.23–0.26 (in-support 0.89–0.92); dither ±2 K 0.75–0.87;
  designed 0.82–1.06 (in-support 1.49–1.64 — designed logs are out of range). Crashed at gru_1s (pre-fix import); relaunched with the
  next-step families at 2 seeds, operational + designed only.
- 2026-10-08 C6d pancreas v1 effects (30 patients × 3 meal seeds; 25 passing the designed-data control; cells with ≥50 % hypo excluded):
  reactive loop (d=0): twin effect never positive; learned/true at 4 h median 0.44; Spearman(true, twin) +0.68 (v0's "uncorrelated" was a
  10-patient artefact — now attenuated but correlated). Meal-bolused loop (d=0.7): positive at 2 h in 100 % of cells, at 4 h in 25 %;
  learned/true 0.11; Spearman +0.75. Additive dither A=0.5 with the two-stage IV: 0.42 / 0.11 (no gain). DOSING v1 protocol FAILED: the
  carbohydrate ramp overshoots (BG0 360, range 240–627) because absorption continues after the drip stops → scenario ill-posed; a v2 protocol
  (steady-state drip tuned over 8 h, held during the test, truth computed at the same operating point) replaces it.
- 2026-10-08 C6e dosing v2 (steady hyperglycaemia held by a tuned drip; truth at the same operating point; 25 passing patients, 3 cohorts):
  reference calculator 2.8 U → BG 134 at 4 h, 0/25 severe. Operational-twin calculators: reactive loop 7.0 U → BG 62, severe 11/22; bolused
  loop 5.9 U (7/23 no dose) → severe 15/23; both Fisher p < 0.001 vs reference. IV twins 4.2 / 5.0 U → severe 3/22, 3/23 (p ≈ 0.10).
  Caveat to report: steady-state BG0 varies 169–586 across patients (drip tuning imperfect for extreme sensitivities); doses are relative.
- 2026-10-08 v1 P1 TRANSFORMER (next-step, NO noise injection, 5 seeds, within support): operational step ratio 0.78–0.95 (!), in-support
  do-RMSE 0.91–1.34 (≈ persistence 1.10; sim-GRU 0.82). The next-step transformer behaves like the direct method: it keeps the actuator effect
  at the price of worse operational rollouts — the opposite trade from simulation-error training. Thesis sharpens: the culprit is
  SIMULATION-ERROR TRAINING (chosen because it gives the best operational rollouts), not neural models per se; next-step GRU is unstable
  (1.57 ± 0.53). Must replicate in P2/P3 and check tf_1s_noise; synthetic TF failed under pure feedback (no reference excitation) — the
  building loop has schedule steps + nonlinearity (informative).
- 2026-10-08 C16 big next-step GRU (256 units, teacher-forced, no noise, 10k steps, 2 seeds): operational step ratio 0.27–0.34 (in-support
  1.27–1.30 > persistence); designed 0.98–1.15. So within next-step training the GRU fails while the transformer keeps the effect
  (0.78–0.95): inductive bias matters, not only the loss. Big transformer (d=128, 4 layers) pending.
- 2026-10-08 C16 big transformer (d=128, 4 layers, next-step, no noise, 2 seeds, P1): operational step ratio 0.73 / 1.00, in-support 0.83 / 0.78
  (better than persistence 1.10) — replicates the small transformer's retention of the effect and improves in-support prediction; designed
  0.64 / 0.77. GPU replication on P2/P3 launched (tf_1s_big and gru_sim_big, operational + designed, 2 seeds).
- 2026-10-08 C16 P2 replication (GPU): big transformer operational step ratio 0.75 / 0.53 (in-support 1.79–1.94 ≈ persistence 1.74); big
  sim-GRU operational 0.19. The P1 pattern (next-step transformer keeps most of the effect; simulation-error GRU loses it) replicates in P2.
- 2026-10-08 C16 P2 complete (GPU): big transformer operational 0.75 / 0.53, designed 1.09 / 0.75; big sim-GRU operational 0.19 / 0.26, designed
  1.08 / 1.00. P3 running.
- 2026-10-08 v1 P1 COMPLETE (256 fits). Operational log, within support, learned fraction [95 % CI] / in-support RMSE (persistence 1.10):
  sim-GRU 0.27 [0.21,0.35] / 0.82; next-step GRU no-noise 1.57±0.53 (unstable, 3.50); next-step GRU + noise 0.17 [0.11,0.24] / 0.84;
  transformer no-noise 0.85 [0.73,1.01] / 1.11; transformer + noise 0.39 [0.30,0.46] / 0.75; ARX(48) 1.40 / 0.75; PEM-SS(2) 1.59 / 1.26;
  grey 0.31 / 1.10. THE TRADE-OFF: every device that improves operational hold-out error (simulation-error training, input-noise injection)
  destroys the actuator effect; the clean next-step transformer and the classical direct methods keep it at the cost of worse operational
  prediction. Dither ±2 K: all neural families 0.54–0.80, ARX 0.89, PEM 1.11. Designed: 0.9–1.3 (grey 0.48 fails).
- 2026-10-08 C12 in-range MPC rerun (target 21.0→21.8 °C, inside the operational range; c7_data/c12_mpc_t21.8.json): embedded 0.48 K;
  operational-twin ensemble λ=0 0.71±0.33 (1.5× embedded), λ=3/10 0.55 (= embedded); dither twin 0.33 (best); designed 0.62. Within support,
  feedback + pessimism nearly rescue the operational twin for a SMALL set-point change; for the larger change (22.5 °C, v0 target) the damage was
  2.6× and pessimism reached only 0.74. Consistent with the identity: the bias bites in proportion to how far the what-if moves the system from
  where the controller held it. Report both targets.
- 2026-10-08 C16 P3 complete (GPU): big transformer operational step ratio 0.48 / 0.23 (P1 0.73–1.00, P2 0.53–0.75) — the transformer's
  retention of the effect is PERIOD-DEPENDENT and lowest in the least-excited period (P3: narrowest operational range 20.2–22.1 °C); big sim-GRU
  0.29–0.37 (as always). Designed: TF 0.59–0.71, GRU 1.03–1.12. Reading: next-step learners recover the plant only to the extent the loop is
  informative (classical informativity), simulation-error learners do not recover it even when it is. Wording in the draft adjusted.
- 2026-10-08 17:30 v1 fits COMPLETE: P1 256, P2 192, P3 192 (P2/P3 have 6 training logs each; the earlier /256 denominators for P2/P3 were wrong).
- 2026-10-08 18:10 DRAFT v1 compiled (paper/main_v1.pdf): reframed thesis, v1 Results from c7_v1/tables.txt, Methods and ED updated; red-team round 2 launched.
- 2026-10-08 19:30 BUG (found by stats referee round 2): v1 step-query starts were 18/30-h spaced (interleaving bug) and 1/2/6 of 10 starts had
  ZERO effect because the BOPTEST override-activation inputs are BOOLEAN and bop.py set them with setReal, which is silently unreliable
  (worked in the continuous-override logs — designed/test_in follow the levels 97–100 %, dither r column shows the offsets — but failed at
  some step-query starts). FIX: setBoolean in bop.Emulator.step; truth regenerated with 7 starts exactly 48 h apart (v1.1): 24-h truth
  2.55±0.33 / 2.97±0.37 / 2.70±0.40 K (all starts responsive; v1 had 1.84/2.09/0.98 dragged down by nulls). All saved models are being
  re-evaluated (c18) against the corrected truth; MPC (overrides each step) to be rerun with the fix; v0 what-ifs (c9) flagged as suspect.
- 2026-10-08 C19 inverse-controller test (control MAJOR 9): embedded PI fitted from the operational log (R² 0.99); −1/Ĉ 24-h response to a +0.3
  step: −0.19 / −0.85 / −0.59 K (P1/P2/P3) vs truth 2.55 / 2.97 / 2.70; linear-law mixture with γ² = coherence(u, schedule) 0.64/0.53/0.50 →
  +1.58 / +1.17 / +1.07 K (fractions 0.62/0.39/0.40). Learned fractions to be compared after c18 (sim-GRU ≈ 0.2, TF ≈ 0.4–0.6 on the new truth).
- 2026-10-08 C20 pancreas ARX comparator (control MAJOR 8): reactive loop — ARX(6/12/24) over-estimates the 4-h insulin effect 4–13× (free-run
  instability); meal-bolused loop — sign wrong in 70–90 % of patients. The classical direct method FAILS on the pancreas logs (nonlinear patient,
  F≠0 with boluses): "the classical remedy is not universal" — state explicitly; thesis restricted to feedback-dominated loops.
- 2026-10-08 20:10 text: recipe-lineage paragraph + 8 refs; conditional identity displayed as eq. (1); v1 pancreas Methods; related-work comparator sentence softened (both biased).
- 2026-10-08 C21 grey-box 2R2C fitted by sign-constrained SIMULATION ERROR (no innovations gain; corrected v1.1 truth): operational step
  ratio 1.10 / 0.69 / 0.25 (P1/P2/P3); designed 1.61–1.78 / 1.14–1.26 / 1.50–1.60 (over-estimates on designed data → structural
  mis-specification of a 2R2C for this building; passes the ±50 % capability rule in P2 only); dither 1.06–1.34 / 0.82–0.90 / 0.69–0.94.
  Report: a physically structured OE model is period-dependent and itself mis-specified; neither grey-box variant is a clean comparator.
- 2026-10-08 23:00 v1.1 RE-SCORED (corrected truth, 7 starts 48 h apart; c7_v1/tables.txt): learned fraction, operational log — sim-GRU
  0.22 [0.16,0.30] / 0.14 [0.08,0.20] / 0.28 [0.20,0.41]; 256-unit 0.19/0.21/0.29; next-step GRU+noise 0.13/0.06/0.19; TF+noise 0.36/0.15/0.22;
  clean TF 0.78 [0.65,0.94] / 0.63 [0.44,0.87] / 0.48 [0.40,0.61] (big TF 0.72/0.63/0.48; HP-selected recipe 0.71 in P1); clean next-step
  GRU 2.2–2.8 (HP-selected recipe 1.22; 256-unit 0.25 — recipe-dependent); ARX(48) 1.40/1.11/0.88, ARX(12) 1.55/1.14/0.90; PEM(2)
  1.64/1.65/1.21 with designed 1.38–1.47 (operational/designed ≈ 1.19/1.12/0.83); grey (K) fails. Dither ±2 K: sim-GRU 0.55–0.87, TF 0.45–0.81,
  ARX 0.76–0.87, PEM 0.94–1.12. Pred (op-week) error / in-range-masked coverage: sim-GRU 0.49/0.48, 0.41/0.81, 0.13/0.40; TF 0.66/0.80,
  0.55/0.98, 0.29/0.56; ARX 0.69/0.90, 0.92/0.98, 0.31/0.64. Inverse-controller test: learned sim-GRU steps +0.56/+0.43/+0.76 K lie between
  −1/Ĉ (−0.19/−0.85/−0.59) and the truth (2.55/2.97/2.70), below the linear-law mixture with schedule coherence (+1.58/+1.17/+1.07) — the
  schedule is weaker excitation than its coherence suggests (synchronous with occupancy); clean TF near the law (1.98/1.88/1.31).
  HP sweep (c17): tf_1s best recipe (h48, lr 1e-3, 10k) → 0.71; gru_1s (h96, lr 1e-3, 10k) → 1.22. MPC rerun (fixed override) running.
- 2026-10-08 23:30 DRAFT v2 text: results_v1.tex rewritten to corrected truth and the 'both biased, differently' thesis; new title; abstract ≤ 230 words; Fig 1 x = operational error; Fig 2 = within-architecture ablation; ED Table 1 rescored; MPC numbers pending the fixed-override rerun (marked).
- 2026-10-09 00:55 C12 in-range MPC rerun WITH the Boolean-override fix (target 21.0→21.8): embedded 0.47; operational-twin ensemble λ=0
  0.66±0.28, λ=3/10 0.54/0.55; dither twin 0.30 (λ=0) → 0.49 (λ=10); designed 0.59–0.67. Same as the pre-fix run within noise (continuous
  overrides were unaffected by the bug). Out-of-range target rerun in progress.
- 2026-10-09 05:00 C12 out-of-range MPC rerun WITH the fix (21.0→22.5): embedded 0.53; operational λ=0 1.41±0.46, λ=3 1.04, λ=10 0.72
  (energy 0.57→0.45, out-of-range actions 0.48→0.31); dither 0.32 → 0.57 (λ=10); designed 0.58–0.70. Same as pre-fix within noise.
  ALL COMPUTE FOR v2 COMPLETE. Paired per-start λ=0 − λ=10 differences in c7_data/c12_summary.json.
- 2026-10-09 05:10 DRAFT v2 COMPLETE (paper/main_v1.tex → main_v2_draft.pdf): no placeholders; all numbers from the v1.1 audit; red-team round 3 launched.
- 2026-10-09 03:20–04:10 Red-team round 3 (reviews/v2_stats_referee.md MINOR; v2_ml_referee.md MINOR; v2_control_referee.md MAJOR). Applied:
  stale truth text in Methods/ED fixed; V5 → inconclusive (p=0.13, CI −0.31..1.68); MPC twins disclosed as v0-P1; Fig 4 s.d./0.55; designed
  recovery 0.66–1.28; abstract rewritten ≤200 words (191) with the 6-h number and the corrected dither claim (noise families 0.2–0.7 dropped
  from it); title → "Digital twins that forecast operation best learn the controller's co-movement, not the system's response"; "opposite
  direction" → "within 10–65% of the effect, direction period-dependent" everywhere; Fig 2 redrawn (256-unit + selected-recipe clean GRUs, op
  error under each bar); Discussion attribution → "whatever buys operational accuracy buys the co-movement"; ED Tables 3–8 generated by
  gen_ed_tables.py (two-clause capability; HP sweep; conformal bands op/sched/in-range-masked; dither amplitude + cost per period: ±2 K costs
  0.89–1.01 K rms and +15..+28% energy, ±0.5 K 0.31 K/+4%, ±1 K 0.56 K/+12%; dither-log law test; horizons + the +0.1 step); Table 1
  regenerated by gen_table1.py (6-h column, ARX 2/12/24/48, "0.72 (one log)"); synthetic planning counts 29/36 & 22/24; conformal band stated
  (6–24 h, in-range-masked); "a fifth of the way" → 0.27/0.34/0.41; −1/Ĉ outcome stated (no twin near the inverse; shrunken plant response
  decaying with horizon); law row on the schedule demoted to an upper bound (band registered, ED item 19); upstream/downstream what-ifs (c9,
  faulty override) withdrawn (ED Fig 9); ED Fig 2 planning labelled pre-fix; grey-OE informativity sentence (C_m degenerate); DART moved to the
  dither lineage; citations added (BOPTEST, simglucose, Dalla Man, Chronos-2, Pearl, Angrist, Strachan).
  NEW RESULTS: (a) per-horizon fractions from saved c18 trajectories — sim-GRU 0.69/0.59/0.40/0.22/0.14 (P1), 0.36/0.32/0.33/0.14/0.10 (P2),
  0.72/0.75/0.59/0.28/0.19 (P3) at 3/6/12/24/48 h; ARX(48) 1.1–1.4 flat; clean TF non-monotone 0.5–1.1. Step in-range fractions (first 24 h):
  base 0.71/0.79/0.80, +0.3 arm 0.33/0.27/0.62. (b) Dither-log law test: γ²(u, logged offset) 0.49–0.75 vs sim-GRU frac24 0.51–0.87; |diff|
  ≤0.07 in 8/11 logs, +0.12..+0.20 in P3 — the law holds where its excitation is exogenous. (c) c22 +0.1 step (0.4→0.5) from the same 7
  starts: truth 0.89/1.05/0.95 K at 24 h; intervention arm in range 0.58/0.57/…; scoring of all saved models in progress. (d) c20b pancreas
  ARX comparator rerun on regenerated v1 logs (stable seed; c6d had used Python's per-process str hash and did not save logs — ED item 21).
- 2026-10-09 04:40 c22 +0.1 step scored (all periods): sim-GRU 0.26/0.15/0.36 at 24 h, 0.60/0.37/0.88 at 6 h (+0.1 arm in range 0.58/0.57/0.81)
  — same shortfall as +0.3, so not extrapolation. c20b pancreas ARX on regenerated v1 logs (n=68 reactive / 67 bolused, 25 passing patients × 3
  seeds): reactive ARX6/12/24 median ratio 6.8/2.3/2.5 (sign right); bolused −0.01/−0.02/−0.04, positive 67/55/63 %; linear twin on the same
  logs 0.42 / 0.10 (2-h positive in 100 % of bolused cells) — reproduces c6d's 0.44/0.11. Draft v2.1 compiled: paper/main_v1.pdf.
- 2026-10-09 04:30 Round-4 red team (reviews/v2_1_*): ML MINOR, stats MINOR, control MAJOR (attribution to co-movement not separated from a
  generic low-frequency deficit). Applied: dither-log law → "comparison" with inverse term (6/11 within 0.1; worse than constant: rms 0.160 vs
  0.098); capability clauses applied as pre-stated (no "borderline pass"; clause 2 unmasked vs unmasked, masked pair noted); ED Table 4
  selection column, "15 of 16 above"; "within X–Y%" → ratio ranges (ARX 0.88–1.65, SS to 1.65; dithered 0.76–1.12); fast-response claim
  "two of three periods" (P2 0.32 at 6 h); 6-h in-range fractions 95–100% all arms; ED Table 9 op./des. by horizon (sim-GRU 0.65–0.76 → 0.14–0.22
  in P1/P3, 0.33→0.14 P2; ARX(48) 1.0–1.4 P1/P2, decays 1.11→0.55 in P3); Fig 2 caption (operational persistence 0.66/0.65/0.51; in-range
  3.5/2.0/5.8); 6-h and +0.1 CIs via hboot.py (c22 per_start): 6-h 0.59 [0.44,0.79]/0.32 [0.24,0.39]/0.75 [0.47,1.12]; +0.1 24 h 0.26
  [0.17,0.36]/0.15 [0.09,0.19]/0.36 [0.22,0.57]. Pancreas: per-patient medians (23 patients), NaN exclusions disclosed (10/5/7 of 68), c20c ARX
  on designed logs: ARX(24) 0.90, ARX(12) 0.56, ARX(6) 0.20, lin 1.00 (no divergence) → reactive-loop over-estimate is the mis-specified direct
  model under feedback, not model class alone; c6f dosing on regenerated saved logs: ref 2.8 U 0/25; op 6.9 U 12/23, 6.2 U 15/22 (no-dose 6/22);
  IV 3/23, 3/23 — reproduces c6e (11/22, 15/23, 7/23, 3/22, 3/23); now the headline (ED items 22–24). c23 shuffled-day replay launched (the
  decisive co-movement-vs-spectrum test): replay logs y range 17–27.5 vs 20–23.4 operational, u mean identical, corr(u_replay,u_op) 0.47.
- 2026-10-09 04:45 C23 SHUFFLED-DAY REPLAY (decisive co-movement test): twins trained on the operational u with days permuted, replayed
  open-loop (same weather/occupancy/marginal/daily profile; corr 0.46–0.47 with the operational u; y range 17–28 vs 20–23 °C): sim-GRU
  0.73–1.21 (vs 0.22/0.14/0.28 operational), noise-GRU 0.54–1.07, noise-TF 0.65–1.04, clean TF 0.96–1.60, ARX 1.00–1.29 (vs 0.88–1.55),
  PEM 1.54–1.81. → the operational log withholds alignment-independence, not spectral excitation; written into abstract/intro/results/
  discussion/methods; ED Table 10. Abstract 200 words. Draft v2.2.
- 2026-10-09 05:50 Round 5 (reviews/v2_2_*): control MINOR; stats+ML focused revision — replay also removes corr(u,T_out) and widens output
  range. C23 STRATIFIED replay (days permuted within quartiles of daily mean T_out; corr(u,T_out) −0.34..−0.62 vs −0.52..−0.70 op; y 17–25 vs
  20–24 °C): sim-GRU 0.74–0.99 at 24 h (0.81–1.16 at 6 h; op err 0.35–0.86 K), clean TF 0.84–1.27, noise-TF 0.42–0.80, noise-GRU 0.13–0.68
  (P2 partial), ARX 1.08–1.38, PEM 1.47–1.66 → co-movement with the UNLOGGED disturbance confirmed with weather collinearity and range kept.
  ED Tables 10/11 regenerated with corr/R²/range/power columns and per-log seeds×starts intervals; `\'` typography fixed; hboot per-call
  seeds; ED Table 4 footer; ED Fig 7 provenance; ED Fig 6 = −1/Ĉ trajectories (instrumental → ED Fig 11). Draft v2.3.
- 2026-10-09 06:30 Submission package split (build_submission.py): overleaf/main.tex (main text + Methods + Extended Data Tables 1–2 + ED
  Fig 1 = −1/Ĉ trajectories; 20 pp) and overleaf/supplement.tex (Supplementary Tables 1–9 = former ED Tables 3–11; Supplementary Notes 1–11;
  8 pp); cross-references remapped automatically; wide tables scaled; replay tables restructured per period. PDFs in pdf/; zip
  paperC_overleaf.zip. Working draft paper/main_v1.tex remains the single-document source.
