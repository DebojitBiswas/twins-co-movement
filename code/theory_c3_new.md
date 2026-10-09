# C3 theory — what is new (v0, 2026-10-07). Each claim has a test; none enters the paper without it.

Lineage stated first: the linear −1/C limit, informativity of closed loops and two-stage/IV identification are classical
(Gustavsson–Ljung–Söderström 1977; Ljung 1999 ch. 13; Van den Hof & Schrama 1993; Forssell & Ljung 1999; least-costly experiment design
Bombois et al. 2006). The claims below are what the classical theory does not say, phrased for learned simulators.

## T1. Decomposition identity (nonlinear; the linear mixing law is its special case)
Setting: y = 𝒢[u] + v (𝒢 a causal, possibly nonlinear, plant operator on input paths; v the disturbance-driven part of the output, with
E v = 0 under the operational law); the controller makes u a function of the disturbance path and of logged exogenous signals.
A simulator trained by simulation error, with a rich enough class, converges to the conditional mean
    𝒢̂[u] = E[y | u] = 𝒢[u] + E[v | u].
**The learned simulator is the true response plus the disturbance the controller reveals through the input.** Under do(u = u*) the true
output is 𝒢[u*] + v with E v = 0, so the interventional error of the twin is exactly E[v | u = u*]: *the disturbance the controller would
have been reacting to had it produced u**. Proof: definition of the L2 projection; E[v | do(u)] = 0 by exogeneity of v. ∎
Corollaries: linear case gives Ĝ = γ²G + (1−γ²)G_op (theory_c2); the twin is exact iff u ⟂ v in the band of interest (= informativity).
Test: existing collapse (c2_collapse, 28 cells); nonlinear check on the pancreas and BOPTEST via T5.

## T2. Uncertainty blindness (new statement, trivial proof, important for ML)
(a) Ensembles / Bayesian posteriors / bootstrap of any consistent estimator of E[y | u] have spread → 0 as N → ∞, while the interventional
error → |E[v | u*]| > 0: the error is a bias of the population optimum, invisible to epistemic uncertainty by construction.
(b) Conformal / calibrated predictive intervals computed on held-out OPERATIONAL data are valid for operational queries and can have
arbitrarily low coverage under do(u*): the shift is in the law of the input process (u no longer a function of v), which no
exchangeability-based guarantee covers; it is not covariate shift (u* may lie inside the training support of u).
Test (cheap, existing data): BOPTEST — calibrate split-conformal intervals of the GRU/TF twins on held-out operational days; measure
coverage on the designed test schedules. Prediction: operational coverage ≈ nominal, interventional coverage ≪ nominal; designed-data
twins keep coverage. Synthetic: ensemble spread vs error already 7–80×.

## T3. Scaling plateau
Predictive error (operational hold-out) → 0 with N and capacity; interventional error → the T1 floor, independent of N and capacity.
"Scaling laws do not transfer to interventions." Test: existing N-sweep (c1: 2e3–2e5, flat); ADD capacity sweep (hidden 8…256,
synthetic + BOPTEST) with both errors on the same axes.

## T4. Planner overshoot law (decision-level, quantitative)
A planner that optimises the input on the twin to reach a target change Δ applies, at steady state, u = Δ/ĝ where ĝ is the twin's DC
gain; the true plant then moves by Δ·g/ĝ. With pure proportional feedback ĝ = γ²g − (1−γ²)/K_p, so the overshoot factor
g/ĝ → ∞ as γ² → (1/(1+gK_p)) and changes sign below it. Test: BOPTEST planning (c7b): predicted ratio of planned mean input to the
designed-twin's planned input = ĝ_designed/ĝ_operational (from the learned step responses) vs observed ratio (0.63/0.27 ≈ 2.3 for the
GRU; learned/G = 0.19 predicts ≈ 5 before saturation at u ≤ 1 — check with the actual non-saturated fraction).

## T5. Certificate theorem (what the naive certificate got wrong)
The attenuation of the learned effect in a frequency band equals the PARTIAL coherence of u with the logged exogenous excitation,
given the twin's other inputs x: γ²_{u r | x}(ω). With no logged excitation γ² = 0 identically and the twin's response must be compared
with G_op = Ĝ_op estimated from the log (−1/Ĉ under pure feedback). The share of input variance explained by logged drivers (c4c) is not
the certificate because the drivers are twin inputs (they are conditioned on, not instruments).
Test: BOPTEST dither sweep — γ² of u with the dither component (12 h–2 d band) 0 / 0.43 / 0.62 / 0.75 vs learned/G (GRU) 0.19 / 0.46 /
0.54 / 0.80 — already supportive; redo with the partial coherence given weather, both buildings, 3 seeds, CI. Pancreas: γ² of insulin
with the additive basal dither vs the IV/plain learned effect ratio.

## T6. Upstream vs downstream what-ifs (the constructive rule; new for twins)
Let r be a logged exogenous signal upstream of the controller (set-point, schedule, price) and u the actuator signal downstream.
(i) The closed-loop map r ↦ y is identifiable from operational data by simulation error on (r, x) → y (r ⟂ v), so a twin trained on
(r, x) answers set-point what-ifs correctly within the support of r.
(ii) The plant map u ↦ y is identifiable only to the extent of γ²_{u r | x}; actuator what-ifs (new controller, demand response that
overrides the actuator, a fault) are not answerable from operational data without excitation.
Rule: *a twin trained on operational data can answer what-ifs about what the controller was told, not about what the actuator did*.
Test (NEW experiment, BOPTEST, both buildings): train closed-loop twins (r, x) → y and actuator twins (u, x) → y on the same operational
log. Query A (upstream): a changed set-point schedule (+1 K occupied hours; a 2-h earlier morning start); truth = emulator with the
embedded controller under the new schedule. Query B (downstream): a demand-response override (heat pump forced to 0.2 for 4 h at 17:00)
with the embedded controller otherwise; truth = emulator. Prediction: closed-loop twin do-RMSE on A < persistence and ≈ designed-level;
actuator twin on B worse than persistence; chaining actuator twin with a learned controller twin (u | r, y) does NOT rescue B (the
override breaks the controller map), but does answer A (consistency check).
This also explains the Twin Houses: the question asked there (heater ROLBS) was downstream; the regulated data could not answer it.

## T7. Excitation budget (practitioner's trade-off; connects to least-costly design)
For set-point dither of amplitude A: regulation cost = Var of the dither-induced output deviation = ∫|T|²Φ_r (T complementary
sensitivity; ≈ A²/3 for slow steps), energy cost ∝ ∫|SC|²Φ_r; benefit = γ²(A) per band. Closed form in the linear case; empirical curve in
BOPTEST (A = 0.5 / 1 / 2 K): comfort cost (RMS deviation from schedule in the dithered logs) vs plan quality (c7b: 1.22 / 0.83 / 0.69 K)
and vs learned/G. Rule of thumb to derive: dither amplitude ≈ the comfort tolerance already accepted by the set-back schedule buys
most of the attainable accuracy. Test: existing logs (cost) + c7b (benefit); add A = 0.25 and 4 K to bracket.

## What this gives the paper
A single identity (T1) from which the ML-relevant consequences follow (T2 uncertainty blindness, T3 scaling plateau, T4 overshoot),
a certificate that is computable from the log (T5), a constructive rule for which what-ifs are answerable (T6), and a cost–benefit
curve for the fix (T7) — each tested on a nonlinear emulator the community uses, plus the pancreas. Foundation-model test (Chronos
fine-tuned on the operational logs) as the empirical headline for T3 if the GPU can be used.
