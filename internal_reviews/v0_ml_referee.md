# Red-team v0 — ML / world-models referee (2026-10-08). Verdict: MAJOR REVISION. Most damaging: M4.
MAJOR
M1 ML lineage absent; prior work makes the central points: Ortega et al. 2021 (delusions in sequence models), de Haan et al. 2019 (causal confusion in IL),
   Swamy et al. 2022 (causal IL under temporally correlated noise; IV with lagged signals), confounded OPE (Tennenholtz 2020, Namkoong 2020, Kallus & Zhou 2020,
   Bruns-Smith 2021, Shi et al. 2022, Zhang & Bareinboim 2016, Lu et al. 2018, Bennett & Kallus 2021), TS treatment effects (Bica 2020 TSD/CRN, Lim 2018,
   Melnychuk 2022, Hatt 2021), anchor regression (Rothenhäusler 2021), ICP, DeepIV/DFIV, MOPO/MOReL/COMBO/RAMBO. Narrow the novelty claim.
M2 Theory: twins are warmed up on observed y → identity must be conditional (E[v_{>t}|y_{≤t},u_{>t}] − E[v_{>t}|y_{≤t}]); one-step families have no
   theorem beyond Prop 2(ii); ARX(20) recovers the plant under pure feedback (0.08) → contradicts "flat in model class"; move to main text; separate claims.
M3 Recipes weak / capability control inconsistent: no HP search, lr fixed, single data seed per regime; designed twins only modestly beat persistence;
   Chronos designed fine-tune learns 22 % of the step → fails the capability rule → headline uninterpretable. Fix: HP sweep on designed validation,
   larger models on buildings, Chronos fine-tune sweep until ≥70 % of step, ≥5 seeds, ≥2 data seeds, pre-stated pass threshold.
M4 Uncertainty blindness argued against seed-only ensembles; no bootstrap/MC-dropout/Bayesian; no MOPO-style pessimistic planner; plans saturate
   (possibly out of support, where disagreement would veto). Fix: bootstrap ensembles + MC-dropout; MOPO-style planner (λ sweep); planned-input vs
   support plot; spread along the plan trajectory. New Fig 3 panel.
M5 Conformal: designed twins also miscover (0.25–0.49 interventions; 0.67–0.77 operational) — not in main text (T2 pre-stated "keep coverage" NOT met);
   0–4 % vs 0–5 % inconsistency; separate weather shift from controller shift; discuss weighted conformal (singular density ratio) and run ACI/EnbPI.
M6 Planning is open-loop → inflates headline; designed twins' plans (1.0) already 2× worse than embedded (0.55). Fix: receding-horizon MPC for all twins;
   fair ratio is operational vs designed twin (2.5–3×).
M7 (a) Chronos not repaired by dither (≈ persistence) — say so; (b) IV negative is a straw man: use control-function (first-stage residual as input,
   zeroed at prediction), DeepIV/DFIV or anchor regression; (c) certificate overclaimed: ordinal, band post hoc, requires declared exogenous
   excitation (not "from the log alone"); P4 (sign-constrained grey-box) silently dropped.
M8 Title overclaims: "controller" is the F=0 special case; commercial GRU over-estimates (schedule/occupancy co-movement); ARX(20) recovers plant;
   "every family" strained (sim-GRU excluded on pancreas; Chronos 22 %). Suggest "learn the co-movement the controller induces, not the response".
M9 Pancreas headline is the linear twin; dosing caveats (BG0 175–359; reference arm 215 at 4 h) absent; hand-tuned PID not a published AP algorithm.
M10 Seeds 3 not 5; no block bootstrap; "29/36 cases" are seed×regime cells; post-hoc changes only in ED.
MINOR m1 number consistency (0–4/0–5; 29 vs 30/36); m2 "thirty-fold larger model" wording; m3 Fig1 legends; m4 Fig3c ordinal; m5 Fig5d designed-Chronos
worst operational error — explain; m6 define do-RMSE/relative error; m7 causal projection remark to Methods; m8 upstream query outside support of r;
m9 planner convergence; m10 uncited bib entries; m11 TODOs; m12 motivate anticipating controller; m13 word budget.
MOST VALUABLE ADDITION: "Conservatism and feedback do not rescue the twin" — MPC through bootstrap ensembles with MOPO-style penalty (λ sweep),
operational vs dithered vs designed; planned-input-vs-support plot. Second: a real dataset with a logged intervention. Third: cross-domain collapse panel.
