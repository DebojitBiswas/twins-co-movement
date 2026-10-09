# Red-team v0 — control / identification referee (2026-10-08). Verdict: MAJOR REVISION (reject if direct-PEM/grey-box recovers the plant from operational logs). Most damaging: M2.
MAJOR
M1 Identity and eq.(1) are classical (Pearl Ch.3 / OVB; Ljung §13.4; GLS 1977 §3; Söderström–Stoica Ch.10; Forssell–Ljung 1999 §3.1 OE bias =
   (1−γ²)×co-movement). Only new ingredient F≠0, undeveloped. "We derive" must go; contribution = empirical demonstration on modern families + decisions.
M2 Headline contradicted by own ED Fig 1 (ARX(20) 0.08 under pure feedback) and by classical theory: error is a property of the MODEL CLASS (OE /
   simulation-error, no noise model) and recipe, not of operational data. MISSING: direct PEM with noise model (ARMAX/BJ/innovations SS) and grey-box
   RC PEM on every system (P4 never run). Informativity of the synthetic loop comes only from clipping/anti-windup/quantisation — never discussed.
M3 Next-step evidence confounded: input-noise injection = errors-in-variables pushing one-step learners toward OE; quoted 1.04/1.18 are d=0.8 cells;
   d=0 cells 0.98 / 0.88–4.10 (unstable); ARX(20) yardstick not shown. Report d=0 and d=0.8 separately; train without noise; longer/larger.
M4 IV negative = forbidden regression (fitted values into nonlinear second stage; Angrist–Pischke §4.6.1) + train/test input mismatch. Need IV loss /
   DeepIV / control function, and the linear indirect / coprime-factor methods on the dithered logs.
M5 Certificate: not log-only (needs instrument validity), computed as plain coherence in a post-hoc band (not partial given weather), no proof of the
   partial statement, not a bound (TF learned 0.49 at γ²=0), applies only to OE learners, 4 points no CI.
M6 "Upstream answerable" unsupported (0.46 vs 0.64); schedule synchronous with hidden gains in both emulators (premise fails); CL=1.00 downstream is
   trivial; no designed-data reference on A/B; proposition = first stage of the indirect method (Ljung §13.4; VdH–Schrama 1993).
M7 Numbers: (a) learned fractions normalised by designed-GRU step (2.56–2.67) not the truth (1.92±0.74) → 0.26/0.62/0.74/1.09; designed GRU over-
   estimates by 33–39 %; (b) Chronos designed = 23 % of truth → capability FAILED; (c) pancreas 10/10 is at 2 h, 5/10 at 4 h; linear static filter twin;
   dosing artefacts; (d) "4–5× worse than own controller" → fair: 2.4–3× worse than designed twins; (e) designed twins cover 15–60 %; (f) overshoot
   formula is for proportional feedback; all controllers have integral action; 48-h plans not steady state.
M8 Identity gaps: twins condition on y-history and x → learned E[y|u,y_hist,x], error ≠ E[v|u*] when v persistent; E[v|u*] undefined out of support —
   designed tests ARE out of support (zone outside 20.5–21.2 °C) → part of failure is extrapolation; within-support step from the operational state is
   the clean test (switched away after seeing results); eq.(1) is non-causal Wiener limit; causal OE → Φ_u-weighted projection (Ljung Thm 8.3).
M9 Pre-specification claims not met as written (post-hoc items 3,5,6,7; 2 seeds pancreas neural; QUALITY ≥5 seeds/CIs; MOPO absent; TODOs; 29 vs 30/36).
M10 Missing baselines (summary): direct PEM w/ noise model; grey-box RC sign-constrained (Bacher & Madsen 2011 / CTSM); indirect & joint I/O on dithered
   logs; proper nonlinear IV; MOPO planner; within-support metric; Chronos with designed vs operational contexts (Fig 5c compares non-comparable quantities).
MINOR 1 title vs law (integral action → zero at DC); 2 cite Bombois/Hjalmarsson, Pearl, Vovk, MOPO, Dreamer, de Haan 2019, Rezende 2020, Namkoong 2020,
Privara 2013; 3 "thirty-fold" (32× units, 670× params); 4 truth 1.92±0.74; 5 ensembles are init-only; designed spreads > error; 6 regime counts;
7 synthetic noise negligible, informativity from nonlinearity; 8 conformal windows overlap; 9 "consistent with" not "predicts"; 10 include Lemma 1;
11 no real-data result — say so; 12 Chronos step with covariate may be covariate handling — designed-context control needed; 13 toy PID not a pump;
14 number inconsistencies.
