# Red-team round 2 — ML referee (2026-10-08). Verdict: MAJOR REVISION (second round). Most damaging: MAJOR 1.
Resolved: M1 lineage, M4 pessimism (substance), M6 open-loop, M7b/c. Partial: M2 theory (Methods still unconditional; stale "eq (1)", "Fig 1c"),
M3 recipes (no HP sweep; next-step GRU flips 1.6–2.3 → 0.27–0.34 with recipe), M5 conformal (ACI not run; designed range mis-stated), M8 title,
M9 pancreas (abstract "insulin raises glucose" is 2 h bolused only; Methods stale), M10 ("7/12" untraceable).
MAJOR 1 Fig 1 plots in-range designed error, on which ARX(48) beats the sim-GRU on BOTH axes in all periods → put operational hold-out error ("pred")
  on x; lead with the within-architecture noise ablation (clean vs noise, same transformer/GRU).
MAJOR 2 "informativity"/"architecture" over-interpreted (3 periods, no excitation measure; synthetic clean TF fails on an informative loop; big TF learns
  less from designed than operational in P1; clean GRU flips with recipe) → quantify excitation per period or demote; delete "not by the model".
MAJOR 3 "the modern recipe" mis-attributed: MBRL (PETS/MBPO/MOPO/MOReL/Dreamer) = one-step teacher-forced ensembles; simulation-error = neural
  system-ID / neural-ODE / physics-ML; noise injection = GNS/MeshGraphNets. Cite each; mechanism hypothesis: noise on inputs = errors-in-variables
  attenuation of the u-pathway compensated by collinear y-history in operational data.
MAJOR 4 Chronos: numbers 0.44/0.30/0.22/0.13/0.18/−0.16 (lr 1e-3 diverged); one seed per recipe; pass rule inconsistent with ±50 % → one rule; narrow
  claim to "Chronos-2 under full fine-tuning with actuator covariate, in our budget"; sell the 0.73-K designed forecast with 0.44-K step as the finding.
MAJOR 5 Pessimism: n=5, s.d. 0.44; in-range 0.71±0.33 vs 0.55±0.36 not significant; "retreat" not separable from "stopped overheating" without applied-u
  traces; "only" loaded; designed-twin MPC (0.60–0.70) no better than pessimistic operational; rescore V5.
MAJOR 6 Eighteen number fixes (see full report in transcript): 256-GRU 0.19–0.37; ARX 0.9–1.6×; designed noise families 0.54–1.30; Chronos range;
  designed coverage 0.83–0.99; op error 0.11–0.49; synthetic 1.03–1.27; 7/12 vs 29/36; energy 4–20 %; Methods pancreas stale; eq(1)/Fig 1c stale;
  "within 10–20 %" (ARX P3 0.77); ED Fig 4 P1-only; c7c under v0 protocol; "insulin raises glucose" 2 h only; P3 is Oct–Dec; PEM over-estimates on
  designed too (attribute only the excess); uncited bib entries.
MAJOR 7 No HP search for next-step families → run small sweep (hidden/lr/steps selected on designed validation) or demote/drop clean next-step GRU.
MAJOR 8 Title over-generalises (ARX/PEM/TF are learned simulators that do not) and "forgotten" unsupported → "Simulation-error training teaches neural
  simulators the controller's co-movement...; the classical prediction-error method does not"; "replaced rather than extended".
MINOR m1–m15 (conditional error formula; stale refs; Methods pancreas; cite BOPTEST/simglucose; PEM designed excess; c7c protocol; define metrics;
  abstract length ≈330 words → 150–200; MPC uncertainty; "0.6–0.9" is sim-GRU only; big-TF designed near floor; "heating-season periods").
