# Round 3 — control referee (2026-10-09). Verdict: MAJOR REVISION (text, consistency, two cheap re-analyses; no retraining).
Most damaging: MAJOR 2 — headline 0.14–0.29 is the 24-h value of a horizon-decaying curve; intervention arm is outside the training range
67/73/38% of the first 24 h (c18 frac_inside_interv 0.33/0.27/0.62). Sim-GRU fraction 0.69/0.36/0.72 at 3 h, 0.59/0.32/0.75 at 6 h,
0.40/0.33/0.59 at 12 h, 0.22/0.14/0.28 at 24 h; ARX(48) 1.1–1.3 at every horizon. Report horizons + in-range fractions; rename query
("step from operational start states"); abstract to carry the 6-h number; optional +0.1 step (0.4→0.5) from the same starts.
MAJOR 1: "opposite direction" holds in P1 only (ARX 1.55/1.40, 1.14/1.11, 0.90/0.88); reword "within 10–60%"; add op/designed column;
spectrum-vs-feedback ambiguity (Ljung Thm 8.3) → limitation or shifted-replay control.
MAJOR 3: −1/Ĉ reported only at 24 h where integral action makes it ≈0; full-trajectory ED figure; "a fifth" → 0.27/0.34/0.41; state outcome
(no twin near −1/Ĉ; shrunken plant response decaying with horizon); law row as upper bound; test law on dither logs (γ² with logged offset);
register band (ED item 19).
MAJOR 4: pass/fail table (now ED Table 3); borderline rule stated symmetrically (tf_1s_noise P2 0.50; grey-OE passes in P2); "P2/P3 carry
the headline" under strict rule.
MAJOR 5: c20 ARX comparator was on v0 ten-adult logs, not v1; rerun on v1 logs; reactive loop (F=0) failure is free-run instability of a
linear model, not anticipation; narrow "classical remedy" to "direct method".
MAJOR 6: v0 Boolean-override-era results still cited (c9 upstream/downstream, c7b open-loop planning in ED Fig 2) → rerun or drop.
MAJOR 7: abstract dither range; dither cost provenance; 29/36 & 22/24; conformal band; audit extended (masked coverage, fractions, cost,
re-scored C16).
MINOR 1–17: ARX all orders in Table 1; G_op(0)=0 sentence; "between" unfalsifiable; title/noise; "independent of data volume" is synthetic;
"halves"; MPC twins v0; "overheat" sign (energy 0.57 vs 0.45); grey-OE degenerate C_m (informativity diagnosis; op/designed 0.65/0.57/0.16);
"unstable"; item 15; conformal band; "recover at 24 h"; Φ_u-weighted projection caveat; Twin Houses contrast.
