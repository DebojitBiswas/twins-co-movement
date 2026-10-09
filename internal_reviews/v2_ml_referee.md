# Round 3 — ML referee (2026-10-09). Verdict: MINOR REVISION, no new experiments.
Most damaging: clean 256-unit next-step GRU (no noise) learns 0.25 with op error 0.44 K — same as sim-GRU; the 48-unit clean GRU in Fig. 2 is
not a working model (in-range RMSE 3.5/2.0/5.8 K vs persistence 1.1/1.7/1.4). So "the recipe, not the architecture, carries the bias" is
contradicted by the same recipe at two sizes; transformer half of the ablation is genuine and robust across the c17 sweep (0.49–1.10, never
inside the noise-TF interval). Attribute to "whatever buys operational accuracy".
MAJOR: 1 stale truth in Methods/ED (fixed with stats round). 2 abstract "repairs every family to 0.5–1.1" overstated (noise families 0.2–0.7).
3 attribution (above). 4 synthetic cell counts 7/12 vs audit 29/36 and 22/24. 5 dither cost inconsistent (0.9 K / 15–21% vs 0.3–0.9 K / few %);
use P1 amplitude series. 6 designed recovery 0.66–1.28 (fixed); add two-clause capability table. 7 HP sweep unreported in Methods/ED.
8 "a fifth of the way" mis-computed → quarter to two-fifths. 9 conformal 48/81/40 not in tables.txt (masked pooled) — define, ED table.
10 "within 10–25%" → ARX(48)/PEM within 6–24%.
MINOR m1–m18: abstract ≤200 words; Fig 4 s.e.m./0.54 (fixed); ED Fig 4/5 ranges; uncited bib entries; "diverged"→"degraded"; Chronos 2-K vs
2.4-K; intro independence claim split; "overheat" unsigned; DART→dither lineage, scheduled sampling cousin; 0.72–0.72; ARX anticipating
number; CF v0 protocol; 0.31→0.30; "consistent with"; dithered MPC paired diff; Methods seeds (2–3 GPU).
