# Round 2 — statistics referee (2026-10-08). MAJOR REVISION again. Most damaging: the "ten non-overlapping starts ≥48 h apart" do not exist —
realised spacings 18/30/18/30/18/30/48/48/48 h (c14 l.67 interleaving); 24-h truth EXACTLY ZERO at 1/10 (P1), 2/10 (P2), 6/10 (P3) starts (heat
pump off in both arms → step never applied) → P3 effectively n=4; CIs degenerate; point estimates survive (ratio-of-sums 0.27/0.18/0.24 on responsive starts).
MAJOR 1 regenerate ≥48-h-spaced responsive starts; effective n; CIs with seed+start variance; explain null starts.
MAJOR 2 capability clause dropped unregistered (sim-GRU fails in P1: 1.27 vs 1.10).
MAJOR 3 "designed recover 0.9–1.3" → 0.54–1.30; borderline cells.
MAJOR 4 ranges misquoted (0.18–0.37; ARX P3 0.89; noise families 0.27–0.74 on dither; energy 20 %).
MAJOR 5 "within range" overclaim: in-RMSE and coverage over ALL samples (14/31/51 % out of range); ARX also under-covers in P3 (0.49–0.60) → in-range masks primary.
MAJOR 6 Decisions omit designed-twin MPC (0.60/0.62); V5 scored without test; n=5 s.d.; MPC twins trained on v0 logs.
MAJOR 7 pancreas Methods stale; BG0 169–586; one-sided Fisher unstated; cap-hit fraction; 5/30 excluded w/o sensitivity; dosing meal seed 0 only.
MAJOR 8 bootstrap misdescribed (iid over overlapping windows pooled across seeds; ratio CI excludes seed variance) → moving-block over days/starts; hierarchical.
MINOR m1–m10 (P3 not winter; P1 test overlaps P2 train; "s.d. 0.8–1.2"; designed coverage 83–99; grey designed; dither cost is v0 P1; in-range fractions 86/69/49; next-step GRU size-dependent not "unstable"; untraceable 7–80×/0.17/Chronos; dither logs share weather → "realisations").
