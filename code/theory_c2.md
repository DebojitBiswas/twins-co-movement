# C2 — theory (v1, 2026-10-06). Draft for the Methods section.

## Setting
Discrete-time, scalar input u, scalar output y, linear time-invariant plant with an unmeasured disturbance:

    y_t = G(q) u_t + H(q) e_t,                         e_t white, variance σ_e²            (plant)
    u_t = C(q) (r_t − y_t) + F(q) e_t,                 r_t logged exogenous excitation,   (controller)
                                                       r ⟂ e, spectrum Φ_r

G, H, C, F are stable rational transfer operators; F ≠ 0 means the controller reacts to the disturbance directly (anticipation,
feed-forward of a forecast, a human who sees what the sensor does not). The loop is well posed and internally stable;
S = (1 + GC)^{-1}. Then

    u = S C r + S (F − C H) e,        y = S G C r + S (G F + H) e.                                              (1)

## Theorem 1 (what a simulation-error twin learns)
Let a model y_t = Ĝ(q) u_t + v_t be fitted by minimising the simulation (output) error E[(y − Ĝu)²] over a class rich enough to contain
every stable (possibly non-causal) LTI filter. The minimiser is the Wiener filter Ĝ = Φ_yu / Φ_u, and

    Ĝ(ω) = γ²(ω) G(ω) + (1 − γ²(ω)) G_op(ω),

    G_op = (G F + H) / (F − C H),       γ² = |S C|² Φ_r / Φ_u = |coh(u, r)|².

Proof: substitute (1); cross-spectra of r and e vanish; Φ_yu = |S|² [ G |C|² Φ_r + (GF + H)(F − CH)* σ_e² ], Φ_u = |S|² [ |C|² Φ_r +
|F − CH|² σ_e² ]; divide. γ² equals the squared coherence of u and r because the r-driven part of u is S C r. ∎

Remarks.
(a) G_op is the ratio in which the disturbance moves the output and the input — the co-movement the operational log contains. Under
pure feedback (F = 0) G_op = −1/C: the twin is the negative inverse controller (classical: Ljung 1999 §13.4; Söderström & Stoica 1989).
(b) Data size: γ² and G_op are properties of the regime, not of N. More operational data shrinks variance around Ĝ, not the bias Ĝ − G.
(c) Sign: with proportional action C(0) = K_p, G_op(0) = −1/K_p (opposite sign to a positive plant gain) — "heating cools", "insulin
raises glucose". With integral action C(0) = ∞, G_op(0) = 0: "the actuator does nothing" at steady state.
(d) Ensembles: members differ only through optimisation and finite-sample noise around the same Ĝ; their spread → 0 as N → ∞ while
the error → |Ĝ − G| > 0. Epistemic uncertainty is blind to this error by construction.
(e) Parametric/causal classes: the fitted model is the projection of Ĝ onto the class in the Φ_u-weighted norm (Ljung 1999, Eq. 8.71
analogue); empirically (c2_collapse) neural simulators land within ≈ 0.2 (relative) of Ĝ.

## Proposition 2 (one-step learners with history)
Let the model be the one-step predictor ŷ_{t+1} = f(y^t, u^{t}) fitted by least squares with unrestricted f (the limit of a sequence
model with long context trained by teacher forcing).
(i) If F = 0 and the closed loop is informative (r persistently exciting, or the controller nonlinear / time-varying / of sufficient
order), the predictor's u-coefficients identify G: the predictor is the joint input–output / direct-method predictor with a free noise
model, consistent under feedback (Ljung 1999 Thm 13.x; Forssell & Ljung 1999). The c1 high-order ARX converges (0.08 at N = 2·10⁵).
(ii) If F ≠ 0, u_t carries information about e_t that is not in (y^t, u^{t−1}); the one-step regression attributes the part of y_{t+1}
explained by e_t to u_t — an omitted-variable bias (paper A, Lemma 1, is the scalar case) that does not vanish with N.
(iii) In practice the asymptotic consistency of (i) is not reached by neural one-step learners at N = 2·10⁵ (c5c: GRU-1s 0.98,
Transformer 0.88–4.1 under pure feedback); rollout of a one-step model compounds the small mis-attribution. Report as a limitation of (i).

## Corollary 3 (decisions)
A planner that optimises u against Ĝ applies a gain error Ĝ − G to the true plant; for |Ĝ(0)| ≪ |G(0)| the planner saturates the
actuator (overshoot, c5: plans 3–23 K off for a 3-K target); for sign(Ĝ) ≠ sign(G) in the planning band it drives the output away from
the target. The twin's own predicted tracking error is small because the plan is optimal for Ĝ.

## Log-only check
From the log alone one can (i) fit the controller Ĉ (u on r − y history), (ii) estimate γ²(ω) where r is logged, (iii) compute the
twin's learned response and compare it with γ²·(unknown G) + (1 − γ²)·Ĝ_op: if γ² ≈ 0 in the band of interest the twin cannot be
trusted there whatever its fit; if the twin's response is close to −1/Ĉ (or Ĝ_op) it has learned the controller. Caution (c4c): the
naive "share of input variance explained by logged drivers" is NOT γ² when those drivers are themselves model inputs; the relevant
quantity is the share explained by excitation that is not a model input (partial coherence).

## Repair
Two-stage / instrumental training (Van den Hof & Schrama 1993): replace u by its projection on lagged r before simulation-error training;
consistent for G when γ² > 0 in the band. Synthetic: 0.30 (IV-GRU) vs 0.9 (plain), 0.11 (two-stage OE). Pancreas: partial (≈ 2× with
meal boluses, none in the reactive loop). Physical structure (sign constraints on conductances / gains) excludes G_op with the wrong sign.
