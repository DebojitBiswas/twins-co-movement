# Paper C — quality bar (Nature Machine Intelligence). Every result must pass this before it enters the paper.

## Per result
1. **Capability control**: every learned model is first shown competent on designed (open-loop) data from the same plant; a model that
   fails its control is excluded and the exclusion reported (e.g. c5 Transformer v0, c6b sim-GRU on pancreas).
2. **Seeds and uncertainty**: ≥ 3 training seeds (≥ 5 for headline numbers); report mean ± s.d. or bootstrap 95 % CI; for data, block
   bootstrap over days/episodes, not over windows.
3. **Pre-specification**: kill criteria / predictions written down before the run (PLAN.md log or a pre-registration with timestamps);
   post-hoc analyses labelled post hoc.
4. **Honest negatives**: failed predictions (naive certificate, one-step consistency in practice, partial IV repair) are reported, not
   dropped.
5. **No leakage**: model selection never uses intervention/test data; drivers that are descendants of the input are excluded.
6. **Numbers audit**: every number in the text traced to a JSON file + script; an audit pass before submission.

## Per paper
- Lineage stated up front (Ljung; Gustavsson–Ljung–Söderström 1977; Söderström & Stoica; Van den Hof & Schrama 1993; Forssell & Ljung
  1999; offline RL / world-model literature); claims limited to what is new.
- Independent domains: synthetic (exact law), artificial pancreas (simglucose, UVA/Padova class), buildings (BOPTEST, community benchmark);
  no data shared with paper A/B (GPU testbed stays in paper B).
- Model families used by the ML community: GRU, Transformer, neural ODE, world model of offline MBRL; each with its capability control.
- Decision-level consequence (planning on the twin) in every domain.
- Reproducibility: one script per figure, fixed seeds, environment file, released code + generated data; BOPTEST commit pinned.
- Red-team round(s) by independent reviewer agents (statistics, control theory, ML) before submission; every major point answered.
- Writing: NMI format, ≤ 6 display items, Extended Data ≤ 10, Methods self-contained, Reporting Summary.
