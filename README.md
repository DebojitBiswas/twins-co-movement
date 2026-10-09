# Digital twins that forecast operation best learn the controller's co-movement, not the system's response

Code, generated data and result files for the paper (Debojit Biswas, Florida Atlantic University, dbiswas2015@fau.edu).

## Layout
- `paper/` — LaTeX sources of the main text (`main.tex`) and Supplementary Information (`supplement.tex`), figures, bibliography.
- `code/` — every script used, plus the pre-registered protocol (`protocol_v1_buildings.md`), the theory notes, the quality checklist and the dated project log (`PLAN.md`, which records every result and every change made after seeing data).
- `results/` — generated logs (`.npy`), step-response truths (`truth_step*.json`), per-model scores (`c15/c16/c18/c22/c23*.json`), aggregated `summary.json` and `tables.txt`, the MPC, conformal, foundation-model and pancreas result files, and `numbers_audit.txt`, the regenerated audit of every number quoted in the paper. Fitted neural weights (47 MB per period) are omitted; `code/c15_v1_models.py` regenerates them.
- `figs/` — figure files; `internal_reviews/` — the five rounds of adversarial internal review and the revision plans.

## Regenerating the paper's numbers
```
python3 code/numbers_audit.py        # -> numbers_audit.txt (every quoted number)
python3 code/analyze_v1.py           # -> c7_v1/summary.json, tables.txt
python3 code/gen_table1.py           # -> paper Table 1
python3 code/gen_ed_tables.py        # -> Supplementary Tables 1-9
python3 code/figures_v1.py           # -> figs/v1_*.png
python3 code/build_submission.py     # -> paper/main.tex, paper/supplement.tex
```
(Run from a directory laid out as in the original project: `results/c7_v1` as `c7_v1`, `results/c7_data` as `c7_data`, etc.)

## Regenerating the data
Building: IBPSA BOPTEST (commit 0f8a467), test case `bestest_hydronic_heat_pump`, run as an FMU through `fmpy` (`code/bop.py`); `code/c14_gen_v1.py` generates the operational, dithered and designed logs and the step truth, `code/c22_small_step.py` the +0.1 step, `code/c23_shuffled_replay.py` the open-loop replays. Diabetes: `simglucose` 0.2.11 (`code/c6_pancreas.py`, `c6d_pancreas_v1.py`, `c20b/c20c`, `c6f`). Synthetic plant: `code/c1*`, `c2*`, `c5*`. Foundation model: Chronos-2 (`code/c11*`). Python 3.12, PyTorch 2.x (CPU for all but the 256-unit/4-layer models), NumPy, SciPy, fmpy.

## Licence
MIT (code). Generated data and figures: CC BY 4.0.
