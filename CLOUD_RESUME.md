# Cloud resume guide

## State (2026-11-25)
| Step | Status | Output |
|---|---|---|
| Generator, strategies S0–S6, attacks A1/A2, 8 tests | done | `src/spl/`, `tests/` |
| Run 1 (preregistered measurements) | done | `results/run1_preregistered_only/` |
| Run 2 (adds shifted-phrasing test set + over-masking metric) | **not run** (interrupted; smoke test passed) | — |
| README results, report | **not written** | `README.md` |

Run 1 findings to keep in mind: S3 (trained detector) reaches 0.000 re-identification only in-distribution (same
templates as training) — run 2 tests it on unseen phrasings; category macro-F1 is saturated at 1.000 for all
strategies, so embedding similarity and "other words kept" are the informative utility measures.

## Environment and run
```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt
python -m pytest -q
python scripts/run_experiment.py        # ~40 min on 6 CPU threads; regenerates identical data (seeded)
```

## Resume prompt for Claude Code on the web
> Read CLAUDE.md and CLOUD_RESUME.md. Check pytest passes, run `python scripts/run_experiment.py` once, commit
> results/ (not logs). Then write the README "Results" section and docs/REPORT.md strictly from results/RESULTS.md
> and results/run1_preregistered_only/, state which preregistered hypotheses (H1–H4) are supported, mark post-hoc
> analyses, and list limitations. Do not change the data generator or the preregistration.
