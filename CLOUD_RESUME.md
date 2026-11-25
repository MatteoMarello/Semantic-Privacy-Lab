# Cloud resume guide

## State (2026-11-25)
| Step | Status | Output |
|---|---|---|
| Generator, strategies S0–S6, attacks A1/A2, tests | done | `src/spl/`, `tests/` |
| Run 1 (preregistered measurements) | done, frozen | `results/run1_preregistered_only/` |
| Run 2 (run 1 reproduced + shifted-phrasing test set + over-masking metric) | done | `results/run2/` |
| Post-hoc probe with unseen attribute values | done | `results/probe_unseen_values/` |
| Analyses and figures | done | `results/analysis/` (`scripts/analyze.py`) |
| README, technical report, study guide | done | `README.md`, `docs/REPORT.md`, `docs/STUDY_GUIDE.md` |

Hardware used for run 2: 4 CPU threads, 15 GB RAM, no GPU; run 2 took 802 s, the probe about 3 minutes.

## Environment and run
```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt
python -m pytest -q
python scripts/analyze.py                  # seconds; regenerates results/analysis/ from saved outputs
```
Do not re-run `scripts/run_experiment.py` into `results/run1_preregistered_only/`. A full re-run is only needed to
verify reproducibility (`SPL_THREADS=4 python scripts/run_experiment.py configs/experiment.json <new dir>`).

## Possible next steps (not started)
- S3 with several training seeds, to put seed variation into the intervals.
- A pattern-free attacker stronger than A2 (e.g. an LLM reading the anonymized ticket against the register).
- Free-text tickets beyond templates.
