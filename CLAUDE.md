# Semantic Privacy Lab — instructions for Claude Code

- Research repo. All public text in English. Every number must come from `results/`: `results/run*/` are written by
  `scripts/run_experiment.py`, `results/analysis/` by `scripts/analyze.py`. Never edit generated files by hand.
- `results/run1_preregistered_only/` is the preregistered run and must not be modified or regenerated.
- Synthetic data only. Never add real personal data, and never copy code or data from MedAnon
  (neural-wave/2026-Fuma-c-anduma): this is an independent study.
- `docs/PREREGISTRATION.md` defines the preregistered analyses. The shifted-phrasing test set and the
  "other words kept" metric are post-hoc additions and must be labelled as such.
- Oracle strategies (S4–S6) are upper bounds, not systems. No formal privacy guarantee may be claimed.
- `python -m pytest -q` must pass before committing. Read `CLOUD_RESUME.md` for what remains.
