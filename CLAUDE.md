# CLAUDE.md

Onboarding for AI agents (and humans) working in this repo.

**Read [PROJECT.md](PROJECT.md) first** — it holds the project context,
decisions, findings, and current direction that are *not* obvious from the code.
Treat it as the source of truth and keep it updated when facts change.

Then, as needed:
- [README.md](README.md) — how to set up, record, align, and plot.
- [docs/alignment_pipeline.md](docs/alignment_pipeline.md) — how offline alignment works.
- [project_overview.md](project_overview.md) — the original one-paragraph brief.

Quick orientation:
- **Goal:** estimate grip force / effort from wearable contact sensing
  (e-skin + force; EMG is ground-truth/comparison; FMG is the intended
  electrode-free EMG replacement — see PROJECT.md).
- **Data:** one folder per trial under `data/<trial_id>/`
  (`manifest.json` + `eskin.csv` + `forces.csv` + `emg_raw.txt`).
- **Offline pipeline:** `src/processing/` + `scripts/align_trial.py`.
- **Env:** `conda activate eskin` then `pip install -r requirements.txt`.
