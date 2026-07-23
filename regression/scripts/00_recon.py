"""Step 1 recon: sanity-check column names, saturation ceiling, resting force
level (incl. archive bias-correction question), and manifest task_kind."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TAXELS = [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]

DATA_TRIALS = [
    "P1_002_20260721_155648",
    "narges_target_10_002_20260722_173151",
    "P1_003_20260721_155726",
    "narges_target_15_003_20260722_173534",
    "PT_target_15_001_20260722_153324",
    "P1_002_20260721_161939",
    "AM_hold_001_20260721_165006",
    "PT_target_30_002_20260722_153507",
    "PT_target_45_003_20260722_153749",
    "PT_target_45_004_20260722_154404",
]
ARCHIVE_TRIALS = ["archive_165516", "archive_173729"]

with open(DATA / "force_bias_calibration.json") as fh:
    bias = json.load(fh)
print(f"force_bias_calibration.json: {bias}\n")

print("--- column name check (first data/ trial) ---")
eskin0 = pd.read_csv(DATA / DATA_TRIALS[0] / "eskin.csv", nrows=1)
force0 = pd.read_csv(DATA / DATA_TRIALS[0] / "forces.csv", nrows=1)
print("eskin.csv columns match expected R00_C00..R15_C15 + wall_time/elapsed_s:",
      list(eskin0.columns[:2]) + ["...256 taxel cols..."] == ["wall_time", "elapsed_s", "...256 taxel cols..."]
      if False else (set(TAXELS) <= set(eskin0.columns) and "elapsed_s" in eskin0.columns))
print("forces.csv columns:", list(force0.columns))

print("\n--- manifest task_kind check (data/ trials) ---")
for t in DATA_TRIALS:
    man = json.loads((DATA / t / "manifest.json").read_text())
    print(f"  {t}: task_kind={man.get('task_kind')!r} target_force_n={man.get('target_force_n')}")

print("\n--- resting force level (median of first 50 force rows) ---")
rest_levels = {}
for t in DATA_TRIALS + ARCHIVE_TRIALS:
    force = pd.read_csv(DATA / t / "forces.csv")
    f1 = force["F1_N"].head(50).median()
    f2 = force["F2_N"].head(50).median()
    rest_levels[t] = (f1, f2)
    tag = "archive" if t in ARCHIVE_TRIALS else "data/"
    print(f"  [{tag}] {t}: F1_med={f1:.3f} F2_med={f2:.3f} sum={f1+f2:.3f}")

print("\n--- saturation-ceiling proxy (max observed taxel value per trial) ---")
maxvals = {}
for t in DATA_TRIALS + ARCHIVE_TRIALS:
    eskin = pd.read_csv(DATA / t / "eskin.csv")
    mv = eskin[TAXELS].to_numpy(float).max()
    maxvals[t] = mv
    print(f"  {t}: max_taxel_value={mv:.3f}")
print(f"\n  overall max across all copied trials: {max(maxvals.values()):.3f}")

print("\n--- archive bias-correction assessment ---")
print(f"  data/ trials resting F1/F2 medians (above) are the reference.")
print(f"  bias file BIAS1={bias['BIAS1']:.3f} BIAS2={bias['BIAS2']:.3f}")
for t in ARCHIVE_TRIALS:
    f1, f2 = rest_levels[t]
    print(f"  {t}: raw resting F1={f1:.3f} (vs -BIAS1={-bias['BIAS1']:.3f}), "
          f"F2={f2:.3f} (vs -BIAS2={-bias['BIAS2']:.3f})")
