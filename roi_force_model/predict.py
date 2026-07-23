"""Run the trained e-skin ROI -> grip-force model on a trial folder.

Usage:
  python roi_force_model/predict.py data/<trial_id>
  python roi_force_model/predict.py data/<trial_id> --model path/to/model.json --csv out.csv

Given a trial folder (needs eskin.csv + forces.csv; forces are used only for
the time grid + optional scoring), it re-runs the full preprocessing on THAT
trial (per-trial baseline + per-trial ROI are re-estimated -- required, the
model was trained on baseline-subtracted ROI features, not raw counts),
predicts F1+F2 per force sample, prints a summary (and R²/RMSE/MAE if the
trial has real force to compare against), and optionally writes a CSV of
time, predicted, actual.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import (  # noqa: E402
    build_samples, load_model, predict_force, metrics,
)

DEFAULT_MODEL = HERE / "outputs" / "roi_force_model.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trial_dir", help="path to data/<trial_id>/")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="model JSON")
    ap.add_argument("--csv", default=None, help="optional output CSV path")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    feats, F, t, aux = build_samples(Path(args.trial_dir))
    pred = predict_force(model, feats)

    # saturation gate (present on below-knee models): where the e-skin reading
    # is saturated, force is physically unresolvable -> flag rather than trust.
    gate = model.get("saturation_gate")
    saturated = np.zeros(len(pred), bool)
    if gate:
        saturated = feats[gate["feature"]].to_numpy(float) >= gate["threshold"]

    info = aux["info"]
    print(f"trial      : {info['trial_id']}")
    print(f"model      : {args.model}")
    print(f"samples    : {len(pred)}  (ROI={info['roi_cells']} cells, "
          f"coverage={info['coverage']:.0%})")
    print(f"predicted  : min={pred.min():.1f}  median={np.median(pred):.1f}  "
          f"max={pred.max():.1f} N")
    if gate:
        print(f"gate       : {gate['feature']} >= {gate['threshold']} ⇒ saturated; "
              f"{saturated.mean():.0%} of samples flagged unresolved (force ≥ knee)")

    # score against recorded force if it looks like real force (not a zero trial)
    if np.nanmax(F) - np.nanmin(F) > 2.0:
        s = metrics(F, pred)
        print(f"actual     : min={F.min():.1f}  median={np.median(F):.1f}  "
              f"max={F.max():.1f} N")
        print(f"score(all) : R²={s['r2']}  RMSE={s['rmse']} N  MAE={s['mae']} N")
        if gate and (~saturated).sum() > 20:
            sg = metrics(F[~saturated], pred[~saturated])
            print(f"score(gate): R²={sg['r2']}  RMSE={sg['rmse']} N  MAE={sg['mae']} N  "
                  f"(on the {(~saturated).sum()} gate-accepted samples — the honest number)")
        else:
            knee = model.get("saturation_knee_N", 30.0)
            below = F < knee
            if below.sum() > 20:
                sb = metrics(F[below], pred[below])
                print(f"score<{knee:.0f}N : R²={sb['r2']}  RMSE={sb['rmse']} N  "
                      f"MAE={sb['mae']} N  (below saturation knee)")

    if args.csv:
        pd.DataFrame({"elapsed_s": t, "pred_F_combined_N": pred,
                      "actual_F_combined_N": F, "saturated_flag": saturated.astype(int)
                      }).to_csv(args.csv, index=False)
        print(f"wrote      : {args.csv}")


if __name__ == "__main__":
    main()
