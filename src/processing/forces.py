"""Post-hoc processing of recorded force-sensor CSVs (wall_time, elapsed_s,
F1_N, F2_N -- see SessionRecorder._write_forces_csv)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ForceFeatures:
    peak_combined_n: float
    peak_f1_n: float
    peak_f2_n: float
    mean_combined_n: float
    time_to_target_s: Optional[float]


def load_forces_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["F_combined"] = df["F1_N"] + df["F2_N"]
    return df


def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def extract_features(df: pd.DataFrame, target_force_n: Optional[float] = None) -> ForceFeatures:
    combined = smooth(df["F_combined"])

    time_to_target = None
    if target_force_n is not None:
        reached = df.index[combined >= target_force_n]
        if len(reached):
            time_to_target = float(df.loc[reached[0], "elapsed_s"])

    return ForceFeatures(
        peak_combined_n=float(combined.max()),
        peak_f1_n=float(df["F1_N"].max()),
        peak_f2_n=float(df["F2_N"].max()),
        mean_combined_n=float(combined.mean()),
        time_to_target_s=time_to_target,
    )
