from pathlib import Path
import numpy as np
import pandas as pd

from common import (
    TAXELS, REPORTS, iter_trials, align_to_force_grid, estimate_lag_s,
    subtract_baseline, trim_idle_mask,
)

REPORTS.mkdir(parents=True, exist_ok=True)

Xs, ys, trials, subjects, targets, times = [], [], [], [], [], []
report = []

for name, eskin, force, man in iter_trials():
    E, t, F, cov = align_to_force_grid(eskin, force)
    lag = estimate_lag_s(E.sum(axis=1), F)

    # Baseline computed on the FULL (untrimmed) trial -- the long idle
    # stretches we're about to trim out are exactly what makes the 5th
    # percentile a trustworthy "no contact" reference.
    E_base = subtract_baseline(E)

    n_before = len(F)
    keep = trim_idle_mask(t, F, man)
    E_trim, t_trim, F_trim = E_base[keep], t[keep], F[keep]
    n_after = int(keep.sum())

    Xs.append(E_trim); ys.append(F_trim); times.append(t_trim)
    trials += [name] * n_after
    subjects += [(man.get("subject_id") if man else "archive")] * n_after
    targets += [(man.get("target_force_n") if man else np.nan)] * n_after

    report.append(dict(
        trial=name,
        subject=(man.get("subject_id") if man else "archive"),
        target_n=(man.get("target_force_n") if man else np.nan),
        n_before=n_before, n_after=n_after,
        kept_frac=round(n_after / n_before, 3) if n_before else 0.0,
        coverage=round(cov, 3), lag_s=round(lag, 3),
        F_min=round(float(F_trim.min()), 2) if n_after else float("nan"),
        F_max=round(float(F_trim.max()), 2) if n_after else float("nan"),
    ))
    print(f"  {name}: n_before={n_before} n_after={n_after} "
          f"({n_after/n_before:.0%} kept) cov={cov:.2f} lag={lag:+.3f}s")

X = np.vstack(Xs); y = np.concatenate(ys)
np.savez_compressed(
    REPORTS / "dataset.npz", X=X, y=y,
    trial=np.array(trials), subject=np.array(subjects),
    target=np.array(targets, float), t=np.concatenate(times),
    taxels=np.array(TAXELS),
)

df = pd.DataFrame(report)
df.to_csv(REPORTS / "trial_summary.csv", index=False)
print()
print(df.to_string(index=False))
print(f"\nTotal paired samples after trimming: {len(y):,} across {len(df)} trials")
print(f"Total before trimming: {df.n_before.sum():,}")
