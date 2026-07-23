"""Train the e-skin ROI -> grip-force (F1+F2) model and save it.

Minimal, honest version:
  * Train on the DYNAMIC trial (the only one with real within-trial force range).
  * Drop SATURATED samples (F1+F2 >= knee): above the knee the e-skin reads flat
    regardless of force, so those samples only corrupt the fit (see README).
  * Add an inference-time SATURATION GATE on the e-skin reading itself
    (roi_mean >= threshold => "force >= knee, unresolved"), since at deployment
    there is no force label to decide what is in range.
  * Evaluate on a TIME-BLOCKED held-out test block (no shuffling -- neighbouring
    ~100 Hz samples are near-duplicates and would leak).

`build_and_eval()` is the single source of truth for the model + numbers;
figures.py imports it so the slide figures match the printed metrics exactly.

Run:  python roi_force_model/train.py   ->  outputs/roi_force_model.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import (  # noqa: E402
    ROOT, DATA, BASE_FEATURES, build_samples, poly2, RidgeNP,
    metrics, save_model,
)

OUT = HERE / "outputs"
TRAIN_TRIAL = DATA / "YL_grasp_dynamic_002_20260723_155859"
KNEE_N = 30.0                 # e-skin saturates above ~this force (see README)
TRAIN_FRAC = 0.70
GAP_S = 1.0                   # gap between train/test blocks (kills autocorr leakage)
ALPHAS = np.logspace(-3, 4, 15)


def _blocked_indices(t, train_frac, gap_s):
    n = len(t); cut = int(train_frac * n)
    test_start = np.searchsorted(t, t[cut] + gap_s)
    return np.arange(0, cut), np.arange(test_start, n)


def _fit(Xbase, y, alpha):
    Xexp, names = poly2(Xbase, BASE_FEATURES)
    return RidgeNP(alpha=alpha).fit(Xexp, y), names


def _select_alpha(Xb, y, sub_tr, val):
    best_a, best = ALPHAS[0], np.inf
    for a in ALPHAS:
        m, _ = _fit(Xb[sub_tr], y[sub_tr], a)
        Xv, _ = poly2(Xb[val], BASE_FEATURES)
        r = metrics(y[val], m.predict(Xv))
        if r["rmse"] < best:
            best, best_a = r["rmse"], a
    return best_a


def build_and_eval():
    """Train + evaluate. Returns everything train.py prints and figures.py plots."""
    feats, F, t, aux = build_samples(TRAIN_TRIAL)
    Xb = feats[BASE_FEATURES].to_numpy(float)
    rmean = feats["roi_mean"].to_numpy(float)

    # deployable gate: the e-skin reading typical of the force knee
    near = (F > KNEE_N - 4) & (F < KNEE_N + 4)
    roi_mean_knee = float(np.median(rmean[near]))

    # time-blocked split; below-knee sub-selection for training
    tr, te = _blocked_indices(t, TRAIN_FRAC, GAP_S)
    v_cut = int(0.80 * len(tr)); gap = int(GAP_S * 100)
    sub_tr, val = tr[:v_cut], tr[v_cut + gap:]
    below = lambda idx: idx[F[idx] < KNEE_N]  # noqa: E731

    alpha = _select_alpha(Xb, F, below(sub_tr), below(val))
    model, names = _fit(Xb[below(tr)], F[below(tr)], alpha)

    # held-out predictions (test block) from the train-block model
    Xte, _ = poly2(Xb[te], BASE_FEATURES)
    pred_te = model.predict(Xte)
    gate_ok = rmean[te] < roi_mean_knee            # samples the gate would trust

    ev = {
        "resolvable_by_label<{:.0f}N".format(KNEE_N):
            metrics(F[te][F[te] < KNEE_N], pred_te[F[te] < KNEE_N]),
        "gate_accepted": metrics(F[te][gate_ok], pred_te[gate_ok]),
        "full_block": metrics(F[te], pred_te),
    }

    # final deployable model: refit on ALL below-knee samples in the trial
    final, names = _fit(Xb[F < KNEE_N], F[F < KNEE_N], alpha)

    return dict(
        feats=feats, F=F, t=t, aux=aux, Xb=Xb, rmean=rmean,
        tr=tr, te=te, pred_te=pred_te, gate_ok=gate_ok,
        model=model, final=final, expanded_names=names, alpha=alpha,
        knee_N=KNEE_N, roi_mean_knee=roi_mean_knee, eval=ev,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = build_and_eval()
    info = r["aux"]["info"]
    g = r["eval"]["gate_accepted"]

    save_model(
        OUT / "roi_force_model.json", r["final"], BASE_FEATURES, r["expanded_names"],
        extra=dict(
            trained_on=info["trial_id"],
            train_filter=f"F1+F2 < {KNEE_N:.0f} N (saturated samples dropped)",
            n_samples=int((r["F"] < KNEE_N).sum()),
            alpha_selected=r["alpha"],
            valid_range_N=[None, KNEE_N],
            saturation_gate={"feature": "roi_mean",
                             "threshold": round(r["roi_mean_knee"], 1),
                             "rule": "roi_mean >= threshold => saturated, force unresolved"},
            eval_heldout_gate_accepted=g,
            target="F1_N + F2_N (Newtons)",
        ),
    )

    print(f"train trial : {info['trial_id']} (n={info['n']}, ROI={info['roi_cells']} cells)")
    print(f"knee        : {KNEE_N:.0f} N   e-skin gate: roi_mean >= {r['roi_mean_knee']:.0f}")
    print(f"alpha       : {r['alpha']:.3g}")
    print("held-out test block (time-blocked, no leakage):")
    for k, s in r["eval"].items():
        print(f"  {k:24s} R2={s['r2']:+.3f}  RMSE={s['rmse']:5.2f} N  MAE={s['mae']:5.2f} N  n={s['n']}")
    print(f"\nsaved -> {(OUT/'roi_force_model.json').relative_to(ROOT)}")
    print("next: `python roi_force_model/figures.py` for the slide figures")


if __name__ == "__main__":
    main()
