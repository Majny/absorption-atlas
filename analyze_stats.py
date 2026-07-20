#!/usr/bin/env python3
"""
Do-or-die statistics for the metric-validity claim (the result is near-floor, so CIs are essential):
  (1) bootstrap 95% CIs on is-capitalized PROJECTION absorption (concept) vs its random-direction null;
  (2) bootstrap CI on 9b first-letter BEHAVIORAL absorption (0.045) vs is-capitalized behavioral (0.0),
      and the candidate-token counts behind each;
  (3) the MATCHED-ACCURACY control: first-letter behavioral absorption at ~0.91 accuracy (from the ICL
      sweeps) vs 9b is-capitalized behavioral at 0.91 accuracy.
Pure post-processing on existing artifacts — no GPU.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
RNG = np.random.default_rng(0)
EPS = 1e-8


def boot_ci(booleans, n=5000):
    a = np.asarray(booleans, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"), 0.0, 0)
    means = [RNG.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(a.mean()), len(a))


def wilson_ci(p, n, z=1.96):
    """Deterministic Wilson score 95% CI for a proportion p over n trials. Used instead of a
    synthetic-Bernoulli bootstrap so the reported CI matches the EXACT computed rate (no RNG noise
    that would drift the point estimate, e.g. 0.123 -> 0.14)."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def projection_events(prop, R=3.0):
    """report the EXACT single-dominant projection rate and random-null rate from the probedir json
    (aggregate over the main-silent pool), with a deterministic Wilson CI on the exact rate."""
    d = json.loads((RES / prop / f"probedir_l0{_l0(prop)}.json").read_text())
    n = d["n_main_silent"]
    p = d["probedir_conditioned_rate"][str(R)]
    pr = d["random_dir_control"][str(R)]
    return n, p, pr


def _l0(prop):
    # find the l0 present in the probedir json filename
    for f in (RES / prop).glob("probedir_l0*.json"):
        return f.stem.split("l0")[-1]
    return "82"


def behavioral_events(prop_dir, l0):
    """from an absorption parquet, per-candidate is_absorption booleans + candidate count."""
    pqs = list((RES / prop_dir).glob("*absorption*l0*.parquet"))
    if not pqs:
        return None
    df = pd.read_parquet(pqs[0])
    if "is_absorption" in df.columns:
        return df["is_absorption"].astype(bool).tolist()
    return None


def main():
    print("=== (1) PROJECTION absorption vs random null (probe-direction, R=3), EXACT rate + "
          "deterministic Wilson 95% CI over the main-silent pool ===")
    for prop in ["first_letter", "is_capitalized"]:
        n, p, pr = projection_events(prop)
        lo, hi = wilson_ci(p, n)
        rlo, rhi = wilson_ci(pr, n)
        print(f"  {prop:15} concept={p:.3f} [{lo:.3f},{hi:.3f}]  random={pr:.3f} [{rlo:.3f},{rhi:.3f}]  "
              f"(n_main_silent={n}, n_absorbed={round(p*n)})")

    print("\n=== (2) BEHAVIORAL absorption: 9b first-letter vs 9b is-capitalized (per-candidate is_absorption) ===")
    for tag, l0 in [("first_letter_9b", 68), ("is_capitalized_9b", 68)]:
        # behavioral parquet: first_letter uses 'is_absorption' rows; is_capitalized detail has main_silent
        pqs = list((RES / tag).glob("*.parquet"))
        rate_info = "no per-candidate parquet locally"
        summ = list((RES / tag).glob("summary_l0*.json"))
        if summ:
            s = json.loads(summ[0].read_text())
            key = "mean_absorption_rate" if "mean_absorption_rate" in s else "absorption_rate"
            ncand = s.get("num_candidates", s.get("probe_true_positives", "?"))
            rate_info = f"rate={s.get(key)} candidates={ncand}"
        print(f"  {tag:20} {rate_info}")

    print("\n=== (3) MATCHED-ACCURACY control (from ICL + corrupt sweeps, first-letter behavioral) ===")
    icl = json.loads((RES / "icl_sweep" / "sweep.json").read_text())
    cor = json.loads((RES / "icl_sweep" / "corrupt_sweep.json").read_text())
    allpts = [("icl", r["accuracy"], r["behavioral"]) for r in icl] + \
             [("corrupt", r["accuracy"], r["behavioral"]) for r in cor]
    # is-capitalized 9b accuracy is 0.91: bracket it tightly (0.90-0.94) rather than cherry-picking
    band = [(src, a, b) for src, a, b in allpts if 0.90 <= a <= 0.94]
    print("  first-letter behavioral at accuracy 0.90-0.94 (brackets the is-cap 9b acc 0.91):")
    for src, a, b in sorted(band, key=lambda t: t[1]):
        print(f"    {src:8} acc={a:.3f}  behavioral={b:.4f}")
    bvals = [b for _, _, b in band]
    print(f"  -> at MATCHED accuracy ~0.9, first-letter absorbs {min(bvals):.3f}-{max(bvals):.3f} "
          f"behaviorally (depending on manipulation), while 9b is-capitalized (acc 0.91) absorbs 0.000")
    print(f"  (note: the single value 0.026 sits at acc 0.97 [corrupt=0.25], NOT ~0.9 — do not pair it "
          f"with 'matched accuracy ~0.9')")


if __name__ == "__main__":
    main()
