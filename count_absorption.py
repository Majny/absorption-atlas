#!/usr/bin/env python3
"""
Write results/absorption_counts.json: the ablation counts behind the README / WRITEUP / paper, with
provenance, so they can be checked without the (gitignored, uncommitted) detail parquets.

Criterion (same as analyze_scalefree.py, conditioned on main-silent scored tokens): a token counts as
absorbed iff the main latent(s) are silent (<1e-8), the top-IG latent is negative, it is probe-aligned
(cos >= 0.025), and |top| >= R * |second|. Rates carry Wilson 95% intervals. For each pool the file also
records the largest absolute IG gap |top| - |second| (Chanin's original criterion needs >= 0.5..4.0) and
the per-probe-true-positive rate at R=3 (mean over letters for first-letter).

Usage:  python count_absorption.py [--results-dir results] [--out results/absorption_counts.json]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-8
COS_MIN = 0.025
RS = (1.5, 2.0, 3.0)
ABS_GAP_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

BINARY = {
    "is_capitalized_2b": ("is_capitalized/absorption_detail_l082.parquet", "is_capitalized/summary_l082.json"),
    "is_capitalized_9b": ("is_capitalized_9b/absorption_detail_l068.parquet", "is_capitalized_9b/summary_l068.json"),
}
LETTER = {
    "first_letter_2b": "gate/absorption_layer12_16k_l082.parquet",
    "first_letter_9b": "first_letter_9b/absorption_layer20_16k_l068.parquet",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(top, sec, cos, silent):
    """top/sec/cos: arrays over scored rows; silent: bool mask of main-silent rows."""
    n = int(silent.sum())
    out = {}
    for R in RS:
        hit = silent & (top < 0) & (cos >= COS_MIN) & (np.abs(top) >= R * np.abs(sec))
        k = int(hit.sum())
        out[f"R={R}"] = {"absorbed": k, "n": n, "rate": round(k / n, 4), "wilson95": wilson(k, n)}
    return out


def provenance(results, rel):
    p = results / rel
    return {"source_parquet": f"results/{rel}", "sha256": sha256(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "absorption_counts.json"))
    a = ap.parse_args()
    results = Path(a.results_dir)

    pools = {}
    for name, (rel, summ) in BINARY.items():
        df = pd.read_parquet(results / rel)
        s = json.loads((results / summ).read_text())
        top, sec = df["top_ablation"].to_numpy(), df["second_ablation"].to_numpy()
        silent = df["main_silent"].astype(bool).to_numpy()
        gap = np.abs(top) - np.abs(sec)
        c = counts(top, sec, df["top_cos"].to_numpy(), silent)
        k3 = c["R=3.0"]["absorbed"]
        pools[name] = {
            **provenance(results, rel),
            "summary_json": f"results/{summ}",
            "num_candidates": s.get("num_candidates"),
            "rows_scored": int(len(df)),
            "main_silent_scored": int(silent.sum()),
            "max_abs_ig_gap": round(float(gap.max()), 4),
            "absolute_gap_absorbed": {str(t): int((silent & (top < 0) & (df["top_cos"].to_numpy() >= COS_MIN)
                                                   & (gap >= t)).sum()) for t in ABS_GAP_THRESHOLDS},
            "conditioned": c,
            "per_probe_tp_R3": k3 / s["sample_portion"] / s["probe_true_positives"],
            "per_probe_tp_inputs": {"sample_portion": s["sample_portion"],
                                    "probe_true_positives": s["probe_true_positives"]},
        }

    for name, rel in LETTER.items():
        df = pd.read_parquet(results / rel)
        top, sec = df["top_ablation_score"].to_numpy(), df["second_ablation_score"].to_numpy()
        cos = df["top_ablation_feat_probe_cos"].to_numpy()
        silent = df["split_feat_acts"].apply(lambda v: all(x < EPS for x in v)).to_numpy()
        gap = np.abs(top) - np.abs(sec)
        per_r3, per_abs = [], []
        for _letter, g in df.groupby("letter"):
            idx = g.index.to_numpy()
            portion = g["sample_portion"].iloc[0]
            ntp = g["num_probe_true_positives"].iloc[0]
            base = silent[idx] & (top[idx] < 0) & (cos[idx] >= COS_MIN)
            r3 = base & (np.abs(top[idx]) >= 3.0 * np.abs(sec[idx]))
            ab = base & (gap[idx] >= 1.0)
            per_r3.append(r3.sum() / portion / ntp if ntp else 0.0)
            per_abs.append(ab.sum() / portion / ntp if ntp else 0.0)
        pools[name] = {
            **provenance(results, rel),
            "rows_scored": int(len(df)),
            "main_silent_scored": int(silent.sum()),
            "conditioned": counts(top, sec, cos, silent),
            "per_probe_tp_R3_mean_over_letters": round(float(np.mean(per_r3)), 4),
            "per_probe_tp_absolute_gap_1.0_mean_over_letters": round(float(np.mean(per_abs)), 4),
        }

    out = {
        "generated_by": "count_absorption.py",
        "criterion": ("absorbed iff main latent(s) silent (<1e-8), top IG attribution < 0, top latent probe "
                      "cosine >= 0.025, and |top| >= R*|second| (scale-free ratio, as in analyze_scalefree.py); "
                      "'conditioned' rates are over main-silent scored tokens, with Wilson 95% intervals"),
        "R_values": list(RS),
        "absolute_gap_criterion": "Chanin: same conditions but |top| - |second| >= threshold instead of the ratio",
        "pools": pools,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
