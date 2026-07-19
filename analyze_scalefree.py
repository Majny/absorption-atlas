#!/usr/bin/env python3
"""
Cross-property absorption comparison under a SCALE-FREE criterion.

The strict Chanin metric uses an ABSOLUTE IG-gap threshold (|top|-|second| >= 1.0) calibrated to the
26-letter logit metric. The binary True/False metric has a ~30x smaller IG scale, so the absolute
gap rejects everything and gives 0 — which could be a scale artifact, not a real absence. To compare
spelling vs structural properties fairly we recompute absorption with a scale-free DOMINANCE ratio
(|top| >= R*|second|) that transfers across metric scales, applied uniformly to every property.

A candidate counts as absorption iff: main latent(s) silent (<EPS), top-IG latent negative,
top latent concept-aligned (cos>=0.025), and top dominates the runner-up (|top| >= R*|second|).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-8
HERE = Path(__file__).parent / "results"
LETTERS = {"first_letter": HERE / "gate" / "absorption_layer12_16k_l082.parquet",
           "last_letter": HERE / "last_letter" / "absorption_layer12_16k_l082.parquet"}
BINARY = ["is_capitalized", "all_caps", "suffix_ing", "suffix_ed"]


def letter_rate(parquet, crit):
    df = pd.read_parquet(parquet)
    per = []
    for _letter, g in df.groupby("letter"):
        portion = g["sample_portion"].iloc[0]
        ntp = g["num_probe_true_positives"].iloc[0]
        n_abs = sum(crit(r) for _, r in g.iterrows()) / portion
        per.append(n_abs / ntp if ntp else 0.0)
    return float(np.mean(per))


def binary_rate(prop, crit):
    df = pd.read_parquet(HERE / prop / "absorption_detail_l082.parquet")
    s = json.loads((HERE / prop / "summary_l082.json").read_text())
    portion, ntp = s["sample_portion"], s["probe_true_positives"]
    n_abs = sum(crit(r) for _, r in df.iterrows()) / portion
    return float(n_abs / ntp if ntp else 0.0)


# --- criteria (row accessors differ between letter and binary parquets) ---
def strict_letter(r):  # original absolute-gap metric (should reproduce the reported number)
    ms = all(a < EPS for a in r["split_feat_acts"])
    top, sec = r["top_ablation_score"], r["second_ablation_score"]
    return ms and top < 0 and (abs(top) - abs(sec)) >= 1.0 and r["top_ablation_feat_probe_cos"] >= 0.025


def sf_letter(R):
    def c(r):
        ms = all(a < EPS for a in r["split_feat_acts"])
        top, sec = r["top_ablation_score"], r["second_ablation_score"]
        return ms and top < 0 and r["top_ablation_feat_probe_cos"] >= 0.025 and abs(top) >= R * abs(sec)
    return c


def sf_binary(R):
    def c(r):
        top, sec = r["top_ablation"], r["second_ablation"]
        return bool(r["main_silent"]) and top < 0 and r["top_cos"] >= 0.025 and abs(top) >= R * abs(sec)
    return c


def main():
    print("=== validation: strict absolute-gap recompute (should match reported ~0.034 / ~0.064) ===")
    for name, pq in LETTERS.items():
        print(f"  {name:16} strict = {letter_rate(pq, strict_letter):.4f}")

    print("\n=== scale-free dominance ratio |top| >= R*|second| (uniform across properties) ===")
    Rs = [1.5, 2.0, 3.0]
    header = "property".ljust(16) + "".join(f"R={R}".rjust(10) for R in Rs)
    print(header)
    for name, pq in LETTERS.items():
        row = name.ljust(16) + "".join(f"{letter_rate(pq, sf_letter(R)):.4f}".rjust(10) for R in Rs)
        print(row)
    for prop in BINARY:
        row = prop.ljust(16) + "".join(f"{binary_rate(prop, sf_binary(R)):.4f}".rjust(10) for R in Rs)
        print(row)


if __name__ == "__main__":
    main()
