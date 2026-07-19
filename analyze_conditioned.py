#!/usr/bin/env python3
"""
The CRUX test (per adversarial review): recompute absorption CONDITIONED ON THE CANDIDATE POOL.

The probe-TP-denominated rate bakes in main-latent quality: a cleaner main latent fires-and-misses
on fewer tokens -> smaller candidate pool -> lower rate BY CONSTRUCTION, independent of whether
absorption generalizes. To remove that tautology, condition on the pool of tokens where the main
latent is SILENT and ask: of those, what fraction show single-dominant, concept-aligned, token-latent
absorption? If first-letter's conditioned rate still dwarfs is_capitalized's, the gap is real. If they
converge, the effect was main-latent quality and the interpretation must be retracted.

Also reports the POOLED first-letter rate (total absorbed / total probe-TP) so the aggregation matches
the single-class binary construction (removes the per-letter-mean fairness concern).
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


def letter_rows(pq):
    df = pd.read_parquet(pq)
    return [{
        "main_silent": all(a < EPS for a in r["split_feat_acts"]),
        "top": r["top_ablation_score"], "second": r["second_ablation_score"],
        "cos": r["top_ablation_feat_probe_cos"],
        "portion": r["sample_portion"], "ntp": r["num_probe_true_positives"], "letter": r["letter"],
    } for _, r in df.iterrows()]


def binary_rows(prop):
    df = pd.read_parquet(HERE / prop / "absorption_detail_l082.parquet")
    s = json.loads((HERE / prop / "summary_l082.json").read_text())
    return [{
        "main_silent": bool(r["main_silent"]), "top": r["top_ablation"], "second": r["second_ablation"],
        "cos": r["top_cos"], "portion": s["sample_portion"], "ntp": s["probe_true_positives"],
    } for _, r in df.iterrows()], s["num_candidates"]


def absorbs(row, R):
    return (row["main_silent"] and row["top"] < 0 and row["cos"] >= 0.025
            and abs(row["top"]) >= R * abs(row["second"]))


def conditioned_and_pooled(rows, R):
    ms = [r for r in rows if r["main_silent"]]
    n_abs = sum(absorbs(r, R) for r in ms)
    cond = n_abs / len(ms) if ms else float("nan")            # of main-silent, fraction absorbing
    return cond, n_abs, len(ms)


def pooled_probe_tp_rate(rows, R):
    n_abs = sum(absorbs(r, R) for r in rows if r["main_silent"])
    portion = rows[0]["portion"] if rows else 1.0
    # total probe-TP = sum over distinct letters (binary: single class)
    per = {}
    for r in rows:
        per[r.get("letter", "_")] = r["ntp"]
    ntp_total = sum(per.values())
    return (n_abs / portion) / ntp_total if ntp_total else float("nan")


def main():
    print("property        R    cond_rate  (n_abs/n_main_silent)   pooled_probeTP_rate")
    for name, pq in LETTERS.items():
        rows = letter_rows(pq)
        for R in (1.5, 2.0, 3.0):
            cond, na, nms = conditioned_and_pooled(rows, R)
            pooled = pooled_probe_tp_rate(rows, R)
            print(f"{name:14} {R:>3}   {cond:.4f}    ({na}/{nms})            {pooled:.4f}")
    for prop in BINARY:
        rows, ncand = binary_rows(prop)
        for R in (1.5, 2.0, 3.0):
            cond, na, nms = conditioned_and_pooled(rows, R)
            pooled = pooled_probe_tp_rate(rows, R)
            print(f"{prop:14} {R:>3}   {cond:.4f}    ({na}/{nms})            {pooled:.4f}")


if __name__ == "__main__":
    main()
