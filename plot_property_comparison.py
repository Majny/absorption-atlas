#!/usr/bin/env python3
"""
Compare feature-absorption-vs-L0 across token properties on the same Gemma Scope SAEs
(layer 12, 16k). The project's central question: is absorption specific to first-letter spelling,
or a general SAE failure mode? If other properties show the same absorption-vs-L0 curve, it's general.

Reads results/<property>/... summaries. Currently: first_letter (results/gate/gate_summary_l0*.json)
and last_letter (results/last_letter/summary_l0*.json).
Usage: python plot_property_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent

# (label, glob) per property; both write {l0, mean_absorption_rate}
SERIES = [
    ("first-letter", "#c0392b", HERE / "results" / "gate", "gate_summary_l0*.json"),
    ("last-letter", "#2c7fb8", HERE / "results" / "last_letter", "summary_l0*.json"),
]


def load(dirpath: Path, pattern: str):
    pts = []
    for f in sorted(dirpath.glob(pattern)):
        d = json.loads(f.read_text())
        pts.append((float(d["l0"]), float(d["mean_absorption_rate"])))
    return sorted(pts)


def main():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, color, dirpath, pattern in SERIES:
        pts = load(dirpath, pattern)
        if not pts:
            print(f"(skip {label}: no files in {dirpath})")
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "o-", color=color, linewidth=2, markersize=7, label=label)
        print(f"{label}: " + ", ".join(f"L0={l:.0f}->{r:.3f}" for l, r in pts))

    ax.set_xscale("log")
    ax.set_xlabel("SAE sparsity L0 (log scale)")
    ax.set_ylabel("Mean full-absorption rate")
    ax.set_title("Feature absorption generalizes across token properties\n"
                 "Gemma-2-2b · Gemma Scope · layer 12 · width 16k")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, title="token property")
    fig.tight_layout()
    out = HERE / "results" / "absorption_property_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
