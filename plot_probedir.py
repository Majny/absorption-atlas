#!/usr/bin/env python3
"""
Probe-direction absorption vs a random-direction sparsity baseline, at the strict single-dominant
threshold (R=3). The gap between concept and random is the real (non-sparsity) absorption signal.
Spelling shows the largest gap; structural properties show real-but-smaller gaps (is-capitalized is
the robust anchor; suffix/all-caps are underpowered, marked).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent / "results"
ORDER = ["first_letter", "last_letter", "is_capitalized", "suffix_ed", "suffix_ing", "all_caps"]
LABELS = {"first_letter": "first-letter", "last_letter": "last-letter",
          "is_capitalized": "is-capitalized", "suffix_ed": "suffix -ed",
          "suffix_ing": "suffix -ing*", "all_caps": "all-caps*"}
SPELLING = {"first_letter", "last_letter"}
R = "3.0"


def main():
    props, concept, random_, ns = [], [], [], []
    for p in ORDER:
        d = json.loads((HERE / p / "probedir_l082.json").read_text())
        props.append(LABELS[p])
        concept.append(d["probedir_conditioned_rate"][R])
        random_.append(d["random_dir_control"][R])
        ns.append(d["n_main_silent"])

    x = np.arange(len(props))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cc = ["#c0392b" if ORDER[i] in SPELLING else "#2c7fb8" for i in range(len(props))]
    ax.bar(x - w / 2, concept, w, label="concept direction", color=cc)
    ax.bar(x + w / 2, random_, w, label="random direction (sparsity baseline)",
           color="#b0b0b0", hatch="//")
    for i, (c, r, n) in enumerate(zip(concept, random_, ns)):
        ax.annotate(f"{c/r:.0f}x" if r > 0 else "", (x[i] - w / 2, c), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8, color="#333")
        ax.annotate(f"n={n}", (x[i], -0.06), textcoords="offset points", xytext=(0, 0),
                    ha="center", fontsize=7, color="#888", annotation_clip=False)
    ax.set_xticks(x)
    ax.set_xticklabels(props, rotation=20, ha="right")
    ax.set_ylabel("single-dominant-latent rate (R=3, conditioned)")
    ax.set_title("Representational feature absorption vs a sparsity baseline\n"
                 "Gemma-2-2b · layer 12 · 16k  (red=spelling, blue=structural; * = underpowered)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(concept) * 1.15)
    fig.tight_layout()
    out = HERE / "probedir_concept_vs_random.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")
    for p, c, r in zip(props, concept, random_):
        print(f"  {p:16} concept={c:.3f} random={r:.3f} excess={c-r:+.3f}")


if __name__ == "__main__":
    main()
