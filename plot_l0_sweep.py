#!/usr/bin/env python3
"""
Plot the absorption-rate-vs-L0 curve from the sweep, overlaid on the SAEBench full-absorption
reference (JumpReLU gemma-2-2b L12/16k) for context. This is the gate's headline figure:
does the *shape* of Chanin's law reproduce on the actual Gemma Scope SAEs?

Reads results/gate/gate_summary_l0*.json (written by run_gate.py per L0).
Usage: python plot_l0_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
GATE = HERE / "results" / "gate"

# SAEBench mean_full_absorption_score vs L0, JumpReLU gemma-2-2b L12 16k
# (huggingface.co/datasets/adamkarvonen/sae_bench_results_0125) — reference trend, NOT Gemma Scope.
SAEBENCH_REF = [(21.3, 0.364), (42.5, 0.256), (85.3, 0.109),
                (172.0, 0.064), (353.7, 0.027), (711.3, 0.014)]


def load_ours():
    pts = []
    for f in sorted(GATE.glob("gate_summary_l0*.json")):
        d = json.loads(f.read_text())
        pts.append((float(d["l0"]), float(d["mean_absorption_rate"])))
    return sorted(pts)


def main():
    ours = load_ours()
    if not ours:
        raise SystemExit(f"no gate_summary_l0*.json in {GATE}")
    print("our Gemma Scope points (L0, mean_full_absorption):")
    for l0, r in ours:
        print(f"  L0={l0:6.1f}  rate={r:.4f}")

    ox, oy = zip(*ours)
    rx, ry = zip(*SAEBENCH_REF)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(rx, ry, "o--", color="#b0b0b0", label="SAEBench JumpReLU (ref)", zorder=1)
    ax.plot(ox, oy, "o-", color="#c0392b", linewidth=2, markersize=7,
            label="Gemma Scope (this work)", zorder=3)
    for l0, r in ours:
        ax.annotate(f"{r:.3f}", (l0, r), textcoords="offset points", xytext=(4, 6),
                    fontsize=8, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xlabel("SAE sparsity L0 (log scale)")
    ax.set_ylabel("Mean full-absorption rate")
    ax.set_title("First-letter feature absorption vs L0\nGemma-2-2b · layer 12 · width 16k")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = GATE / "absorption_vs_l0.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
