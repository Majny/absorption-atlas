import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
icl = json.loads(Path("results/icl_sweep/sweep.json").read_text())
cor = json.loads(Path("results/icl_sweep/corrupt_sweep.json").read_text())
fig, ax1 = plt.subplots(figsize=(6.6,4.3))
ax2 = ax1.twinx()
# behavioral (left), representational (right); two manipulations, distinct markers
for data, mk, lab in [(icl,"o","fewer ICL examples"), (cor,"^","corrupted ICL answers")]:
    a=[r["accuracy"] for r in data]; b=[r["behavioral"] for r in data]; rp=[r["representational"] for r in data]
    ax1.scatter(a,b, marker=mk, color="#c0392b", s=55, label=f"behavioral · {lab}")
    ax2.scatter(a,rp, marker=mk, color="#2c7fb8", s=45, facecolors="none", label=f"representational · {lab}")
ax1.set_xlabel("first-letter task accuracy")
ax1.set_ylabel("behavioral (ablation) absorption", color="#c0392b"); ax1.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_ylabel("representational (projection) absorption", color="#2c7fb8"); ax2.tick_params(axis="y", labelcolor="#2c7fb8")
ax1.set_ylim(0,0.04); ax2.set_ylim(0,0.75)
ax1.set_title("Behavioral SAE-absorption is gated by task accuracy;\nrepresentational absorption is not (Gemma-2-2b first-letter, L12/16k)")
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, fontsize=7, loc="center right", frameon=False)
fig.tight_layout(); fig.savefig("results/icl_sweep/mechanism.png", dpi=150); print("saved combined mechanism.png")
