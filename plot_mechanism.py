import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.loads((Path("results/icl_sweep/sweep.json")).read_text())
acc = [r["accuracy"] for r in d]; beh = [r["behavioral"] for r in d]; rep = [r["representational"] for r in d]; ks=[r["k"] for r in d]
fig, ax1 = plt.subplots(figsize=(6.4,4.2))
ax1.plot(acc, beh, "o-", color="#c0392b", label="behavioral (ablation)")
ax1.set_xlabel("first-letter task accuracy (varied via #ICL examples)"); ax1.set_ylabel("behavioral absorption", color="#c0392b")
ax1.tick_params(axis="y", labelcolor="#c0392b"); ax1.set_ylim(0, max(beh)*1.3)
ax2 = ax1.twinx(); ax2.plot(acc, rep, "s--", color="#2c7fb8", label="representational (projection)")
ax2.set_ylabel("representational absorption", color="#2c7fb8"); ax2.tick_params(axis="y", labelcolor="#2c7fb8"); ax2.set_ylim(0, max(rep)*1.3)
for a,b,k in zip(acc,beh,ks): ax1.annotate(f"k={k}", (a,b), textcoords="offset points", xytext=(4,-10), fontsize=8)
ax1.set_title("Behavioral absorption is task-gated; representational is not\nGemma-2-2b first-letter, L12/16k")
fig.tight_layout(); fig.savefig("results/icl_sweep/mechanism.png", dpi=150); print("saved mechanism.png")
