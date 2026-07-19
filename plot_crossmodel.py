import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# (label, task_accuracy, behavioral_absorption, color, marker)
pts = [
    ("first-letter · 2b", 0.95, 0.032, "#c0392b", "o"),
    ("first-letter · 9b", 0.96, 0.045, "#c0392b", "s"),
    ("is-capitalized · 2b", 0.60, 0.000, "#2c7fb8", "o"),
    ("is-capitalized · 9b", 0.91, 0.000, "#2c7fb8", "s"),
]
fig, ax = plt.subplots(figsize=(6.6,4.3))
for lab,acc,beh,c,m in pts:
    ax.scatter(acc, beh, color=c, marker=m, s=120, edgecolor="k", zorder=3)
    ax.annotate(lab, (acc,beh), textcoords="offset points", xytext=(6, 6 if beh>0 else 8), fontsize=8)
ax.axhline(0, color="#ccc", lw=0.8)
ax.set_xlabel("model's task accuracy on the property")
ax.set_ylabel("behavioral (ablation) absorption")
ax.set_xlim(0.5,1.0); ax.set_ylim(-0.006, 0.055)
ax.set_title("Structural behavioral-absorption stays 0 even when the model does the task\n"
             "spelling absorbs on both models; is-capitalized on neither (o=2b ●, s=9b ■)")
fig.tight_layout(); fig.savefig("results/crossmodel_behavioral.png", dpi=150); print("saved crossmodel_behavioral.png")
