"""
HÓFVARPNIRHCON

Combined stability figure:
(a) Independent training-set MAE scatter
(b) MAE distribution violin plot

Input:
    slice_convergence_results.csv

Output:
    slice_stability_combined.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================
# Load data
# ==============================================================

INPUT_FILE = "slice_convergence_results.csv"
OUTPUT_FILE = "slice_stability_combined.png"

df = pd.read_csv(INPUT_FILE)

# Extract each dataset
tan = df[df["Dataset"] == "Taniguchi"]
dav = df[df["Dataset"] == "Davis"]
he = df[df["Dataset"] == "He"]
jin = df[df["Dataset"] == "Jin"]

print(df.head())
print()
print("Datasets found:")
print(df["Dataset"].unique())
print()
print(f"Taniguchi: {len(tan)} training sets")
print(f"Davis: {len(dav)} training sets")
print(f"He: {len(he)} training sets")
print(f"Jin: {len(jin)} training sets")


# ==============================================================
# Create figure
# ==============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# ==============================================================
# Panel A - Scatter stability (ALL 4 DATASETS)
# ==============================================================

ax = axes[0]

# Taniguchi (largest dataset)
ax.scatter(
    tan["Slice"],
    tan["MAE"],
    s=45,
    marker="o",
    label="Taniguchi (68 sets)",
    alpha=0.7
)

# Davis
ax.scatter(
    dav["Slice"],
    dav["MAE"],
    s=55,
    marker="^",
    label="Davis (6 sets)",
    alpha=0.7
)

# He
ax.scatter(
    he["Slice"],
    he["MAE"],
    s=50,
    marker="s",
    label="He (14 sets)",
    alpha=0.7
)

# Jin
ax.scatter(
    jin["Slice"],
    jin["MAE"],
    s=50,
    marker="D",
    label="Jin (4 sets)",
    alpha=0.7
)

# Mean lines for each dataset
ax.axhline(
    tan["MAE"].mean(),
    linestyle="--",
    linewidth=1,
    color="blue",
    alpha=0.5
)

ax.axhline(
    dav["MAE"].mean(),
    linestyle=":",
    linewidth=1.5,
    color="orange",
    alpha=0.5
)

ax.axhline(
    he["MAE"].mean(),
    linestyle="-.",
    linewidth=1,
    color="green",
    alpha=0.5
)

ax.axhline(
    jin["MAE"].mean(),
    linestyle="--",
    linewidth=1,
    color="red",
    alpha=0.5
)

ax.set_ylabel("MAE (g/cm$^3$)")
ax.set_title("(a) Prediction stability across training sets")
ax.grid(alpha=0.3)
ax.legend(loc="best")

# Optional: Add a text box with summary stats
stats_text = (
    f"Mean MAE:\n"
    f"Taniguchi: {tan['MAE'].mean():.4f}\n"
    f"Davis: {dav['MAE'].mean():.4f}\n"
    f"He: {he['MAE'].mean():.4f}\n"
    f"Jin: {jin['MAE'].mean():.4f}"
)

ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    fontsize=9
)


# ==============================================================
# Panel B - Violin distribution (ALL 4 DATASETS)
# ==============================================================

ax = axes[1]

# Order datasets by size (optional)
values = [
    dav["MAE"].values,      # 6 sets
    jin["MAE"].values,      # 4 sets
    he["MAE"].values,       # 14 sets
    tan["MAE"].values       # 68 sets
]

dataset_labels = ["Davis", "Jin", "He", "Taniguchi"]

# Create violin plot
parts = ax.violinplot(
    values,
    showmeans=True,
    showmedians=True,
    showextrema=True
)

# Color the violins
colors = ["orange", "red", "green", "blue"]
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.3)

# Individual points with jitter
for i, vals in enumerate(values, start=1):
    jitter = np.random.normal(i, 0.04, size=len(vals))
    ax.scatter(
        jitter,
        vals,
        s=30,
        alpha=0.6,
        color=colors[i-1]
    )

ax.set_xticks(range(1, len(dataset_labels) + 1))
ax.set_xticklabels(dataset_labels)
ax.set_ylabel("MAE (g/cm$^3$)")
ax.set_title("(b) MAE distribution by dataset")
ax.grid(axis="y", alpha=0.3)

# Add sample size annotations
for i, label in enumerate(dataset_labels, start=1):
    n = len(values[i-1])
    ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", 
            ha="center", fontsize=9, fontweight="bold")


# ==============================================================
# Final formatting
# ==============================================================

fig.suptitle(
    "Stability Across Independent 2,500-Molecule Training Sets",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print()
print("Saved:")
print(OUTPUT_FILE)