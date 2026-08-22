"""
HÓFVARPNIRHCON — Learning Curve Figure Generator

Reads:
    learning_curve_summary.csv

Creates:
    learning_curve_convergence.png

"""

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE
# ============================================================

INPUT_FILE = "learning_curve_summary.csv"
OUTPUT_FILE = "learning_curve_convergence.png"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
print(df)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 6))

# Dynamically extract all unique datasets from the CSV
datasets = df["Dataset"].unique()

for dataset in datasets:
    subset = (
        df[df["Dataset"] == dataset]
        .sort_values("Training_Size")
    )

    # Using plt.plot instead of plt.errorbar for a cleaner look
    plt.plot(
        subset["Training_Size"],
        subset["MAE_mean"],
        marker="o",
        linewidth=2,
        markersize=6,
        label=dataset
    )


# ============================================================
# FORMATTING
# ============================================================

plt.xscale("log")

plt.xlabel(
    "Training molecules",
    fontsize=12
)

plt.ylabel(
    "Mean absolute error (g cm$^{-3}$)",
    fontsize=12
)

plt.title(
    "H\'ofvarpnirHCON convergence with increasing training size",
    fontsize=13
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.4
)

plt.legend(
    fontsize=11,
    loc="best"
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")
