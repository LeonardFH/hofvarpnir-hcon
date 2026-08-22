import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load overlap data
df = pd.read_csv("Davis_Taniguchi_overlap.csv")

# Calculate statistics
davis_mean = df["Davis_Density"].mean()
taniguchi_mean = df["Taniguchi_Density"].mean()
mean_diff = (df["Davis_Density"] - df["Taniguchi_Density"]).mean()
std_diff = (df["Davis_Density"] - df["Taniguchi_Density"]).std()

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================
# LEFT: Histogram of density distributions
# ============================================================

ax1.hist(df["Davis_Density"], bins=30, alpha=0.6, label="Davis", color='blue', edgecolor='black', linewidth=0.5)
ax1.hist(df["Taniguchi_Density"], bins=30, alpha=0.6, label="Taniguchi", color='orange', edgecolor='black', linewidth=0.5)

ax1.axvline(davis_mean, color='blue', linestyle='dashed', linewidth=1.5, label=f'Davis mean: {davis_mean:.3f}')
ax1.axvline(taniguchi_mean, color='orange', linestyle='dashed', linewidth=1.5, label=f'Taniguchi mean: {taniguchi_mean:.3f}')

ax1.set_xlabel("Crystal Density (g/cm³)", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Density Distributions for Same Molecules", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

stats_text = f"Mean diff: {mean_diff:.4f} g/cm³\nStd diff: {std_diff:.4f}"
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# RIGHT: Correlation scatter plot
# ============================================================

ax2.scatter(df["Taniguchi_Density"], df["Davis_Density"], alpha=0.4, s=20, color='purple')

min_val = min(df["Davis_Density"].min(), df["Taniguchi_Density"].min()) - 0.05
max_val = max(df["Davis_Density"].max(), df["Taniguchi_Density"].max()) + 0.05
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect agreement")

r2 = np.corrcoef(df["Taniguchi_Density"], df["Davis_Density"])[0,1]**2
corr_text = f"R² = {r2:.4f}\nn = {len(df):,}"
ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel("Taniguchi Density (g/cm³)", fontsize=12)
ax2.set_ylabel("Davis Density (g/cm³)", fontsize=12)
ax2.set_title("Davis vs Taniguchi Densities for Same Molecules", fontsize=14, fontweight='bold')

# ✅ MOVED LEGEND TO UPPER RIGHT
ax2.legend(loc='upper right', fontsize=10)

ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim([min_val, max_val])
ax2.set_ylim([min_val, max_val])

# ============================================================
# TIGHT LAYOUT AND SAVE
# ============================================================

plt.tight_layout()
plt.savefig("davis_taniguchi_comparison.png", dpi=300, bbox_inches='tight')
#plt.savefig("davis_taniguchi_comparison.pdf", bbox_inches='tight')

print("✓ Figure saved as: davis_taniguchi_comparison.png")
print("✓ Figure saved as: davis_taniguchi_comparison.pdf")
print()
print("Statistics:")
print(f"  Davis mean:        {davis_mean:.4f} g/cm³")
print(f"  Taniguchi mean:    {taniguchi_mean:.4f} g/cm³")
print(f"  Mean difference:   {mean_diff:.4f} g/cm³")
print(f"  Std difference:    {std_diff:.4f} g/cm³")
print(f"  Correlation R²:    {r2:.4f}")
print(f"  Overlap molecules: {len(df):,}")

plt.show()