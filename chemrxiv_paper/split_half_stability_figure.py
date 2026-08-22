import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# UPDATED Davis dataset data (from the new pi^2/phi run)
davis_data = [
    ('C', 'C', 1, 0.054169445),
    ('C', 'C', 2, 0.568548982),
    ('C', 'C', 3, 0.028911665),
    ('C', 'H', 1, 0.015293590),
    ('C', 'N', 1, 0.137412173),
    ('C', 'N', 2, 0.244566307),
    ('C', 'N', 3, 0.001117250),
    ('C', 'O', 1, 0.062901690),
    ('C', 'O', 2, 0.145893551),
    ('C', 'O', 3, 0.264851273),
    ('H', 'N', 1, 0.031994631),
    ('H', 'O', 1, 0.155658602),
    ('N', 'N', 1, 0.147703380),
    ('N', 'N', 2, 0.343665314),
    ('N', 'N', 3, 0.063684145),
    ('N', 'O', 1, 0.162649951),
    ('N', 'O', 2, 0.369413961),
    ('N', 'O', 3, 0.377460882),
    ('O', 'O', 1, 1.839848321),
]

# Convert to DataFrame
df = pd.DataFrame(davis_data, columns=['Atom1', 'Atom2', 'Order', 'Delta'])
df['Bond'] = df['Atom1'] + '-' + df['Atom2'] + ' (' + df['Order'].astype(str) + ')'
df = df.sort_values('Delta', ascending=False).reset_index(drop=True)

# Create 2-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ========================================
# PANEL 1: Horizontal Bar Chart (Left)
# ========================================

# Color mapping - highlight O-O as red
colors = ['#d73027' if bond == 'O-O (1)' else 
          '#fc8d59' if delta > 0.5 else
          '#fee08b' if delta > 0.2 else
          '#d9ef8b' if delta > 0.05 else
          '#91cf60' 
          for bond, delta in zip(df['Bond'], df['Delta'])]

bars = ax1.barh(df['Bond'], df['Delta'], color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (idx, row) in enumerate(df.iterrows()):
    ax1.text(row['Delta'] + 0.02, i, f'{row["Delta"]:.3f}', 
             va='center', fontsize=9)

# Mark mean and median (Will now correctly be 0.264 and 0.148)
ax1.axvline(df['Delta'].mean(), color='black', linestyle='--', alpha=0.6, 
            linewidth=1.5, label=f'Mean = {df["Delta"].mean():.3f}')
ax1.axvline(df['Delta'].median(), color='blue', linestyle=':', alpha=0.6,
            linewidth=1.5, label=f'Median = {df["Delta"].median():.3f}')

ax1.set_xlabel('Coefficient Difference ($\Delta$)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Bar Chart View', fontsize=13, fontweight='bold', pad=10)

# MOVED LEGEND TO TOP RIGHT
ax1.legend(loc='upper right', framealpha=0.9)

ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
ax1.set_facecolor('#f8f9fa')

# ========================================
# PANEL 2: Traditional Heatmap (Right)
# ========================================

# Create a matrix format
pivot_data = df.pivot_table(index=['Atom1', 'Order'], 
                            columns='Atom2', 
                            values='Delta', 
                            fill_value=0)

# Sort rows by max value for better visualization
row_order = pivot_data.max(axis=1).sort_values(ascending=False).index
pivot_data = pivot_data.loc[row_order]

# Create heatmap
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
            cbar_kws={'label': '$\Delta$ (Coefficient Difference)', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            ax=ax2, vmin=0, vmax=df['Delta'].max())

ax2.set_title('(b) Matrix Heatmap View', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Second Atom', fontsize=12)
ax2.set_ylabel('First Atom (Bond Order)', fontsize=12)

# Rotate x-labels for readability
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=10)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)

# ========================================
# FINAL TOUCHES
# ========================================

# Overall figure title
fig.suptitle('Davis Dataset: Bond-Overlap Coefficient Differences Between Independent Subsets',
             fontsize=15, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save
plt.savefig('davis_two_panel.png', dpi=600, bbox_inches='tight')
plt.show()

print("Saved updated davis_two_panel.png")