import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hofvarpnirhcon import predict_density

WEIGHTS_PATH = "my_weights_235k.pkl"

molecules = {
    "ROY": {
        "smiles": "N#Cc1c(Nc2ccccc2[N+](=O)[O-])sc(C)c1",
        "exp_densities": [1.445, 1.447, 1.455, 1.456, 1.462, 1.473, 1.482, 1.508],
    },
    "Carbamazepine": {
        "smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
        "exp_densities": [1.34, 1.24, 1.35, 1.29],
    },
    "Aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "exp_densities": [1.40, 1.34],
    },
    "Paracetamol": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "exp_densities": [1.293, 1.336, 1.37],
    },
    "Sulfathiazole": {
        "smiles": "Nc1ccc(cc1)S(=O)(=O)Nc2nccs2",
        "exp_densities": [1.50, 1.52, 1.55, 1.58, 1.57],
    },
    "Ritonavir": {
        "smiles": "CC(C)c1nc(CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)cs1",
        "exp_densities": [1.28, 1.25],
    },
}

rows = []
for name, data in molecules.items():
    smiles = data["smiles"]
    exp = np.array(data["exp_densities"])
    median = np.median(exp)
    mean = np.mean(exp)

    try:
        result = predict_density(smiles, WEIGHTS_PATH, full_output=True)
        pred = result["density"]
    except Exception as e:
        print(f"{name}: prediction failed: {e}")
        continue

    err_median = (pred - median) / median * 100
    err_mean = (pred - mean) / mean * 100

    rows.append({
        "Molecule": name,
        "Predicted": round(pred, 3),
        "Exp_min": exp.min(),
        "Exp_max": exp.max(),
        "Exp_median": round(median, 3),
        "Exp_mean": round(mean, 3),
        "Error_vs_median_%": round(err_median, 2),
        "Error_vs_mean_%": round(err_mean, 2),
    })
    print(f"\n{name}")
    print(f"  Predicted density: {pred:.3f} g/cm³")
    print(f"  Experimental median: {median:.3f} g/cm³, mean: {mean:.3f} g/cm³")
    print(f"  Error vs median: {err_median:+.2f}%, vs mean: {err_mean:+.2f}%")

df_results = pd.DataFrame(rows)
print("\nSummary table:")
print(df_results.to_string(index=False))

# Save
df_results.to_csv("famous_polymorph_predictions.csv", index=False)


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 6, figsize=(18, 6), sharey=False)
fig.suptitle("Predicted Density vs Experimental Polymorph Densities")

for ax, name, data in zip(axes, molecules.keys(), molecules.values()):
    exp = np.array(data["exp_densities"])
    median = np.median(exp)
    mean = np.mean(exp)

    # We need pred values; assume they are in df_results if rerun, or we hardcode later.
    pred = df_results.loc[df_results["Molecule"] == name, "Predicted"].values
    pred = pred[0] if len(pred) > 0 else np.nan

    # Experimental points
    ax.scatter(exp, range(len(exp)), color="black", s=50, zorder=3, label="Exp forms")

    # Lines
    ax.axvline(median, color="green", linestyle="--", label=f"Median {median:.3f}")
    ax.axvline(mean, color="blue", linestyle=":", label=f"Mean {mean:.3f}")
    ax.axvline(pred, color="red", linewidth=2, label=f"Pred {pred:.3f}")

    ax.set_yticks(range(len(exp)))
    ax.set_yticklabels([f"F{i+1}" for i in range(len(exp))])
    ax.set_xlabel("Density (g/cm³)")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("famous_polymorphs_density_plot.png", dpi=150)
plt.show()