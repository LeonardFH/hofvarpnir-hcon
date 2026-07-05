# HófvarpnirHCON

A modular Python framework for molecular property prediction from SMILES strings.

HófvarpnirHCON (pronounced "HOFF-varp-neer-HCON") is designed as a fast and extensible framework for predicting molecular properties of organic compounds containing C, H, O, and N.

Named after the flying horse of the Norse goddess Gná, reflecting the software's intended speed and range across molecular property spaces.

## Current Status

At present, the package implements:

- Crystal density prediction for organic molecules

The framework is designed with extensibility in mind, allowing additional molecular property predictors to be added in future versions.

## A Friendly Note

Hi there,

I built HófvarpnirHCON because crystal density prediction should be fast, transparent, and accessible. I'm glad you found it.

If you need to get in touch: leonardfhaasbroek@gmail.com

## License

This project is distributed under the BSD 3-Clause License.

## Data Sources

The training data may be obtained from:

- Davis, J. V.; Marrs, F. W.; Cawkwell, M. J.; Manner, V. W. Machine Learning Models for High Explosive Crystal Density and Performance. Chem. Mater. 2024, 36, 11109–11118. DOI: 10.1021/acs.chemmater.4c01978

- Mathieu, D. Sensitivity of Energetic Materials: Theoretical Relationships to Detonation Performance and Molecular Structure. Ind. Eng. Chem. Res. 2017, 56, 8191–8201. DOI: 10.1021/acs.iecr.7b02021

These datasets are available as Supporting Information with their respective papers.

## Community Benchmarks

If you use HófvarpnirHCON on your own dataset, I invite you to share your results.

Email: **leonardfhaasbroek@gmail.com**

Please include:
- MAE, RMSE, R²
- Number of molecules
- Number of cocrystals
- Dataset description and source (if public)

Results will be posted here (with your permission).

## Documentation

For a detailed explanation of the method, see [docs/METHODS.md](docs/METHODS.md).

## Installation

pip install hofvarpnir-hcon

## Quick Start: Train and Predict in Thonny

```python
#Copy and paste this entire script into Thonny and run it:

from hofvarpnirhcon import train_density, predict_density, predict_density_batch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# ============================================================
# STEP 1: Download a dataset from one of the papers above
# Save it as "trainingdata.csv" with columns: SMILES, Density
# ============================================================

# ============================================================
# STEP 2: Train your own weights
# ============================================================

print("Training model...")
weights = train_density(
    data_path="trainingdata.csv",
    output_path="my_weights.pkl",
    filter_cocrystals=True,         # Train on pure crystals only (recommended)
    filter_hcon=True,               # Train on H,C,O,N atoms only (recommended)
    verbose=True
)
print("Training complete! Weights saved to my_weights.pkl")

# ============================================================
# STEP 3: Load the dataset for predictions
# ============================================================

df = pd.read_csv("trainingdata.csv")
smiles_list = df["SMILES"].tolist()
actuals = df["Density"].values

# ============================================================
# STEP 4: Single molecule prediction
# ============================================================

print("\n" + "=" * 60)
print("SINGLE MOLECULE PREDICTION")
print("=" * 60)

test_smiles = smiles_list[0]
test_actual = actuals[0]
pred = predict_density(test_smiles, weights_path="my_weights.pkl")
print(f"SMILES: {test_smiles}")
print(f"Actual density: {test_actual:.4f} g/cm³")
print(f"Predicted density: {pred:.4f} g/cm³")
print(f"Error: {abs(pred - test_actual):.4f} g/cm³")

# ============================================================
# STEP 5: Batch prediction on entire dataset
# ============================================================

print("\n" + "=" * 60)
print("BATCH PREDICTION")
print("=" * 60)

print(f"Predicting {len(smiles_list)} molecules...")
predictions = predict_density_batch(
    smiles_list=smiles_list,
    weights_path="my_weights.pkl",
    verbose=True
)

# ============================================================
# STEP 6: Calculate MAE and show results (FILTER NONE VALUES)
# ============================================================

# Filter out None values (failed predictions)
valid_mask = [p is not None for p in predictions]
valid_actuals = np.array(actuals)[valid_mask]
valid_predictions = [p for p in predictions if p is not None]

print(f"\n✅ Valid predictions: {len(valid_predictions):,} / {len(smiles_list):,}")

if len(valid_predictions) == 0:
    print("❌ No valid predictions. Check your SMILES strings.")
    exit()

mae = mean_absolute_error(valid_actuals, valid_predictions)
rmse = np.sqrt(np.mean((np.array(valid_predictions) - valid_actuals) ** 2))
r2 = np.corrcoef(valid_predictions, valid_actuals)[0, 1] ** 2

print(f"\nModel Performance:")
print(f"  MAE:  {mae:.4f} g/cm³")
print(f"  RMSE: {rmse:.4f} g/cm³")
print(f"  R²:   {r2:.4f}")

print("\nFirst 10 predictions:")
print("-" * 70)
print(f"{'SMILES':<35} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("-" * 70)

for i in range(min(10, len(valid_predictions))):
    smiles = smiles_list[i][:35]
    actual = valid_actuals[i]
    pred = valid_predictions[i]
    error = abs(pred - actual)
    print(f"{smiles:<35} {actual:>10.4f} {pred:>10.4f} {error:>10.4f}")

print("-" * 70)
print(f"MAE: {mae:.4f} g/cm³")
print("\n✅ All done! Weights saved to my_weights.pkl")

# ============================================================
# STEP 7: Save results to CSV
# ============================================================

results_df = pd.DataFrame({
    'SMILES': smiles_list[:len(valid_predictions)],
    'Actual_Density': valid_actuals,
    'Predicted_Density': valid_predictions,
    'Error': np.array(valid_predictions) - valid_actuals,
    'Abs_Error': np.abs(np.array(valid_predictions) - valid_actuals),
})

results_df.to_csv('prediction_results.csv', index=False)
print("\n💾 Results saved to: prediction_results.csv")

# ============================================================
# USAGE EXAMPLES
# ============================================================

# Single molecule prediction:
from hofvarpnirhcon import predict_density

density = predict_density("CCO", weights_path="my_weights.pkl")
print(f"{density:.3f} g/cm³")

# Batch prediction:
from hofvarpnirhcon import predict_density_batch

smiles_list = ["CCO", "CC", "c1ccccc1", "O"]
results = predict_density_batch(smiles_list, weights_path="my_weights.pkl")

for smiles, density in zip(smiles_list, results):
    print(f"{smiles}: {density:.3f} g/cm³")
```

## Performance

- MAE:   ~0.0300 g/cm³ on CHON molecules
- Speed: ~1,800 molecules/second (1 core/thread)
- Speed: ~2,700 molecules/second (2 core/thread)
- Speed: ~3,500 molecules/second (4 core/thread - max achieved)

## Tips for Best Performance

For optimal accuracy, we recommend training separate dictionaries for each chemical family:

- **HCON only** (C, H, N, O) — best overall performance
- **HCON + F** — fluorine-containing molecules
- **HCON + Cl** — chlorine-containing molecules
- **HCON + S** — sulfur-containing molecules
- **HCON + P** — phosphorus-containing molecules

**Avoid mixing different heteroatom types** (e.g., S and Cl together) in a single training run, as this can degrade prediction accuracy.

For molecules containing rare halogens (Br, I), we recommend using the HCON-only dictionaries, as there is insufficient data to train reliable halogen-specific overlaps.

## Important Note on Polymorphs

The model predicts a single crystal density per SMILES string. For molecules with multiple known polymorphs (e.g., ROY, carbamazepine), the prediction corresponds to a **centroid** density within the experimental range. It does **not** predict individual polymorph forms.

## Citation

If you use this software in your research, please cite:

Haasbroek, L. F. (2026). HófvarpnirHCON: Fast dictionary-based crystal density prediction. 