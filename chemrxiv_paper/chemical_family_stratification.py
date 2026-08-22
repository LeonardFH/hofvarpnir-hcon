"""
HÓFVARPNIRHCON FAMILY AUTO BENCHMARKER

Runs single vs auto MW strategy for chemical families
detected by atom presence.

Datasets:
    takuyhaa.csv (Taniguchi)
    davis.csv (Davis)

Output:
    family_benchmark_results.csv
"""

import pandas as pd
import numpy as np
import os
import time
import multiprocessing

from rdkit import Chem

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

from hofvarpnirhcon import (
    train_density,
    predict_density_batch
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    "Taylor": "t1000.csv",
    "Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}

SMILES_COLUMN = "SMILES"
DENSITY_COLUMN = "Density"

# Chemical families to analyze
TARGET_ATOMS = [
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "Si",
    "B"
]

OUTPUT_FILE = "family_benchmark_results.csv"


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def contains_atom(smiles, atom_symbol):
    """Check if SMILES contains a specific atom type."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return atom_symbol in atoms


def get_family_name(smiles):
    """
    Determine which chemical family a molecule belongs to.
    Returns the first matching atom type, or 'CHON' if none.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'CHON'
    
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    
    # Check for heteroatoms in priority order
    for atom in TARGET_ATOMS:
        if atom in atoms:
            return atom
    
    return 'CHON'


# ============================================================
# RUN ONE FAMILY ON ONE DATASET
# ============================================================

def run_family(df, dataset_name, family):
    """Run benchmark for one family on one dataset."""
    
    print("\n")
    print("=" * 80)
    print(f"DATASET: {dataset_name}  |  FAMILY: {family}")
    print("=" * 80)
    
    # Filter to this family
    if family == 'CHON':
        # CHON = molecules with NO heteroatoms
        family_df = df[
            ~df["SMILES"].apply(
                lambda x: any(contains_atom(x, atom) for atom in TARGET_ATOMS)
            )
        ].copy()
    else:
        family_df = df[
            df["SMILES"].apply(
                lambda x: contains_atom(x, family)
            )
        ].copy()
    
    print(f"Molecules found: {len(family_df):,}")
    
    if len(family_df) < 200:
        print("Skipping - too few molecules (<200)")
        return []
    
    # Save temporary file
    temp_file = f"{dataset_name}_{family}_temp.csv"
    family_df[["SMILES", "Density"]].to_csv(temp_file, index=False)
    
    family_results = []
    
    for strategy in ["single", "auto"]:
        
        print("\n")
        print("-" * 60)
        print(f"STRATEGY: {strategy.upper()}")
        print("-" * 60)
        
        weight_file = f"{dataset_name}_{family}_{strategy}_weights.pkl"
        
        # ====================================================
        # TRAINING
        # ====================================================
        print("\nTRAINING PHASE")
        print("=" * 60)
        
        start_train = time.perf_counter()
        
        train_density(
            data_path=temp_file,
            output_path=weight_file,
            strategy=strategy,
            filter_hcon=False,
            filter_cocrystals=True,
            verbose=True
        )
        
        train_time = time.perf_counter() - start_train
        print(f"\nTraining Time: {train_time:.4f} seconds")
        
        # ====================================================
        # PREDICTION
        # ====================================================
        smiles_list = family_df["SMILES"].tolist()
        actuals = family_df["Density"].values
        
        print("\nPREDICTION PHASE")
        print("=" * 60)
        print(f"Predicting {len(smiles_list):,} molecules...")
        
        start_pred = time.perf_counter()
        
        predictions = predict_density_batch(
            smiles_list=smiles_list,
            weights_path=weight_file,
            verbose=True
        )
        
        pred_time = time.perf_counter() - start_pred
        
        # ====================================================
        # METRICS
        # ====================================================
        valid_mask = [p is not None for p in predictions]
        valid_actuals = np.array(actuals)[valid_mask]
        valid_predictions = [p for p in predictions if p is not None]
        
        mae = mean_absolute_error(valid_actuals, valid_predictions)
        rmse = root_mean_squared_error(valid_actuals, valid_predictions)
        r2 = np.corrcoef(valid_predictions, valid_actuals)[0, 1] ** 2
        throughput = len(valid_predictions) / pred_time
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"Valid predictions: {len(valid_predictions):,} / {len(smiles_list):,}")
        print(f"MAE:        {mae:.4f} g/cm³")
        print(f"RMSE:       {rmse:.4f} g/cm³")
        print(f"R²:         {r2:.4f}")
        print(f"\nPrediction Time: {pred_time:.4f} seconds")
        print(f"Throughput:      {throughput:.2f} molecules/sec")
        
        family_results.append({
            "dataset": dataset_name,
            "family": family,
            "strategy": strategy,
            "molecules": len(family_df),
            "valid_predictions": len(valid_predictions),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "train_seconds": train_time,
            "prediction_seconds": pred_time,
            "throughput": throughput
        })
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return family_results


# ============================================================
# MAIN
# ============================================================

def main():
    
    print("=" * 80)
    print("HÓFVARPNIRHCON FAMILY BENCHMARK")
    print("=" * 80)
    print("\nDatasets:")
    for name, file in DATASETS.items():
        print(f"  - {name}: {file}")
    print("\nChemical families:")
    for atom in TARGET_ATOMS:
        print(f"  - {atom}")
    print("  - CHON (no heteroatoms)")
    print("=" * 80)
    
    all_results = []
    
    for dataset_name, csv_file in DATASETS.items():
        
        print("\n")
        print("*" * 80)
        print(f"LOADING DATASET: {dataset_name}")
        print("*" * 80)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=[SMILES_COLUMN, DENSITY_COLUMN])
        df = df.rename(columns={
            SMILES_COLUMN: "SMILES",
            DENSITY_COLUMN: "Density"
        })
        
        print(f"Total molecules: {len(df):,}")
        
        # Get all families present in this dataset
        families_present = set()
        for smile in df["SMILES"]:
            families_present.add(get_family_name(smile))
        
        # Also run CHON if there are any
        families_present.add('CHON')
        families_present = sorted(list(families_present))
        
        print(f"Families found: {', '.join(families_present)}")
        
        for family in families_present:
            results = run_family(df, dataset_name, family)
            all_results.extend(results)
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n")
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nSummary Table:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Also print a nice summary
    print("\n")
    print("=" * 80)
    print("PERFORMANCE SUMMARY BY DATASET AND FAMILY")
    print("=" * 80)
    
    for dataset_name in DATASETS.keys():
        print(f"\n{dataset_name}:")
        subset = results_df[results_df["dataset"] == dataset_name]
        for family in subset["family"].unique():
            family_subset = subset[subset["family"] == family]
            single_mae = family_subset[family_subset["strategy"] == "single"]["MAE"].values[0] if len(family_subset[family_subset["strategy"] == "single"]) > 0 else None
            auto_mae = family_subset[family_subset["strategy"] == "auto"]["MAE"].values[0] if len(family_subset[family_subset["strategy"] == "auto"]) > 0 else None
            molecules = family_subset["molecules"].values[0]
            
            if single_mae is not None and auto_mae is not None:
                improvement = (single_mae - auto_mae) / single_mae * 100
                print(f"  {family:4s} ({molecules:6,} mols): Single={single_mae:.4f}  Auto={auto_mae:.4f}  Improvement={improvement:5.1f}%")
            elif single_mae is not None:
                print(f"  {family:4s} ({molecules:6,} mols): Single={single_mae:.4f}  Auto=N/A")
            elif auto_mae is not None:
                print(f"  {family:4s} ({molecules:6,} mols): Single=N/A  Auto={auto_mae:.4f}")


# ============================================================
# WINDOWS MULTI-CORE PROTECTION
# ============================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()