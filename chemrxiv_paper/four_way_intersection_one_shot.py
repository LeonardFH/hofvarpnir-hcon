"""
HÓFVARPNIRHCON 4-WAY INTERSECTION VALIDATION (ONE-SHOT)
-------------------------------------------------------
1. Loads the 4 largest datasets (Taniguchi, He, Davis, Jin).
2. Filters for CHON molecules and canonicalizes SMILES.
3. Finds the exact intersection of molecules present in ALL 4 datasets.
4. Trains on each dataset's density labels and evaluates against all 4 
   to build a 4x4 MAE confusion matrix.
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from hofvarpnirhcon import train_density, predict_density_batch

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "He":        "he_35250.csv",
    "Davis":     "davis.csv",
    "Jin":       "wuhe.csv"
}

SMILES_COL = "SMILES"
DENSITY_COL = "Density"
ALLOWED_ATOMS = {"C", "H", "O", "N"}

# We use 'single' strategy here to avoid MW-stratification noise 
# on this small intersection subset, isolating the pure bond-dictionary transfer.
STRATEGY = "single" 

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def process_smiles(smiles):
    """Returns (canonical_smiles, is_chon) or (None, False)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False
            
        # Check elements
        atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if not atoms.issubset(ALLOWED_ATOMS):
            return None, False
            
        return Chem.MolToSmiles(mol, canonical=True), True
    except:
        return None, False

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("HÓFVARPNIRHCON 4-WAY INTERSECTION VALIDATION (ONE-SHOT)")
    print("=" * 80)
    
    # ----------------------------------------------------------
    # PHASE 1: LOAD, FILTER, AND FIND INTERSECTION
    # ----------------------------------------------------------
    print("\n[PHASE 1] Loading datasets and filtering for CHON molecules...")
    data_dict = {}
    
    for name, file in DATASETS.items():
        print(f"  Loading {name} ({file})...", end=" ")
        df = pd.read_csv(file)
        
        processed = df[SMILES_COL].apply(process_smiles)
        df["CAN_SMILES"] = processed.apply(lambda x: x[0])
        df["IS_CHON"] = processed.apply(lambda x: x[1])
        
        df_chon = df[df["IS_CHON"]].dropna(subset=["CAN_SMILES"]).copy()
        df_chon = df_chon.drop_duplicates(subset=["CAN_SMILES"], keep="first")
        
        df_clean = df_chon[["CAN_SMILES", DENSITY_COL]].copy()
        df_clean = df_clean.rename(columns={DENSITY_COL: f"{name}_Density"})
        
        data_dict[name] = df_clean
        print(f"{len(df_clean):,} unique CHON molecules retained.")

    # Find intersection
    print("\n[PHASE 2] Finding 4-way intersection...")
    sets = [set(df["CAN_SMILES"]) for df in data_dict.values()]
    intersection_smiles = sets[0].intersection(*sets[1:])
    
    print(f"  -> Found {len(intersection_smiles):,} molecules present in ALL 4 datasets.")
    
    # Build Master DataFrame
    overlap_df = pd.DataFrame({"CAN_SMILES": list(intersection_smiles)})
    for name, df in data_dict.items():
        overlap_df = overlap_df.merge(df, on="CAN_SMILES", how="left")
        
    smiles_list = overlap_df["CAN_SMILES"].tolist()
    
    # ----------------------------------------------------------
    # PHASE 3: 4x4 CROSS-VALIDATION MATRIX
    # ----------------------------------------------------------
    print("\n[PHASE 3] Running 16 Train/Test Experiments...")
    dataset_names = list(DATASETS.keys())
    results_matrix = np.zeros((len(dataset_names), len(dataset_names)))
    
    for i, train_ds in enumerate(dataset_names):
        print(f"\n{'='*20} TRAINING ON: {train_ds} {'='*20}")
        
        # Prepare training CSV
        train_data = overlap_df[["CAN_SMILES", f"{train_ds}_Density"]].copy()
        train_data.columns = ["SMILES", "Density"]
        temp_file = f"temp_4way_{train_ds}.csv"
        train_data.to_csv(temp_file, index=False)
        
        # Train Dictionary
        weight_file = f"temp_4way_{train_ds}_weights.pkl"
        start_train = time.perf_counter()
        
        train_density(
            data_path=temp_file,
            output_path=weight_file,
            strategy=STRATEGY,
            filter_cocrystals=True,
            filter_hcon=False,
            verbose=False
        )
        
        # Predict
        predictions = predict_density_batch(
            smiles_list=smiles_list,
            weights_path=weight_file,
            verbose=False,
            n_cores=1
        )
        
        # Clean up valid predictions mask
        valid_mask = [p is not None for p in predictions]
        y_pred = np.array([p for p in predictions if p is not None])
        
        # Evaluate against ALL 4 datasets
        for j, test_ds in enumerate(dataset_names):
            y_true = overlap_df[f"{test_ds}_Density"].values[valid_mask]
            mae = mean_absolute_error(y_true, y_pred)
            results_matrix[i, j] = mae
            
        # Cleanup
        if os.path.exists(temp_file): os.remove(temp_file)
        if os.path.exists(weight_file): os.remove(weight_file)

    # ----------------------------------------------------------
    # PHASE 4: PRINT RESULTS
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("4x4 MAE CONFUSION MATRIX (g/cm³)")
    print("Rows = Training Dataset | Cols = Testing Dataset (Target Density)")
    print("=" * 80)
    
    # Header (Fixed backslash issue for Python < 3.12)
    col1 = "Train \\ Test"
    header = f"{col1:<12}" + "".join([f"{ds:<12}" for ds in dataset_names])
    print(header)
    print("-" * 60)
    
    # Rows
    for i, train_ds in enumerate(dataset_names):
        row_str = f"{train_ds:<12}"
        for j in range(len(dataset_names)):
            val = results_matrix[i, j]
            # Highlight the diagonal (Within-Dataset)
            if i == j:
                row_str += f"[{val:.5f}]  "
            else:
                row_str += f" {val:.5f}   "
        print(row_str)
        
    print("\n" + "=" * 80)
    print("Note: Values in [brackets] are Within-Dataset (Train=Test).")
    print("=" * 80)
    
    df = pd.read_csv("CHON_4way_overlap.csv")
    overlap = df[df["Dataset_Count"] == 4].copy()

    cols = ["Taniguchi_Density", "He_Density", "Davis_Density", "Jin_Density"]

    print("="*60)
    print("DENSITY STATISTICS FOR 1,007 OVERLAP MOLECULES")
    print("="*60)
    print(overlap[cols].describe().loc[["mean", "50%", "std", "min", "max"]].round(4))

if __name__ == "__main__":
    main()