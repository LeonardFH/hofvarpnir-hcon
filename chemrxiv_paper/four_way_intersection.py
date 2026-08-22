"""
HÓFVARPNIRHCON 4-WAY CHON OVERLAP ANALYSIS
------------------------------------------
Finds all overlapping CHON molecules across the 4 largest datasets:
Taniguchi, He, Davis, and Jin.

Outputs:
    - CHON_4way_overlap.csv (molecules present in >= 2 datasets)
    - CHON_all_unique.csv (master list of all unique CHON molecules)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from itertools import combinations

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
    print("HÓFVARPNIRHCON 4-WAY CHON OVERLAP ANALYSIS")
    print("=" * 80)
    
    data_dict = {}
    
    # 1. Load and filter datasets
    for name, file in DATASETS.items():
        print(f"\nLoading {name} ({file})...")
        df = pd.read_csv(file)
        
        # Process SMILES
        processed = df[SMILES_COL].apply(process_smiles)
        df["CAN_SMILES"] = processed.apply(lambda x: x[0])
        df["IS_CHON"] = processed.apply(lambda x: x[1])
        
        # Filter to valid CHON molecules and drop duplicates
        df_chon = df[df["IS_CHON"]].dropna(subset=["CAN_SMILES"]).copy()
        df_chon = df_chon.drop_duplicates(subset=["CAN_SMILES"], keep="first")
        
        # Keep only necessary columns and rename Density
        df_clean = df_chon[["CAN_SMILES", DENSITY_COL]].copy()
        df_clean = df_clean.rename(columns={DENSITY_COL: f"{name}_Density"})
        
        data_dict[name] = df_clean
        print(f"  -> {len(df_clean):,} unique CHON molecules retained.")

    # 2. Build Master DataFrame
    print("\n" + "=" * 80)
    print("BUILDING MASTER OVERLAP DATAFRAME")
    print("=" * 80)
    
    all_smiles = set()
    for df in data_dict.values():
        all_smiles.update(df["CAN_SMILES"].tolist())
        
    print(f"Total unique CHON molecules across all 4 datasets: {len(all_smiles):,}")
    
    master_df = pd.DataFrame({"CAN_SMILES": list(all_smiles)})
    for name, df in data_dict.items():
        master_df = master_df.merge(df, on="CAN_SMILES", how="left")
        
    # Count dataset presence
    density_cols = [f"{name}_Density" for name in DATASETS.keys()]
    master_df["Dataset_Count"] = master_df[density_cols].notna().sum(axis=1)
    
    # 3. Pairwise Overlaps & Density Comparisons
    print("\n" + "=" * 80)
    print("PAIRWISE OVERLAPS & DENSITY DISCREPANCIES")
    print("=" * 80)
    print(f"{'Dataset 1':<12} | {'Dataset 2':<12} | {'Overlap':<8} | {'Mean Abs Diff':<14} | {'Max Diff'}")
    print("-" * 75)
    
    for d1, d2 in combinations(DATASETS.keys(), 2):
        # Find overlap
        overlap_mask = master_df[[f"{d1}_Density", f"{d2}_Density"]].notna().all(axis=1)
        overlap_df = master_df[overlap_mask]
        count = len(overlap_df)
        
        if count > 0:
            diff = np.abs(overlap_df[f"{d1}_Density"] - overlap_df[f"{d2}_Density"])
            mean_diff = diff.mean()
            max_diff = diff.max()
            print(f"{d1:<12} | {d2:<12} | {count:<8,} | {mean_diff:<14.5f} | {max_diff:.5f}")
        else:
            print(f"{d1:<12} | {d2:<12} | {count:<8,} | {'N/A':<14} | {'N/A'}")

    # 4. Filter and Save
    overlap_master = master_df[master_df["Dataset_Count"] > 1].copy()
    overlap_master = overlap_master.sort_values(by="Dataset_Count", ascending=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF MULTI-DATASET MOLECULES")
    print("=" * 80)
    print(overlap_master["Dataset_Count"].value_counts().sort_index(ascending=False).to_string())
    
    # Save files
    overlap_master.to_csv("CHON_4way_overlap.csv", index=False)
    master_df.to_csv("CHON_all_unique.csv", index=False)
    
    print("\n" + "=" * 80)
    print("SAVED FILES:")
    print("  1. CHON_4way_overlap.csv (Molecules present in >= 2 datasets)")
    print("  2. CHON_all_unique.csv   (Master list of all unique CHON molecules)")
    print("=" * 80)

if __name__ == "__main__":
    main()