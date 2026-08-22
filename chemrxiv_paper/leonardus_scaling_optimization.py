"""
LEONARDUS SCALING FACTOR SEARCH

Model:

V_L = C * Σ(atom_mass^(1/3))

Density:

ρ = MW / V_L


No bond corrections.
No NNLS.
No molecular weight classes.

Purpose:
Determine whether a single global scaling factor
can convert the composition-derived volume into
an effective crystal volume.
"""


import numpy as np
import pandas as pd

from rdkit import Chem
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")



# ============================================================
# DATASETS
# ============================================================


DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    "Taylor": "t1000.csv",
    "Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}



# ============================================================
# DESCRIPTORS
# ============================================================


def get_molecular_weight(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    return sum(
        atom.GetMass()
        for atom in mol.GetAtoms()
    )



def base_volume(smiles):

    """
    Unscaled Leonardus volume

    V = Σ(atom_mass^(1/3))

    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None


    mol = Chem.AddHs(mol)


    return sum(
        atom.GetMass() ** (1/3)
        for atom in mol.GetAtoms()
    )



# ============================================================
# PREPARE DATASET ONCE
# ============================================================


def prepare_dataset(name, filename):

    print("\n")
    print("="*80)
    print("PREPARING", name)
    print("="*80)


    df = pd.read_csv(filename)


    df = df.dropna(
        subset=[
            "SMILES",
            "Density"
        ]
    )


    data = []

    failed = []


    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        smiles = row["SMILES"]


        mol = Chem.MolFromSmiles(smiles)


        if mol is None:

            failed.append(smiles)
            continue


        mw = get_molecular_weight(smiles)

        vol = base_volume(smiles)


        if mw is None or vol is None:

            failed.append(smiles)
            continue


        data.append({

            "SMILES": smiles,

            "MW": mw,

            "BaseVolume": vol,

            "Density": row["Density"]

        })


    print()
    print("Input:", len(df))
    print("Valid:", len(data))
    print("Failed:", len(failed))


    return pd.DataFrame(data)




# ============================================================
# SCAN C
# ============================================================


def scan_scaling(dataset, name):

    scales = np.arange(
        2.5,
        3.5,
        0.01
    )

    results = []

    print("\n")
    print("="*80)
    print(f"SCANNING C VALUES: {name}")
    print("="*80)

    for C in scales:

        predicted = []

        for _, row in dataset.iterrows():

            volume = (
                C *
                row["BaseVolume"]
            )

            density = (
                row["MW"]
                /
                volume
            )

            predicted.append(density)


        predicted = np.array(predicted)

        actual = dataset["Density"].values


        mae = mean_absolute_error(
            actual,
            predicted
        )


        rmse = np.sqrt(
            np.mean(
                (actual-predicted)**2
            )
        )


        r2 = r2_score(
            actual,
            predicted
        )


        results.append({

            "Dataset": name,

            "C": C,

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2

        })


        # PRINT EVERY STEP
        print(
            f"C = {C:5.2f} | "
            f"MAE = {mae:.5f} | "
            f"RMSE = {rmse:.5f} | "
            f"R2 = {r2:.5f}"
        )


    return pd.DataFrame(results)



# ============================================================
# MAIN
# ============================================================


all_results = []


for name, file in DATASETS.items():


    dataset = prepare_dataset(
        name,
        file
    )


    scan = scan_scaling(
        dataset,
        name
    )


    all_results.append(scan)



    best = scan.loc[
        scan["MAE"].idxmin()
    ]


    print("\n")
    print("="*80)
    print(name)
    print("="*80)


    print(
        "Best C:",
        best["C"]
    )

    print(
        "MAE:",
        f"{best['MAE']:.4f}"
    )

    print(
        "R2:",
        f"{best['R2']:.4f}"
    )



results = pd.concat(
    all_results,
    ignore_index=True
)


results.to_csv(
    "Leonardus_scaling_scan_C_1_to_10.csv",
    index=False
)



print("\n")
print("="*80)
print("FINAL SUMMARY")
print("="*80)



summary = (
    results
    .sort_values("MAE")
    .groupby("Dataset")
    .first()
)


print(
    summary.to_string()
)


print("\nSaved:")
print(
    "Leonardus_scaling_scan_C_1_to_10.csv"
)