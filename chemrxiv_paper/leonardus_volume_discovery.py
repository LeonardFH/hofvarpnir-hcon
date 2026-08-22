"""
MASS SCALING DISCOVERY EXPERIMENT

Stage 1:
Raw atomic mass as volume proxy

V_mass = Σ(atom_mass)

Stage 2:
Cubic-root mass transformation

V_root = Σ(atom_mass^(1/3))

No scaling factor.
No bond corrections.
No NNLS.
No molecular weight classes.

Purpose:
Demonstrate the discovery path:
Can molecular composition contain information
about effective crystal volume?
"""

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

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

def get_atomic_masses(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    masses = [
        atom.GetMass()
        for atom in mol.GetAtoms()
    ]

    return masses



def molecular_mass(masses):

    return sum(masses)



def raw_mass_volume(masses):

    """
    Stage 1

    V = Σ(m_i)

    """

    return sum(masses)



def cubic_root_volume(masses):

    """
    Stage 2

    V = Σ(m_i^(1/3))

    """

    return sum(
        m ** (1/3)
        for m in masses
    )



# ============================================================
# RUN DATASET
# ============================================================

def run_dataset(name, filename):

    print("\n")
    print("="*80)
    print(name)
    print("="*80)


    df = pd.read_csv(filename)


    df = df.dropna(
        subset=[
            "SMILES",
            "Density"
        ]
    )


    print(
        "Input molecules:",
        len(df)
    )


    raw_predictions = []
    root_predictions = []

    actuals = []

    failed = []

    print("\nProcessing...")


    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):


        smiles = row["SMILES"]

        density = row["Density"]


        masses = get_atomic_masses(smiles)


        if masses is None:

            failed.append(smiles)
            continue



        mw = molecular_mass(masses)


        # Stage 1

        v_mass = raw_mass_volume(masses)

        pred_mass = mw / v_mass



        # Stage 2

        v_root = cubic_root_volume(masses)

        pred_root = mw / v_root



        raw_predictions.append(
            pred_mass
        )

        root_predictions.append(
            pred_root
        )

        actuals.append(
            density
        )



    actuals = np.array(actuals)

    raw_predictions = np.array(raw_predictions)

    root_predictions = np.array(root_predictions)



    print()

    print(
        "Valid molecules:",
        len(actuals)
    )

    print(
        "Failed:",
        len(failed)
    )


    # ========================================================
    # METRICS
    # ========================================================

    results = []


    for label, prediction in [

        ("Raw_mass",
         raw_predictions),

        ("Mass_power_1_3",
         root_predictions)

    ]:


        mae = mean_absolute_error(
            actuals,
            prediction
        )


        rmse = np.sqrt(
            np.mean(
                (actuals-prediction)**2
            )
        )


        r2 = r2_score(
            actuals,
            prediction
        )


        print("\n", label)

        print(
            "MAE:",
            f"{mae:.4f} g/cm3"
        )

        print(
            "RMSE:",
            f"{rmse:.4f}"
        )

        print(
            "R2:",
            f"{r2:.4f}"
        )


        results.append({

            "Dataset":
                name,

            "Method":
                label,

            "Valid":
                len(actuals),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2

        })


    return results



# ============================================================
# MAIN
# ============================================================


all_results = []


for name, file in DATASETS.items():

    results = run_dataset(
        name,
        file
    )

    all_results.extend(
        results
    )



summary = pd.DataFrame(
    all_results
)


print("\n")
print("="*80)
print("FINAL SUMMARY")
print("="*80)


print(
    summary.to_string(
        index=False
    )
)


summary.to_csv(
    "mass_discovery_experiment.csv",
    index=False
)


print("\nSaved:")
print(
    "mass_discovery_experiment.csv"
)