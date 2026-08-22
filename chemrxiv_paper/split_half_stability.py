"""
Independent Bond Dictionary Stability Test

For each dataset:

Random split into A and B

Train:
A -> beta_A
B -> beta_B

Compare:
|beta_A - beta_B|

The test evaluates whether independently
trained bond dictionaries converge toward
similar parameter values.

Only coefficient differences are reported.
Actual bond weights are not published.
"""

import pandas as pd
import pickle
import numpy as np

from hofvarpnirhcon import train_density


# ==========================================================
# CONFIG
# ==========================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    "Taylor": "t1000.csv",
    "Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}


SEED = 42


# ==========================================================
# FUNCTION
# ==========================================================

def compare_dictionary(dataset_name, dataset_path):

    print("\n")
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)


    # -------------------------------
    # Load dataset
    # -------------------------------

    df = pd.read_csv(dataset_path)

    df = df[
        [
            "SMILES",
            "Density"
        ]
    ].dropna()

    df = df.reset_index(drop=True)


    print("Total molecules:", len(df))


    # -------------------------------
    # Random split
    # -------------------------------

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    half = len(df)//2


    group_A = df.iloc[:half].copy()
    group_B = df.iloc[half:].copy()


    print(
        "Group A:",
        len(group_A),
        " Group B:",
        len(group_B)
    )


    file_A = f"{dataset_name}_A.csv"
    file_B = f"{dataset_name}_B.csv"


    group_A.to_csv(
        file_A,
        index=False
    )

    group_B.to_csv(
        file_B,
        index=False
    )


    # -------------------------------
    # Train dictionaries
    # -------------------------------

    train_density(
        data_path=file_A,
        output_path=f"{dataset_name}_weights_A.pkl",
        strategy="single",
        filter_hcon=False,
        filter_cocrystals=False,
        verbose=False
    )


    train_density(
        data_path=file_B,
        output_path=f"{dataset_name}_weights_B.pkl",
        strategy="single",
        filter_hcon=False,
        filter_cocrystals=False,
        verbose=False
    )


    # -------------------------------
    # Load dictionaries
    # -------------------------------

    with open(
        f"{dataset_name}_weights_A.pkl",
        "rb"
    ) as f:
        weights_A = pickle.load(f)


    with open(
        f"{dataset_name}_weights_B.pkl",
        "rb"
    ) as f:
        weights_B = pickle.load(f)


    beta_A = weights_A["global"]
    beta_B = weights_B["global"]


    # -------------------------------
    # Compare
    # -------------------------------

    all_bonds = sorted(
        set(beta_A.keys())
        |
        set(beta_B.keys())
    )


    deltas = []


    print("\nBond coefficient differences")
    print("-" * 70)


    for bond in all_bonds:

        if bond in beta_A and bond in beta_B:

            delta = abs(
                float(beta_A[bond])
                -
                float(beta_B[bond])
            )

            deltas.append(delta)


            print(
                f"{str(bond):20s}"
                f"Δ={delta:.9f}"
            )


        else:

            print(
                f"{str(bond):20s}"
                "Not present in both dictionaries"
            )


    deltas = np.array(deltas)


    print("\nSummary")
    print("-" * 70)

    print(
        "Bond types compared:",
        len(deltas)
    )

    if len(deltas) > 0:

        print(
            "Maximum Δ:",
            np.max(deltas)
        )

        print(
            "Mean Δ:",
            np.mean(deltas)
        )

        print(
            "Median Δ:",
            np.median(deltas)
        )

        print(
            "Coefficient RMSE:",
            np.sqrt(
                np.mean(
                    deltas**2
                )
            )
        )


    return deltas



# ==========================================================
# RUN ALL DATASETS
# ==========================================================

results = {}


for name, path in DATASETS.items():

    results[name] = compare_dictionary(
        name,
        path
    )


print("\nFinished all datasets.")