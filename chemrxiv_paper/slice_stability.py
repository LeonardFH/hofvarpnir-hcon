# ==============================================================================
# HÓFVARPNIRHCON
#
# 2500 Molecule Slice Dictionary Convergence Experiment
#
# Protocol:
#
# Dataset
# |
# |--> random shuffle
# |
# |--> exact 2500 molecule slices
# |
# |--> train single global dictionary on ONE slice
# |
# |--> predict REMAINING molecules only
# |
# |--> record metrics
#
# Incomplete final slices are discarded.
#
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from hofvarpnirhcon import (
    train_density,
    predict_density_batch
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    #"Taylor": "t1000.csv",
    #"Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}


SMILES_COLUMN = "SMILES"

DENSITY_COLUMN = "Density"

SLICE_SIZE = 2500

SHUFFLE_SEED = 42

TRAIN_STRATEGY = "single"

OUTPUT_FILE = (
    "slice_convergence_results.csv"
)


# ==============================================================================
# LOAD DATASET
# ==============================================================================

def load_dataset(filename):

    df = pd.read_csv(filename)

    df = df[
        [
            SMILES_COLUMN,
            DENSITY_COLUMN
        ]
    ].dropna()


    df.columns = [
        "SMILES",
        "Density"
    ]


    df = df.reset_index(drop=True)

    return df



# ==============================================================================
# RUN ONE DATASET
# ==============================================================================

def run_dataset(dataset_name, filename):

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)


    df = load_dataset(filename)


    print(
        f"Loaded molecules: {len(df):,}"
    )


    # --------------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=SHUFFLE_SEED
    ).reset_index(drop=True)



    smiles = df["SMILES"].tolist()

    truth = df["Density"].values



    # --------------------------------------------------------------
    # Complete slices only
    # --------------------------------------------------------------

    n_slices = (
        len(df)
        //
        SLICE_SIZE
    )


    print(
        f"Complete slices: {n_slices}"
    )


    results = []



    # --------------------------------------------------------------
    # Slice loop
    # --------------------------------------------------------------

    for slice_id in range(n_slices):


        start = (
            slice_id
            *
            SLICE_SIZE
        )


        end = (
            start
            +
            SLICE_SIZE
        )


        print()
        print("-" * 80)

        print(
            f"{dataset_name} "
            f"slice {slice_id+1}/{n_slices}"
        )

        print(
            f"Training molecules: {start}:{end}"
        )



        # ----------------------------------------------------------
        # Training slice
        # ----------------------------------------------------------

        training_df = df.iloc[
            start:end
        ]


        # ----------------------------------------------------------
        # Prediction set = everything EXCEPT training slice
        # ----------------------------------------------------------

        prediction_mask = np.ones(
            len(df),
            dtype=bool
        )


        prediction_mask[start:end] = False



        prediction_smiles = [
            smiles[i]
            for i in range(len(smiles))
            if prediction_mask[i]
        ]


        prediction_truth = truth[
            prediction_mask
        ]



        train_file = (
            "_temporary_slice_training.csv"
        )


        weights_file = (
            "_temporary_slice_weights.pkl"
        )



        training_df.to_csv(
            train_file,
            index=False
        )



        # ----------------------------------------------------------
        # Train single dictionary
        # ----------------------------------------------------------

        train_density(

            data_path=train_file,

            output_path=weights_file,

            strategy=TRAIN_STRATEGY,

            filter_hcon=False,

            filter_cocrystals=False,

            verbose=False

        )



        # ----------------------------------------------------------
        # Predict remaining molecules only
        # ----------------------------------------------------------

        prediction = predict_density_batch(

            smiles_list=prediction_smiles,

            weights_path=weights_file,

            verbose=False

        )


        prediction = np.array(
            prediction,
            dtype=float
        )



        valid = (

            np.isfinite(prediction)

            &

            np.isfinite(prediction_truth)

        )



        mae = mean_absolute_error(

            prediction_truth[valid],

            prediction[valid]

        )


        rmse = root_mean_squared_error(

            prediction_truth[valid],

            prediction[valid]

        )


        r2 = r2_score(

            prediction_truth[valid],

            prediction[valid]

        )



        print(
            f"Prediction molecules: {valid.sum():,}"
        )

        print(
            f"MAE  : {mae:.6f}"
        )

        print(
            f"RMSE : {rmse:.6f}"
        )

        print(
            f"R2   : {r2:.6f}"
        )



        results.append({

            "Dataset":
                dataset_name,

            "Slice":
                slice_id + 1,

            "Training_Size":
                SLICE_SIZE,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2

        })



    return results



# ==============================================================================
# MAIN
# ==============================================================================

def main():

    start_time = time.perf_counter()


    all_results = []



    for dataset_name, filename in DATASETS.items():


        dataset_results = run_dataset(

            dataset_name,

            filename

        )


        all_results.extend(
            dataset_results
        )



    results = pd.DataFrame(
        all_results
    )


    results.to_csv(
        OUTPUT_FILE,
        index=False
    )



    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


    print(results)


    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    print()

    print(
        f"Runtime: {elapsed/60:.2f} minutes"
    )



if __name__ == "__main__":

    main()