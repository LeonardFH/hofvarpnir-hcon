# ==============================================================================
# HÓFVARPNIRHCON — CROSS DATASET VALIDATION
#
# Davis -> Taniguchi
# Taniguchi -> Davis
#
# ==============================================================================

from hofvarpnirhcon import train_density, predict_density_batch

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
import time


INPUT_FILE = "Davis_Taniguchi_overlap.csv"

SMILES_COLUMN = "SMILES"

DAVIS_COLUMN = "Davis_Density"
TANIGUCHI_COLUMN = "Taniguchi_Density"


def train_and_test(
        train_density_column,
        test_density_column,
        label):

    print("\n")
    print("="*70)
    print(label)
    print("="*70)


    # ------------------------------------------------------
    # Create training file
    # ------------------------------------------------------

    train_df = pd.read_csv(INPUT_FILE)

    train_data = train_df[
        [SMILES_COLUMN, train_density_column]
    ].copy()

    train_data.columns = [
        "SMILES",
        "Density"
    ]

    temp_file = f"temp_{label}.csv"

    train_data.to_csv(
        temp_file,
        index=False
    )


    # ------------------------------------------------------
    # Train dictionary
    # ------------------------------------------------------

    print("\nTraining dictionary...")

    start = time.perf_counter()

    train_density(
        data_path=temp_file,
        output_path="cross_weights.pkl",
        filter_cocrystals=True,
        filter_hcon=False,
        verbose=True
    )

    train_time = time.perf_counter()-start


    # ------------------------------------------------------
    # Predict target dataset
    # ------------------------------------------------------

    print("\nPredicting target density...")

    smiles = train_df[SMILES_COLUMN].tolist()

    actual = train_df[test_density_column].values


    start = time.perf_counter()

    predictions = predict_density_batch(
        smiles_list=smiles,
        weights_path="cross_weights.pkl",
        verbose=True,
        n_cores=1
    )

    pred_time = time.perf_counter()-start


    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    mask = [
        p is not None
        for p in predictions
    ]


    y_true = actual[mask]

    y_pred = np.array(
        [p for p in predictions if p is not None]
    )


    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = root_mean_squared_error(
        y_true,
        y_pred
    )


    r2 = np.corrcoef(
        y_true,
        y_pred
    )[0,1]**2


    print("\nRESULTS")
    print("-"*40)

    print(f"Training density : {train_density_column}")
    print(f"Target density   : {test_density_column}")
    print(f"Molecules        : {len(y_true)}")

    print(f"MAE              : {mae:.5f}")
    print(f"RMSE             : {rmse:.5f}")
    print(f"R²               : {r2:.5f}")

    print(f"Train time       : {train_time:.2f}s")
    print(f"Predict time     : {pred_time:.2f}s")


    # save predictions

    output = pd.DataFrame({

        "SMILES":
            np.array(smiles)[mask],

        "Actual":
            y_true,

        "Prediction":
            y_pred,

        "Error":
            y_pred-y_true

    })


    filename = (
        label.replace(" ","_")
        +
        "_results.csv"
    )

    output.to_csv(
        filename,
        index=False
    )


    if os.path.exists(temp_file):
        os.remove(temp_file)



# ==============================================================================
# RUN BOTH DIRECTIONS
# ==============================================================================


train_and_test(
    "Davis_Density",
    "Taniguchi_Density",
    "Davis_train_Taniguchi_test"
)


train_and_test(
    "Taniguchi_Density",
    "Davis_Density",
    "Taniguchi_train_Davis_test"
)