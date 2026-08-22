# ==============================================================================
# HÓFVARPNIRHCON — 10-FOLD CROSS-VALIDATION BENCHMARK
# ==============================================================================
#
# Runs strict 10-fold cross-validation independently on:
#
#   Taniguchi : takuyhaa.csv
#   Davis     : davis.csv
#   Taylor    : t1000.csv
#   Mathieu   : mathieu.csv
#   Jin       : wuhe.csv
#   He        : he_35250.csv
#
# Each molecule appears in exactly one held-out test fold.
# The model is trained ONLY on the remaining 90% for each fold.
#
# Outputs:
#   1. Per-fold metrics for every dataset
#   2. Dataset-level comparison table
#   3. Pooled unseen predictions for every dataset
#   4. CSV validation matrices for paper/reproducibility
#
# ==============================================================================

import os
import sys
import time
import tempfile

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from hofvarpnirhcon import train_density, predict_density_batch


# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    "Taylor": "t1000.csv",
    "Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}

SMILES_COLUMN_NAME = "SMILES"
DENSITY_COLUMN_NAME = "Density"

N_FOLDS = 10
RANDOM_STATE = 42

STRATEGY = "auto"

# ==============================================================================
# SINGLE DATASET 10-FOLD RUN
# ==============================================================================

def run_dataset_cv(dataset_name, csv_file):

    print("\n")
    print("=" * 100)
    print(f"DATASET: {dataset_name}")
    print(f"FILE:    {csv_file}")
    print("=" * 100)

    start_overall = time.perf_counter()

    # --------------------------------------------------------------------------
    # File verification
    # --------------------------------------------------------------------------

    if not os.path.exists(csv_file):
        print(f"❌ Critical Error: '{csv_file}' was not found.")
        return None

    # --------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------

    try:

        raw_df = pd.read_csv(csv_file)

        if (
            SMILES_COLUMN_NAME not in raw_df.columns
            or DENSITY_COLUMN_NAME not in raw_df.columns
        ):
            print(
                f"❌ Critical Error: required columns not found in {csv_file}"
            )
            print(f"Expected: '{SMILES_COLUMN_NAME}' and "
                  f"'{DENSITY_COLUMN_NAME}'")
            print(f"Found: {list(raw_df.columns)}")
            return None

        df = raw_df[
            [SMILES_COLUMN_NAME, DENSITY_COLUMN_NAME]
        ].dropna().reset_index(drop=True)

        df.columns = ["SMILES", "Density"]

    except Exception as e:

        print(f"❌ Error loading {csv_file}: {e}")
        return None

    print(f"✅ Loaded {len(df):,} clean molecules.")

    # --------------------------------------------------------------------------
    # K-fold splitter
    # --------------------------------------------------------------------------

    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    all_unseen_actuals = []
    all_unseen_predictions = []

    fold_records = []

    # --------------------------------------------------------------------------
    # Temporary directory for training files / weights
    # --------------------------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        for fold_idx, (train_indices, test_indices) in enumerate(
            kf.split(df), 1
        ):

            fold_start = time.perf_counter()

            # ------------------------------------------------------------------
            # Isolate training and completely unseen test data
            # ------------------------------------------------------------------

            train_df = df.iloc[train_indices].copy()
            test_df = df.iloc[test_indices].copy()

            temp_train_csv = os.path.join(
                temp_dir,
                f"{dataset_name}_train_f{fold_idx}.csv"
            )

            temp_weights_pkl = os.path.join(
                temp_dir,
                f"{dataset_name}_weights_f{fold_idx}.pkl"
            )

            train_df.to_csv(
                temp_train_csv,
                index=False
            )

            # ------------------------------------------------------------------
            # Train ONLY on the 90% training subset
            # ------------------------------------------------------------------

            train_density(
                data_path=temp_train_csv,
                output_path=temp_weights_pkl,
                filter_cocrystals=True,
                filter_hcon=False,
                verbose=False,
                strategy=STRATEGY,
            )

            # ------------------------------------------------------------------
            # Predict the completely hidden 10%
            # ------------------------------------------------------------------

            test_smiles = test_df["SMILES"].tolist()
            test_actuals = test_df["Density"].values

            predictions = predict_density_batch(
                smiles_list=test_smiles,
                weights_path=temp_weights_pkl,
                verbose=False
            )

            # ------------------------------------------------------------------
            # Remove failed predictions
            # ------------------------------------------------------------------

            valid_mask = np.array(
                [p is not None for p in predictions],
                dtype=bool
            )

            valid_actuals = np.asarray(test_actuals)[valid_mask]

            valid_predictions = np.asarray(
                [p for p in predictions if p is not None],
                dtype=float
            )

            if len(valid_predictions) == 0:

                print(
                    f"Fold {fold_idx:2d} | "
                    f"NO VALID PREDICTIONS"
                )

                continue

            # ------------------------------------------------------------------
            # Fold metrics
            # ------------------------------------------------------------------

            f_mae = mean_absolute_error(
                valid_actuals,
                valid_predictions
            )

            f_rmse = root_mean_squared_error(
                valid_actuals,
                valid_predictions
            )

            f_r2 = r2_score(
                valid_actuals,
                valid_predictions
            )

            fold_maes.append(f_mae)
            fold_rmses.append(f_rmse)
            fold_r2s.append(f_r2)

            # ------------------------------------------------------------------
            # Store all unseen predictions
            # ------------------------------------------------------------------

            all_unseen_actuals.extend(valid_actuals)
            all_unseen_predictions.extend(valid_predictions)

            # ------------------------------------------------------------------
            # Timing
            # ------------------------------------------------------------------

            fold_duration = (
                time.perf_counter() - fold_start
            )

            fold_records.append({
                "Dataset": dataset_name,
                "Fold": fold_idx,
                "Train_Size": len(train_df),
                "Test_Size": len(test_df),
                "Valid_Predictions": len(valid_predictions),
                "MAE": f_mae,
                "RMSE": f_rmse,
                "R2": f_r2,
                "Time_s": fold_duration
            })

            print(
                f"Fold {fold_idx:2d} | "
                f"Train {len(train_df):7,} | "
                f"Test {len(test_df):7,} | "
                f"Valid {len(valid_predictions):7,} | "
                f"MAE {f_mae:.5f} | "
                f"RMSE {f_rmse:.5f} | "
                f"R² {f_r2:.5f} | "
                f"{fold_duration:.2f}s"
            )

    # ==============================================================================
    # DATASET AGGREGATE
    # ==============================================================================

    if len(all_unseen_predictions) == 0:

        print(f"❌ No valid predictions generated for {dataset_name}.")
        return None

    all_unseen_actuals = np.asarray(
        all_unseen_actuals,
        dtype=float
    )

    all_unseen_predictions = np.asarray(
        all_unseen_predictions,
        dtype=float
    )

    # --------------------------------------------------------------------------
    # Mean fold metrics
    # --------------------------------------------------------------------------

    mean_fold_mae = np.mean(fold_maes)
    std_fold_mae = np.std(fold_maes, ddof=1) if len(fold_maes) > 1 else 0.0

    mean_fold_rmse = np.mean(fold_rmses)
    std_fold_rmse = (
        np.std(fold_rmses, ddof=1)
        if len(fold_rmses) > 1
        else 0.0
    )

    mean_fold_r2 = np.mean(fold_r2s)
    std_fold_r2 = (
        np.std(fold_r2s, ddof=1)
        if len(fold_r2s) > 1
        else 0.0
    )

    # --------------------------------------------------------------------------
    # Pooled metrics
    #
    # Every molecule has been predicted exactly once by a model that did not
    # train on that molecule.
    # --------------------------------------------------------------------------

    overall_mae = mean_absolute_error(
        all_unseen_actuals,
        all_unseen_predictions
    )

    overall_rmse = root_mean_squared_error(
        all_unseen_actuals,
        all_unseen_predictions
    )

    overall_r2 = r2_score(
        all_unseen_actuals,
        all_unseen_predictions
    )

    total_duration = (
        time.perf_counter() - start_overall
    )

    # ==============================================================================
    # DATASET SUMMARY
    # ==============================================================================

    print("\n" + "-" * 100)
    print(f"{dataset_name} — 10-FOLD CROSS-VALIDATION SUMMARY")
    print("-" * 100)

    print(
        f"Mean Fold MAE:       {mean_fold_mae:.5f} ± "
        f"{std_fold_mae:.5f} g/cm³"
    )

    print(
        f"Mean Fold RMSE:      {mean_fold_rmse:.5f} ± "
        f"{std_fold_rmse:.5f} g/cm³"
    )

    print(
        f"Mean Fold R²:        {mean_fold_r2:.5f} ± "
        f"{std_fold_r2:.5f}"
    )

    print("-" * 100)

    print(
        f"Overall Unseen MAE:  {overall_mae:.5f} g/cm³"
    )

    print(
        f"Overall Unseen RMSE: {overall_rmse:.5f} g/cm³"
    )

    print(
        f"Overall Unseen R²:   {overall_r2:.5f}"
    )

    print(
        f"Total Unseen:        {len(all_unseen_predictions):,}"
    )

    print(
        f"Total Runtime:       {total_duration:.2f} s"
    )

    # ==============================================================================
    # SAVE PREDICTION MATRIX
    # ==============================================================================

    results_df = pd.DataFrame({
        "Actual_Density": all_unseen_actuals,
        "Predicted_Density": all_unseen_predictions,
        "Abs_Error": np.abs(
            all_unseen_predictions - all_unseen_actuals
        )
    })

    prediction_filename = (
        f"{dataset_name.lower()}_10fold_unseen_results.csv"
    )

    results_df.to_csv(
        prediction_filename,
        index=False
    )

    print(
        f"💾 Saved unseen predictions: {prediction_filename}"
    )

    return {
        "Dataset": dataset_name,
        "N_Molecules": len(df),
        "Valid_Unseen": len(all_unseen_predictions),

        "Mean_Fold_MAE": mean_fold_mae,
        "Std_Fold_MAE": std_fold_mae,

        "Mean_Fold_RMSE": mean_fold_rmse,
        "Std_Fold_RMSE": std_fold_rmse,

        "Mean_Fold_R2": mean_fold_r2,
        "Std_Fold_R2": std_fold_r2,

        "Overall_Unseen_MAE": overall_mae,
        "Overall_Unseen_RMSE": overall_rmse,
        "Overall_Unseen_R2": overall_r2,

        "Runtime_s": total_duration,

        "Fold_Records": fold_records
    }


# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================

def main():

    benchmark_start = time.perf_counter()

    print("\n")
    print("=" * 100)
    print("HÓFVARPNIRHCON — STRICT 10-FOLD CROSS-VALIDATION BENCHMARK")
    print("=" * 100)
    print()
    print(f"Datasets:      {len(DATASETS)}")
    print(f"Folds:         {N_FOLDS}")
    print(f"Random state:  {RANDOM_STATE}")
    print()
    print(
        "Each molecule is used as an unseen test molecule exactly once."
    )
    print(
        "Each fold is trained exclusively on its corresponding 90% subset."
    )
    print("=" * 100)

    dataset_results = []
    all_fold_records = []

    # --------------------------------------------------------------------------
    # Run all four datasets
    # --------------------------------------------------------------------------

    for dataset_name, csv_file in DATASETS.items():

        result = run_dataset_cv(
            dataset_name,
            csv_file
        )

        if result is not None:

            # Keep fold records separately
            all_fold_records.extend(
                result.pop("Fold_Records")
            )

            dataset_results.append(result)

    # ==============================================================================
    # FINAL COMPARISON TABLE
    # ==============================================================================

    print("\n\n")
    print("=" * 120)
    print("FINAL 10-FOLD CROSS-VALIDATION COMPARISON")
    print("=" * 120)

    if not dataset_results:

        print("❌ No datasets completed successfully.")
        return

    comparison_df = pd.DataFrame(
        dataset_results
    )

    # --------------------------------------------------------------------------
    # Human-readable console table
    # --------------------------------------------------------------------------

    display_columns = [
        "Dataset",
        "N_Molecules",
        "Valid_Unseen",
        "Mean_Fold_MAE",
        "Std_Fold_MAE",
        "Mean_Fold_RMSE",
        "Std_Fold_RMSE",
        "Mean_Fold_R2",
        "Std_Fold_R2",
        "Overall_Unseen_MAE",
        "Overall_Unseen_RMSE",
        "Overall_Unseen_R2",
        "Runtime_s"
    ]

    display_df = comparison_df[
        display_columns
    ].copy()

    print()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    # ==============================================================================
    # PAPER-FRIENDLY SUMMARY TABLE
    # ==============================================================================

    paper_df = comparison_df[
        [
            "Dataset",
            "N_Molecules",
            "Overall_Unseen_MAE",
            "Overall_Unseen_RMSE",
            "Overall_Unseen_R2",
            "Mean_Fold_MAE",
            "Std_Fold_MAE"
        ]
    ].copy()

    paper_df.columns = [
        "Dataset",
        "N",
        "Unseen MAE (g/cm³)",
        "Unseen RMSE (g/cm³)",
        "Unseen R²",
        "Mean Fold MAE (g/cm³)",
        "Fold MAE SD"
    ]

    paper_df.to_csv(
        "tenfold_dataset_comparison.csv",
        index=False
    )

    # ==============================================================================
    # SAVE COMPLETE FOLD RECORD
    # ==============================================================================

    fold_df = pd.DataFrame(
        all_fold_records
    )

    fold_df.to_csv(
        "tenfold_all_fold_results.csv",
        index=False
    )

    # ==============================================================================
    # FINAL RUNTIME
    # ==============================================================================

    benchmark_duration = (
        time.perf_counter() - benchmark_start
    )

    print("\n")
    print("=" * 120)
    print("PAPER-FRIENDLY SUMMARY")
    print("=" * 120)

    print()

    print(
        paper_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\n")
    print("=" * 120)

    print(
        "💾 Saved comparison table:"
        " tenfold_dataset_comparison.csv"
    )

    print(
        "💾 Saved complete fold results:"
        " tenfold_all_fold_results.csv"
    )

    print(
        f"⏱ Total benchmark runtime: "
        f"{benchmark_duration:.2f} seconds"
    )

    print("=" * 120)


# ==============================================================================
# WINDOWS / MULTI-PROCESS SAFE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()