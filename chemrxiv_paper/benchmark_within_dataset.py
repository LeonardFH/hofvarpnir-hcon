"""
HÓFVARPNIRHCON — MULTI-DATASET BENCHMARK RUNNER
------------------------------------------------
Trains on each dataset individually and predicts the same dataset
(within-dataset performance). Outputs a summary table with MAE, RMSE, R²
for all six datasets.

Usage:
    python one_run_benchmark.py

Outputs:
    - Console table with MAE, RMSE, R² for each dataset
    - Saves individual prediction results as CSV files
    - Saves summary table as `benchmark_summary.csv`
"""

import pandas as pd
import numpy as np
import os
import sys
import time

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from hofvarpnirhcon import train_density, predict_density_batch


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define all six datasets
DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "He": "he_35250.csv",
    "Davis": "davis.csv",
    "Jin": "wuhe.csv",
    "Taylor": "t1000.csv",
    "Mathieu": "mathieu.csv"
}

SMILES_COLUMN_NAME = "SMILES"
DENSITY_COLUMN_NAME = "Density"

STRATEGY = "auto"
FILTER_COCRYSTALS = True
FILTER_HCON = False

OUTPUT_SUMMARY = "benchmark_summary.csv"


# ==============================================================================
# RUN ONE DATASET
# ==============================================================================

def run_dataset(dataset_name, csv_file):
    """Train on full dataset, predict same dataset, return metrics."""

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print(f"FILE:    {csv_file}")
    print("=" * 70)

    # ----------------------------------------------------------
    # File verification
    # ----------------------------------------------------------

    if not os.path.exists(csv_file):
        print(f"❌ Error: '{csv_file}' not found.")
        return None

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------

    try:
        raw_df = pd.read_csv(csv_file)
        if SMILES_COLUMN_NAME not in raw_df.columns or DENSITY_COLUMN_NAME not in raw_df.columns:
            print(f"❌ Error: Required columns not found in {csv_file}")
            print(f"Expected: '{SMILES_COLUMN_NAME}' and '{DENSITY_COLUMN_NAME}'")
            print(f"Found: {list(raw_df.columns)}")
            return None

        df = raw_df[[SMILES_COLUMN_NAME, DENSITY_COLUMN_NAME]].dropna()
        df.columns = ["SMILES", "Density"]
        df = df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return None

    print(f"✅ Loaded {len(df):,} molecules.")

    # ----------------------------------------------------------
    # Temporary training file
    # ----------------------------------------------------------

    temp_file = f"_temp_{dataset_name}.csv"
    df[["SMILES", "Density"]].to_csv(temp_file, index=False)

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------

    print("Training dictionary...")
    start_train = time.perf_counter()

    weights = train_density(
        data_path=temp_file,
        output_path=f"_{dataset_name}_weights.pkl",
        strategy=STRATEGY,
        filter_cocrystals=FILTER_COCRYSTALS,
        filter_hcon=FILTER_HCON,
        verbose=False
    )

    train_time = time.perf_counter() - start_train

    # ----------------------------------------------------------
    # Predict same dataset
    # ----------------------------------------------------------

    print("Predicting...")
    smiles_list = df["SMILES"].tolist()
    actuals = df["Density"].values

    start_pred = time.perf_counter()

    predictions = predict_density_batch(
        smiles_list=smiles_list,
        weights_path=f"_{dataset_name}_weights.pkl",
        verbose=False
    )

    pred_time = time.perf_counter() - start_pred

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    valid_mask = [p is not None for p in predictions]
    valid_actuals = np.array(actuals)[valid_mask]
    valid_predictions = [p for p in predictions if p is not None]

    mae = mean_absolute_error(valid_actuals, valid_predictions)
    rmse = root_mean_squared_error(valid_actuals, valid_predictions)
    r2 = r2_score(valid_actuals, valid_predictions)

    print(f"✅ Valid predictions: {len(valid_predictions):,} / {len(smiles_list):,}")
    print(f"   MAE:  {mae:.5f} g/cm³")
    print(f"   RMSE: {rmse:.5f} g/cm³")
    print(f"   R²:   {r2:.5f}")
    print(f"   Train time: {train_time:.2f}s, Predict time: {pred_time:.2f}s")

    # ----------------------------------------------------------
    # Save individual results
    # ----------------------------------------------------------

    results_df = pd.DataFrame({
        "SMILES": np.array(smiles_list)[valid_mask],
        "Actual": valid_actuals,
        "Predicted": valid_predictions,
        "Error": np.array(valid_predictions) - valid_actuals,
        "Abs_Error": np.abs(np.array(valid_predictions) - valid_actuals)
    })

    results_df.to_csv(f"{dataset_name.lower()}_benchmark_results.csv", index=False)

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists(f"_{dataset_name}_weights.pkl"):
        os.remove(f"_{dataset_name}_weights.pkl")

    # ----------------------------------------------------------
    # Return metrics
    # ----------------------------------------------------------

    return {
        "Dataset": dataset_name,
        "N": len(valid_predictions),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Train_Time_s": train_time,
        "Predict_Time_s": pred_time
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("=" * 80)
    print("HÓFVARPNIRHCON — MULTI-DATASET BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Strategy: {STRATEGY}")
    print("=" * 80)

    all_results = []

    for dataset_name, csv_file in DATASETS.items():
        result = run_dataset(dataset_name, csv_file)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n❌ No datasets completed successfully.")
        return

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================

    summary_df = pd.DataFrame(all_results)

    print("\n\n")
    print("=" * 80)
    print("SUMMARY TABLE — WITHIN-DATASET PERFORMANCE")
    print("=" * 80)
    print()
    print(summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    ))

    # ==========================================================================
    # PAPER-FRIENDLY TABLE (Table 23 format)
    # ==========================================================================

    print("\n\n")
    print("=" * 80)
    print("TABLE 23 — HÓFVARPNIRHCON WITHIN-DATASET PERFORMANCE")
    print("=" * 80)
    print()
    print(f"{'Dataset':<12} {'N':>8} {'MAE (g/cm³)':>14} {'RMSE (g/cm³)':>15} {'R²':>10}")
    print("-" * 65)

    for _, row in summary_df.iterrows():
        print(f"{row['Dataset']:<12} {row['N']:>8,} {row['MAE']:>14.5f} {row['RMSE']:>15.5f} {row['R2']:>10.5f}")

    # ==========================================================================
    # SAVE SUMMARY
    # ==========================================================================

    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print("\n" + "=" * 80)
    print(f"💾 Summary saved to: {OUTPUT_SUMMARY}")
    print("=" * 80)


# ==============================================================================
# WINDOWS MULTI-PROCESS SAFE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()