# ==============================================================================
# HÓFVARPNIRHCON — BENCHMARK RUNNER (OneRun.py)
# ==============================================================================
# INSTRUCTIONS: Set your custom file path and column names here. 
# The script will automatically parse and normalise them for the predictor.
# ==============================================================================

CSV_FILE_NAME = "yourdata.csv"      # <-- CHANGE THIS to your file name
SMILES_COLUMN_NAME = "SMILES"       # <-- CHANGE THIS to your SMILES column
DENSITY_COLUMN_NAME = "Density"     # <-- CHANGE THIS to your density column

# ==============================================================================
# ENGINE START
# ==============================================================================
from hofvarpnirhcon import train_density, predict_density_batch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
import sys
import time

def run_benchmark():
    # Start overall execution timer
    start_overall = time.perf_counter()

    # Pre-execution file verification check
    if not os.path.exists(CSV_FILE_NAME):
        print(f"❌ Critical Error: The data file '{CSV_FILE_NAME}' was not found.")
        print("Please place the target file in this directory or update CSV_FILE_NAME.")
        sys.exit(1)

    # Step 0: Pre-load and normalise columns to prevent downstream header crashes
    try:
        raw_df = pd.read_csv(CSV_FILE_NAME)
        
        # Verify columns exist before starting the clock
        if SMILES_COLUMN_NAME not in raw_df.columns or DENSITY_COLUMN_NAME not in raw_df.columns:
            print(f"❌ Critical Error: Could not find specified columns in {CSV_FILE_NAME}")
            print(f"Expected: '{SMILES_COLUMN_NAME}' and '{DENSITY_COLUMN_NAME}'")
            print(f"Found columns: {list(raw_df.columns)}")
            sys.exit(1)
            
        # Dynamically standardise column mappings for the package requirements
        normalised_df = raw_df[[SMILES_COLUMN_NAME, DENSITY_COLUMN_NAME]].copy()
        normalised_df.columns = ["SMILES", "Density"]
        
        # Save standard temporary training matrix to disk
        temp_train_file = "hofvarpnir_temp_input.csv"
        normalised_df.to_csv(temp_train_file, index=False)
        
    except Exception as e:
        print(f"❌ Error initialising dataset: {e}")
        sys.exit(1)


    # ============================================================
    # STEP 1: Train weights (with high-precision training timer)
    # ============================================================
    print("============================================================")
    print("TRAINING PHASE")
    print("============================================================")
    start_train = time.perf_counter()

    weights = train_density(
        data_path=temp_train_file,
        output_path="my_weights.pkl",
        filter_cocrystals=True,         # Train on pure crystals only (recommended)
        filter_hcon=False,              # Train on all available atoms
        verbose=True
    )

    end_train = time.perf_counter()
    train_duration = end_train - start_train
    print(f"\n⏱️ Training Time: {train_duration:.4f} seconds")


    # ============================================================
    # STEP 2: Load Data for Predictions
    # ============================================================
    smiles_list = normalised_df["SMILES"].tolist()
    actuals = normalised_df["Density"].values


    # ============================================================
    # STEP 3: Batch Prediction (with high-precision prediction timer)
    # ============================================================
    print("\n" + "=" * 60)
    print("BATCH PREDICTION PHASE")
    print("=" * 60)
    print(f"Predicting {len(smiles_list):,} molecules...")

    start_pred = time.perf_counter()

    predictions = predict_density_batch(
        smiles_list=smiles_list,
        weights_path="my_weights.pkl",
        verbose=True
    )

    end_pred = time.perf_counter()
    pred_duration = end_pred - start_pred


    # ============================================================
    # STEP 4: Filter None Values & Calculate Final Performance Metrics
    # ============================================================
    valid_mask = [p is not None for p in predictions]
    valid_actuals = np.array(actuals)[valid_mask]
    valid_predictions = [p for p in predictions if p is not None]

    # Throughput Calculation
    throughput = len(valid_predictions) / pred_duration

    print(f"\n✅ Valid predictions: {len(valid_predictions):,} / {len(smiles_list):,}")

    mae = mean_absolute_error(valid_actuals, valid_predictions)
    rmse = root_mean_squared_error(valid_actuals, valid_predictions)
    r2 = np.corrcoef(valid_predictions, valid_actuals)[0, 1] ** 2

    print(f"\nModel Performance:")
    print(f"  MAE:        {mae:.4f} g/cm³")
    print(f"  RMSE:       {rmse:.4f} g/cm³")
    print(f"  R²:         {r2:.4f}")

    print(f"\n⏱️ Prediction Time:  {pred_duration:.4f} seconds")
    print(f"⚡ Throughput Rate:  {throughput:.2f} molecules/second")


    # ============================================================
    # STEP 5: Stop Overall Timer & Save Output
    # ============================================================
    results_df = pd.DataFrame({
        'SMILES': smiles_list[:len(valid_predictions)],
        'Actual_Density': valid_actuals,
        'Predicted_Density': valid_predictions,
        'Error': np.array(valid_predictions) - valid_actuals,
    })
    results_df.to_csv('prediction_results.csv', index=False)

    # Clean up internal placeholder file from disk
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)

    end_overall = time.perf_counter()
    overall_duration = end_overall - start_overall

    print("\n" + "=" * 60)
    print("🏁 TIMING SUMMARY")
    print("=" * 60)
    print(f"Total Dataset Size: {len(smiles_list):,} molecules")
    print(f"Training Duration:   {train_duration:.4f} seconds")
    print(f"Prediction Duration: {pred_duration:.4f} seconds")
    print(f"Overall Script Time: {overall_duration:.4f} seconds")
    print("=" * 60)
    print("💾 Results saved to: prediction_results.csv")


# This is the vital shield protecting the Windows multi-core processes!
if __name__ == '__main__':
    run_benchmark()

