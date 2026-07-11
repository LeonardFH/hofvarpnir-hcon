# ==============================================================================
# HÓFVARPNIRHCON — BENCHMARK RUNNER (TenFoldRun.py)
# ==============================================================================
# INSTRUCTIONS: Set your custom file path and column names here. 
# The script will execute a strict 10-fold cross-validation on unseen splits.
# ==============================================================================

CSV_FILE_NAME = "yourdata.csv"      # <-- CHANGE THIS to your file name
SMILES_COLUMN_NAME = "SMILES"       # <-- CHANGE THIS to your SMILES column
DENSITY_COLUMN_NAME = "Density"     # <-- CHANGE THIS to your density column

# ==============================================================================
# ENGINE START
# ==============================================================================
import os
import sys
import time
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from hofvarpnirhcon import train_density, predict_density_batch

def run_cross_validation():
    # Start overall execution timer
    start_overall = time.perf_counter()

    # Pre-execution file verification check
    if not os.path.exists(CSV_FILE_NAME):
        print(f"❌ Critical Error: The data file '{CSV_FILE_NAME}' was not found.")
        print("Please place the target file in this directory or update CSV_FILE_NAME.")
        sys.exit(1)

    print(f"📦 Loading {CSV_FILE_NAME} for strict 10-fold validation loop...")
    try:
        raw_df = pd.read_csv(CSV_FILE_NAME)
        
        # Verify columns exist before starting
        if SMILES_COLUMN_NAME not in raw_df.columns or DENSITY_COLUMN_NAME not in raw_df.columns:
            print(f"❌ Critical Error: Could not find specified columns in {CSV_FILE_NAME}")
            print(f"Expected: '{SMILES_COLUMN_NAME}' and '{DENSITY_COLUMN_NAME}'")
            print(f"Found columns: {list(raw_df.columns)}")
            sys.exit(1)
            
        # Dynamically map custom parameters into standard internal layouts
        df = raw_df[[SMILES_COLUMN_NAME, DENSITY_COLUMN_NAME]].dropna().reset_index(drop=True)
        df.columns = ["SMILES", "Density"]
        
    except Exception as e:
        print(f"❌ Error initialising dataset: {e}")
        sys.exit(1)

    print(f"✅ Loaded {len(df):,} clean rows. Initialising splits...")
    print("=" * 85)
    print(f"{'Fold':<8} | {'Train Size':<12} | {'Test Size':<11} | {'Valid Preds':<12} | {'Fold MAE (g/cm³)':<16} | {'Time (s)':<8}")
    print("=" * 85)

    # Setup KFold splitter
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    all_unseen_actuals = []
    all_unseen_predictions = []

    # Create a secure temporary directory to hold intermediate weight/data fragments
    with tempfile.TemporaryDirectory() as temp_dir:
        
        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(df), 1):
            fold_start = time.perf_counter()
            
            # Isolate the 90/10 split dataframes
            train_df = df.iloc[train_indices]
            test_df = df.iloc[test_indices]
            
            # Save temporary training chunk
            temp_train_csv = os.path.join(temp_dir, f"temp_train_f{fold_idx}.csv")
            temp_weights_pkl = os.path.join(temp_dir, f"temp_weights_f{fold_idx}.pkl")
            train_df.to_csv(temp_train_csv, index=False)
            
            # Train ONLY on the 90% background subset (Muted logs for layout clarity)
            train_density(
                data_path=temp_train_csv,
                output_path=temp_weights_pkl,
                filter_cocrystals=True,
                filter_hcon=False,
                verbose=False
            )
            
            # Extract target unseen features for batch prediction
            test_smiles = test_df["SMILES"].tolist()
            test_actuals = test_df["Density"].values
            
            # Run batch prediction loop over the 10% completely hidden segment
            predictions = predict_density_batch(
                smiles_list=test_smiles,
                weights_path=temp_weights_pkl,
                verbose=False
            )
            
            # Strip out unpredicted or broken formatting gaps
            valid_mask = [p is not None for p in predictions]
            valid_actuals = np.array(test_actuals)[valid_mask]
            valid_predictions = np.array([p for p in predictions if p is not None])
            
            if len(valid_predictions) == 0:
                print(f"Fold {fold_idx:<3} | Failed to resolve target structures.")
                continue
                
            # Calculate metrics for current fold split
            f_mae = mean_absolute_error(valid_actuals, valid_predictions)
            f_rmse = root_mean_squared_error(valid_actuals, valid_predictions)
            
            try:
                f_r2 = np.corrcoef(valid_predictions, valid_actuals)[0, 1] ** 2
            except Exception:
                f_r2 = 0.0
                
            # Log metrics to global tracking arrays
            fold_maes.append(f_mae)
            fold_rmses.append(f_rmse)
            fold_r2s.append(f_r2)
            
            all_unseen_actuals.extend(valid_actuals)
            all_unseen_predictions.extend(valid_predictions)
            
            fold_end = time.perf_counter()
            fold_duration = fold_end - fold_start
            
            print(f"Fold {fold_idx:<3} | {len(train_df):<12,} | {len(test_df):<11,} | {len(valid_predictions):<12,} | {f_mae:.4f} g/cm³     | {fold_duration:.2f}s")

    # ==============================================================================
    # FINAL AGGREGATE SUMMARY
    # ==============================================================================
    print("=" * 85)
    print("\n🏁 FINAL 10-FOLD VERIFICATION REPORT")
    print("-" * 45)
    print(f"Average Fold MAE:        {np.mean(fold_maes):.4f} g/cm³")
    print(f"Average Fold RMSE:       {np.mean(fold_rmses):.4f} g/cm³")
    print(f"Average Fold R²:         {np.mean(fold_r2s):.4f}")
    print("-" * 45)

    # True global metrics compiled across every unseen validation window
    overall_mae = mean_absolute_error(all_unseen_actuals, all_unseen_predictions)
    overall_rmse = root_mean_squared_error(all_unseen_actuals, all_unseen_predictions)
    overall_r2 = np.corrcoef(all_unseen_predictions, all_unseen_actuals)[0, 1] ** 2
    end_overall = time.perf_counter()
    overall_duration = end_overall - start_overall

    print(f"Overall Unseen MAE:      {overall_mae:.4f} g/cm³")
    print(f"Overall Unseen RMSE:     {overall_rmse:.4f} g/cm³")
    print(f"Overall Unseen Total R²: {overall_r2:.4f}")
    print(f"Total Unseen Molecules:  {len(all_unseen_predictions):,}")
    print(f"Total Execution Time:    {overall_duration:.2f} seconds")
    print("-" * 45)

    # Save final cross-validated validation records to disk for paper backup
    results_df = pd.DataFrame({
        'Actual_Density': all_unseen_actuals,
        'Predicted_Density': all_unseen_predictions,
        'Abs_Error': np.abs(np.array(all_unseen_predictions) - np.array(all_unseen_actuals))
    })
    results_df.to_csv("cross_validation_unseen_results.csv", index=False)
    print("💾 Validation matrix exported cleanly to: cross_validation_unseen_results.csv")


# This is the vital shield protecting the Windows multi-core processes!
if __name__ == '__main__':
    run_cross_validation()
