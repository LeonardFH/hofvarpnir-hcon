# Examples

This folder contains ready-to-run scripts demonstrating HófvarpnirHCON.

## OneRun.py
Single training + prediction run on your CSV file.
- Trains bond dictionaries using NNLS
- Predicts all molecules in the dataset
- Reports MAE, RMSE, R², and throughput

## TenFoldRun.py
10-fold cross-validation benchmark.
- Splits data into 10 folds
- Trains and predicts on each fold
- Reports average MAE, RMSE, R² across all folds

## Usage
1. Place your CSV file in the project root
2. Update `CSV_FILE_NAME`, `SMILES_COLUMN_NAME`, `DENSITY_COLUMN_NAME` in the script
3. Run: `python examples/OneRun.py`