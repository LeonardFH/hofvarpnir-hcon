"""
HÓFVARPNIRHCON — MULTI-DATASET LEARNING CURVE CONVERGENCE

Experiment:
Determine how quickly the bond-overlap dictionary reaches a stable
prediction regime as training size increases.

Protocol:
- Random shuffle dataset
- Train on increasing subsets
- Predict the complete dataset
- Repeat 5 times with different random seeds
- Calculate mean ± std performance

Datasets:
Taniguchi : large-scale convergence
Davis     : primary benchmark
Jin       : primary benchmark
Taylor    : medium-scale validation
Mathieu   : excluded (small dataset)

"""

from hofvarpnirhcon import train_density, predict_density_batch

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

import os
import time


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",   
    "Davis": "davis.csv",           
    "Taylor": "t1000.csv",          
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}


# ============================================================
# CONFIGURATION
# ============================================================


SMILES_COLUMN_NAME = "SMILES"

DENSITY_COLUMN_NAME = "Density"


SEEDS = [
    42,
    123,
    456,
    789,
    999
]


TRAINING_SIZES = {

    "Taniguchi": [
        500,
        1000,
        2500,
        5000,
        10000,
        25000,
        50000,
        100000
    ],

    "Davis": [
        500,
        1000,
        2500,
        5000,
        10000
    ],
    
    "Jin": [
        500,
        1000,
        2500,
        5000,
        10000
    ],
    
    "He": [
        500,
        1000,
        2500,
        5000,
        10000
    ],


    "Taylor": [
        100,
        250,
        500,
        750
    ]
}


RAW_OUTPUT = "learning_curve_raw_results.csv"

SUMMARY_OUTPUT = "learning_curve_summary.csv"



# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset(path):

    df = pd.read_csv(path)


    df = df[
        [
            SMILES_COLUMN_NAME,
            DENSITY_COLUMN_NAME
        ]
    ].copy()


    df.columns = [
        "SMILES",
        "Density"
    ]


    df = df.dropna()


    return df



# ============================================================
# TRAIN + PREDICT
# ============================================================


def run_single_training(
        train_df,
        full_df,
        tag
):


    train_file = f"{tag}_train.csv"

    weight_file = f"{tag}_weights.pkl"


    train_df.to_csv(
        train_file,
        index=False
    )


    start_train = time.perf_counter()


    train_density(
        data_path=train_file,
        output_path=weight_file,
        filter_cocrystals=True,
        filter_hcon=False,
        verbose=False
    )


    train_time = (
        time.perf_counter()
        -
        start_train
    )


    start_prediction = time.perf_counter()


    predictions = predict_density_batch(
        smiles_list=full_df["SMILES"].tolist(),
        weights_path=weight_file,
        verbose=False
    )


    prediction_time = (
        time.perf_counter()
        -
        start_prediction
    )


    actual = full_df["Density"].values


    mask = [
        p is not None
        for p in predictions
    ]


    pred = np.array(
        [
            p
            for p in predictions
            if p is not None
        ]
    )


    actual = actual[mask]


    mae = mean_absolute_error(
        actual,
        pred
    )


    rmse = root_mean_squared_error(
        actual,
        pred
    )


    r2 = np.corrcoef(
        pred,
        actual
    )[0, 1] ** 2



    # cleanup

    if os.path.exists(train_file):
        os.remove(train_file)


    if os.path.exists(weight_file):
        os.remove(weight_file)



    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Train_Time": train_time,

        "Prediction_Time": prediction_time,

        "Valid_Count": len(pred)
    }



# ============================================================
# MAIN EXPERIMENT
# ============================================================


def run_learning_curve():


    all_results = []


    for dataset_name, filename in DATASETS.items():


        print("\n")
        print("=" * 80)
        print(dataset_name)
        print("=" * 80)


        df = load_dataset(filename)


        total_size = len(df)


        print(
            f"Dataset size: {total_size:,}"
        )


        sizes = TRAINING_SIZES[dataset_name].copy()


        # Always include full dataset

        if total_size not in sizes:

            sizes.append(total_size)



        sizes = sorted(
            [
                s
                for s in sizes
                if s <= total_size
            ]
        )


        for seed in SEEDS:


            print(
                f"\nSeed {seed}"
            )


            shuffled = (
                df
                .sample(
                    frac=1,
                    random_state=seed
                )
                .reset_index(drop=True)
            )



            for size in sizes:


                print(
                    f"{dataset_name} "
                    f"| seed={seed} "
                    f"| training={size:,}"
                )


                train_df = (
                    shuffled
                    .iloc[:size]
                    .copy()
                )


                tag = (
                    f"{dataset_name}_"
                    f"{seed}_"
                    f"{size}"
                )


                metrics = run_single_training(
                    train_df,
                    df,
                    tag
                )


                all_results.append(
                    {

                    "Dataset": dataset_name,

                    "Seed": seed,

                    "Training_Size": size,

                    **metrics

                    }
                )


                print(
                    f"MAE={metrics['MAE']:.5f}"
                )



    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================


    results_df = pd.DataFrame(
        all_results
    )


    results_df.to_csv(
        RAW_OUTPUT,
        index=False
    )



    # ========================================================
    # SUMMARY
    # ========================================================


    summary = (
        results_df
        .groupby(
            [
                "Dataset",
                "Training_Size"
            ]
        )
        .agg(
            {

            "MAE":
                ["mean","std"],

            "RMSE":
                ["mean","std"],

            "R2":
                ["mean","std"]

            }
        )
    )


    summary.columns = [
        "_".join(col)
        for col in summary.columns
    ]


    summary = (
        summary
        .reset_index()
    )


    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )



    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(summary.to_string(index=False))


    print("\nSaved:")
    print(RAW_OUTPUT)
    print(SUMMARY_OUTPUT)



# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    run_learning_curve()