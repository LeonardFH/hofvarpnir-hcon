# ==========================================================
# CROSS DICTIONARY STABILITY EXPERIMENT
#
# Multiple datasets
# Multiple random seeds
#
# Outputs:
#   Five MAEs
#   Mean MAE
#   MAE spread
#   Relative spread %
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from hofvarpnirhcon import (
    train_density,
    predict_density_batch
)


# ==========================================================
# CONFIG
# ==========================================================

DATASETS = {
    "Taniguchi": "takuyhaa.csv",
    "Davis": "davis.csv",
    "Taylor": "t1000.csv",
    #"Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}


SEEDS = [
    42,
    123,
    456,
    789,
    999
]


STRATEGY = "auto"


RESULT_FILE = "cross_dictionary_stability_results.csv"


# ==========================================================
# TRAIN DICTIONARY
# ==========================================================

def main():
    def train_dictionary(datafile, weightfile):

        train_density(

            data_path=datafile,

            output_path=weightfile,

            strategy=STRATEGY,

            filter_hcon=False,

            filter_cocrystals=False,

            verbose=False
        )


    # ==========================================================
    # EVALUATION
    # ==========================================================

    def evaluate(weights, dataset):

        df = pd.read_csv(dataset)

        prediction = predict_density_batch(

            smiles_list=df["SMILES"].tolist(),

            weights_path=weights,

            verbose=False

        )

        prediction = np.asarray(
            prediction,
            dtype=float
        )

        truth = df["Density"].values


        valid = (
            np.isfinite(prediction)
            &
            np.isfinite(truth)
        )


        mae = mean_absolute_error(
            truth[valid],
            prediction[valid]
        )


        rmse = root_mean_squared_error(
            truth[valid],
            prediction[valid]
        )


        r2 = r2_score(
            truth[valid],
            prediction[valid]
        )


        return mae, rmse, r2



    # ==========================================================
    # MAIN EXPERIMENT
    # ==========================================================

    results = []


    for dataset_name, filename in DATASETS.items():

        print()
        print("="*80)
        print(dataset_name)
        print("="*80)


        df = pd.read_csv(filename)

        df = df[
            [
                "SMILES",
                "Density"
            ]
        ].dropna()


        for seed in SEEDS:


            print()
            print("Seed", seed)


            shuffled = df.sample(
                frac=1,
                random_state=seed
            ).reset_index(drop=True)


            half = len(shuffled)//2


            group_A = shuffled.iloc[:half].copy()

            group_B = shuffled.iloc[half:].copy()


            group_A_file = "_group_A.csv"

            group_B_file = "_group_B.csv"


            weight_A = "_weights_A.pkl"

            weight_B = "_weights_B.pkl"


            group_A.to_csv(
                group_A_file,
                index=False
            )

            group_B.to_csv(
                group_B_file,
                index=False
            )


            print("Training A")

            train_dictionary(
                group_A_file,
                weight_A
            )


            print("Training B")

            train_dictionary(
                group_B_file,
                weight_B
            )


            # --------------------------------------------------
            # Four evaluations
            # --------------------------------------------------

            A_A = evaluate(
                weight_A,
                group_A_file
            )

            A_B = evaluate(
                weight_A,
                group_B_file
            )

            B_B = evaluate(
                weight_B,
                group_B_file
            )

            B_A = evaluate(
                weight_B,
                group_A_file
            )


            mae_values = np.array(
                [
                    A_A[0],
                    A_B[0],
                    B_B[0],
                    B_A[0]
                ]
            )


            mean_mae = np.mean(
                mae_values
            )


            spread = (
                np.max(mae_values)
                -
                np.min(mae_values)
            )


            spread_percent = (
                spread
                /
                mean_mae
                *
                100
            )


            print(
                f"Mean MAE      : {mean_mae:.6f}"
            )

            print(
                f"MAE spread    : {spread:.6f}"
            )

            print(
                f"Spread (%)    : {spread_percent:.3f}%"
            )


            results.append({

                "Dataset": dataset_name,

                "Seed": seed,

                "A_to_A_MAE": A_A[0],

                "A_to_B_MAE": A_B[0],

                "B_to_B_MAE": B_B[0],

                "B_to_A_MAE": B_A[0],

                "Mean_MAE": mean_mae,

                "MAE_Spread": spread,

                "Relative_Spread_%": spread_percent

            })



    # ==========================================================
    # SAVE RESULTS + SEED-TO-SEED STABILITY
    # ==========================================================

    results = pd.DataFrame(results)


    # ----------------------------------------------------------
    # Calculate variability of Mean_MAE across random seeds
    # ----------------------------------------------------------

    seed_summary = []

    for dataset_name in DATASETS.keys():

        subset = results[
            results["Dataset"] == dataset_name
        ]

        seed_mean_values = subset[
            "Mean_MAE"
        ].values


        seed_summary.append({

            "Dataset": dataset_name,

            "Mean_MAE_over_seeds":
                np.mean(seed_mean_values),

            "Seed_to_seed_range":
                np.max(seed_mean_values)
                -
                np.min(seed_mean_values),

            "Seed_to_seed_std":
                np.std(
                    seed_mean_values,
                    ddof=1
                )

        })


    seed_summary = pd.DataFrame(seed_summary)



    # ----------------------------------------------------------
    # Save detailed results
    # ----------------------------------------------------------

    results.to_csv(
        RESULT_FILE,
        index=False
    )



    # ----------------------------------------------------------
    # Save summary table
    # ----------------------------------------------------------

    SUMMARY_FILE = (
        "cross_dictionary_seed_summary.csv"
    )


    seed_summary.to_csv(
        SUMMARY_FILE,
        index=False
    )



    print()
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)

    print(results)


    print()
    print("="*80)
    print("SEED-TO-SEED SUMMARY")
    print("="*80)

    print(seed_summary)


    print()
    print(
    "Saved:",
    RESULT_FILE
    )

    print(
    "Saved:",
    SUMMARY_FILE
    )

if __name__ == "__main__":
    main()