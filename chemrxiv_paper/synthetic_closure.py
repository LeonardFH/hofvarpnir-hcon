"""
HÓFVARPNIRHCON

Synthetic Dictionary Closure Experiment

Multiple datasets

Protocol:

Dataset
|
random shuffle
|
split into Group A and Group B
|
+-----------------------------+
|                             |
Train dictionary A             |
using measured densities       |
|                             |
Predict A                      |
baseline                       |
|                             |
Predict B                     |
|                             |
Generate synthetic B'          |
|                             |
Train dictionary B'            |
using synthetic densities      |
|                             |
Predict A and B                |
|                             |
Compare direct vs synthetic    |
pathways                      |
+-----------------------------+

"""

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
    "Mathieu": "mathieu.csv",
    "Jin" : "wuhe.csv",
    "He" : "he_35250.csv"
}


SMILES_COLUMN = "SMILES"
DENSITY_COLUMN = "Density"

SEED = 42

STRATEGY = "auto"


RESULT_FILE = "synthetic_closure_results.csv"



# ==========================================================
# TRAIN DICTIONARY
# ==========================================================

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
# MAIN
# ==========================================================

def main():

    results = []


    for dataset_name, filename in DATASETS.items():


        print("\n")
        print("="*70)
        print(dataset_name)
        print("="*70)


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


        print(
            "Total molecules:",
            len(df)
        )


        # --------------------------------------------------
        # RANDOM SPLIT
        # --------------------------------------------------

        df = df.sample(

            frac=1,

            random_state=SEED

        ).reset_index(drop=True)



        half = len(df)//2


        group_A = df.iloc[:half].copy()

        group_B = df.iloc[half:].copy()



        print(
            "Group A:",
            len(group_A)
        )

        print(
            "Group B:",
            len(group_B)
        )



        # --------------------------------------------------
        # FILE NAMES
        # --------------------------------------------------

        prefix = dataset_name.lower()


        group_A_file = (
            f"{prefix}_group_A.csv"
        )


        group_B_file = (
            f"{prefix}_group_B.csv"
        )


        synthetic_B_file = (
            f"{prefix}_synthetic_B.csv"
        )


        weights_A = (
            f"{prefix}_weights_A.pkl"
        )


        weights_B = (
            f"{prefix}_weights_B.pkl"
        )



        group_A.to_csv(
            group_A_file,
            index=False
        )


        group_B.to_csv(
            group_B_file,
            index=False
        )



        # --------------------------------------------------
        # TRAIN A DICTIONARY
        # --------------------------------------------------

        print(
            "Training dictionary A"
        )


        train_dictionary(

            group_A_file,

            weights_A

        )



        # --------------------------------------------------
        # A -> A
        # --------------------------------------------------

        A_A = evaluate(

            weights_A,

            group_A_file

        )



        # --------------------------------------------------
        # A -> B DIRECT PATH
        # --------------------------------------------------

        prediction_B_direct = predict_density_batch(

            smiles_list=group_B["SMILES"].tolist(),

            weights_path=weights_A,

            verbose=False

        )


        prediction_B_direct = np.asarray(

            prediction_B_direct,

            dtype=float

        )



        truth_B = group_B["Density"].values



        valid = (

            np.isfinite(prediction_B_direct)

            &

            np.isfinite(truth_B)

        )



        direct_mae = mean_absolute_error(

            truth_B[valid],

            prediction_B_direct[valid]

        )


        direct_rmse = root_mean_squared_error(

            truth_B[valid],

            prediction_B_direct[valid]

        )


        direct_r2 = r2_score(

            truth_B[valid],

            prediction_B_direct[valid]

        )



        # --------------------------------------------------
        # CREATE SYNTHETIC B'
        # --------------------------------------------------

        synthetic_B = pd.DataFrame(

            {

                "SMILES":

                    group_B["SMILES"],


                "Density":

                    prediction_B_direct

            }

        )



        synthetic_B.to_csv(

            synthetic_B_file,

            index=False

        )



        # --------------------------------------------------
        # TRAIN RECOVERED DICTIONARY
        # --------------------------------------------------

        print(
            "Training synthetic dictionary B'"
        )


        train_dictionary(

            synthetic_B_file,

            weights_B

        )



        # --------------------------------------------------
        # SYNTHETIC PATH
        # B' dictionary -> B
        # --------------------------------------------------

        prediction_B_synthetic = predict_density_batch(

            smiles_list=group_B["SMILES"].tolist(),

            weights_path=weights_B,

            verbose=False

        )


        prediction_B_synthetic = np.asarray(

            prediction_B_synthetic,

            dtype=float

        )



        valid = (

            np.isfinite(prediction_B_direct)

            &

            np.isfinite(prediction_B_synthetic)

            &

            np.isfinite(truth_B)

        )



        synthetic_mae = mean_absolute_error(

            truth_B[valid],

            prediction_B_synthetic[valid]

        )


        synthetic_rmse = root_mean_squared_error(

            truth_B[valid],

            prediction_B_synthetic[valid]

        )


        synthetic_r2 = r2_score(

            truth_B[valid],

            prediction_B_synthetic[valid]

        )



        # --------------------------------------------------
        # Prediction differences
        # --------------------------------------------------

        prediction_difference = (

            prediction_B_direct

            -

            prediction_B_synthetic

        )


        max_difference = np.max(

            np.abs(

                prediction_difference[valid]

            )

        )


        mean_difference = np.mean(

            np.abs(

                prediction_difference[valid]

            )

        )


        prediction_rmse = root_mean_squared_error(

            prediction_B_direct[valid],

            prediction_B_synthetic[valid]

        )



        # --------------------------------------------------
        # RECOVERED DICTIONARY -> A
        # --------------------------------------------------

        recovered_A = evaluate(

            weights_B,

            group_A_file

        )



        print()

        print(
            "Direct MAE:",
            direct_mae
        )

        print(
            "Synthetic MAE:",
            synthetic_mae
        )

        print(
            "Prediction RMSE:",
            prediction_rmse
        )



        results.append(

            {

                "Dataset":
                    dataset_name,


                "N":
                    len(df),


                "A_to_A_MAE":
                    A_A[0],


                "A_to_B_direct_MAE":
                    direct_mae,


                "B_synthetic_to_B_MAE":
                    synthetic_mae,


                "Recovered_B_to_A_MAE":
                    recovered_A[0],


                "Direct_RMSE":
                    direct_rmse,


                "Synthetic_RMSE":
                    synthetic_rmse,


                "Direct_R2":
                    direct_r2,


                "Synthetic_R2":
                    synthetic_r2,


                "Max_prediction_difference":
                    max_difference,


                "Mean_prediction_difference":
                    mean_difference,


                "Prediction_vector_RMSE":
                    prediction_rmse

            }

        )



    # ======================================================
    # SAVE
    # ======================================================

    results = pd.DataFrame(results)


    results.to_csv(

        RESULT_FILE,

        index=False

    )


    print()

    print("="*70)

    print("FINAL RESULTS")

    print("="*70)

    print(results)


    print()

    print(
        "Saved:",
        RESULT_FILE
    )



if __name__ == "__main__":

    main()