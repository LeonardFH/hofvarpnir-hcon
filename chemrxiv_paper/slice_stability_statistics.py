# ==============================================================================
# HÓFVARPNIRHCON
#
# Slice Stability Statistical Analysis
#
# Reads:
#     slice_convergence_results.csv
#
# Produces:
#     slice_stability_statistics.csv
#
# Calculates:
#     Mean MAE
#     Median MAE
#     Standard deviation
#     Coefficient of variation
#     Minimum / Maximum
#     Q1 / Q3
#     IQR
#     95% confidence interval
#
# ==============================================================================

import pandas as pd
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE = "slice_convergence_results.csv"

OUTPUT_FILE = "slice_stability_statistics.csv"

DATASET_COLUMN = "Dataset"
METRIC_COLUMN = "MAE"



# ==============================================================================
# LOAD DATA
# ==============================================================================

df = pd.read_csv(INPUT_FILE)


print()
print("Datasets found:")
print(df[DATASET_COLUMN].unique())



# ==============================================================================
# STATISTICS
# ==============================================================================

results = []


for dataset in df[DATASET_COLUMN].unique():

    subset = df[
        df[DATASET_COLUMN] == dataset
    ]


    values = subset[METRIC_COLUMN]


    n = len(values)

    mean = values.mean()

    median = values.median()

    std = values.std(ddof=1)

    cv = (
        std / mean
    ) * 100


    minimum = values.min()

    maximum = values.max()


    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    iqr = q3 - q1


    # 95% confidence interval of mean

    ci95 = (
        1.96
        *
        std
        /
        np.sqrt(n)
    )

    ci_lower = mean - ci95

    ci_upper = mean + ci95



    results.append({

        "Dataset":
            dataset,

        "Training_sets":
            n,

        "Mean_MAE":
            mean,

        "Median_MAE":
            median,

        "Std":
            std,

        "CV_percent":
            cv,

        "Minimum":
            minimum,

        "Q1":
            q1,

        "Q3":
            q3,

        "IQR":
            iqr,

        "Maximum":
            maximum,

        "95CI_Lower":
            ci_lower,

        "95CI_Upper":
            ci_upper

    })



# ==============================================================================
# OUTPUT
# ==============================================================================

stats = pd.DataFrame(results)


stats.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)



print()

print(stats.to_string(index=False))

print()

print("Saved:")
print(OUTPUT_FILE)