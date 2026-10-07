import os
import pandas as pd
import numpy as np


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "wesad_features.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_features.csv"
)


def create_features():

    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # -----------------------------------------
    # Additional physiological features
    # -----------------------------------------

    # ECG variability
    df["ecg_range"] = df["ecg_max"] - df["ecg_min"]

    # EDA variability
    df["eda_range"] = df["eda_max"] - df["eda_min"]

    # Temperature variation
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    # Respiration variation
    df["resp_range"] = df["resp_max"] - df["resp_min"]

    # Ratios / combined features
    df["eda_temp_ratio"] = (
        df["eda_mean"] / (np.abs(df["temp_mean"]) + 1e-8)
    )

    # -----------------------------------------
    # Remove invalid values
    # -----------------------------------------

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    # -----------------------------------------
    # Save final dataset
    # -----------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature engineering completed!")
    print("Final shape:", df.shape)
    print("Saved to:", OUTPUT_FILE)

    print("\nFinal columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    create_features()