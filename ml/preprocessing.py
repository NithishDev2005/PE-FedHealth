import os
import pickle
import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WESAD_DIR = os.path.join(BASE_DIR, "data", "raw", "WESAD")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SAMPLING_RATE = 700
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS


# --------------------------------------------------
# LOAD ONE SUBJECT
# --------------------------------------------------

def load_subject(subject_id):
    file_path = os.path.join(
        WESAD_DIR,
        subject_id,
        f"{subject_id}.pkl"
    )

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    return data


# --------------------------------------------------
# EXTRACT FEATURES FROM ONE WINDOW
# --------------------------------------------------

def extract_features(signal_window):

    return {
        "mean": np.mean(signal_window),
        "std": np.std(signal_window),
        "min": np.min(signal_window),
        "max": np.max(signal_window),
        "median": np.median(signal_window),
    }


# --------------------------------------------------
# PROCESS ONE SUBJECT
# --------------------------------------------------

def process_subject(subject_id):

    print(f"Processing {subject_id}...")

    data = load_subject(subject_id)

    chest = data["signal"]["chest"]
    labels = data["label"]

    # Flatten sensor signals
    ecg = chest["ECG"].flatten()
    eda = chest["EDA"].flatten()
    temp = chest["Temp"].flatten()
    resp = chest["Resp"].flatten()

    # Use the common length
    length = min(
        len(ecg),
        len(eda),
        len(temp),
        len(resp),
        len(labels)
    )

    ecg = ecg[:length]
    eda = eda[:length]
    temp = temp[:length]
    resp = resp[:length]
    labels = labels[:length]

    rows = []

    # Create 10-second windows
    for start in range(0, length - WINDOW_SIZE, WINDOW_SIZE):

        end = start + WINDOW_SIZE

        ecg_window = ecg[start:end]
        eda_window = eda[start:end]
        temp_window = temp[start:end]
        resp_window = resp[start:end]
        label_window = labels[start:end]

        # Determine majority label
        unique_labels, counts = np.unique(
            label_window,
            return_counts=True
        )

        majority_label = unique_labels[np.argmax(counts)]

        # Extract features
        ecg_features = extract_features(ecg_window)
        eda_features = extract_features(eda_window)
        temp_features = extract_features(temp_window)
        resp_features = extract_features(resp_window)

        row = {
            "subject": subject_id,
            "label": int(majority_label),

            "ecg_mean": ecg_features["mean"],
            "ecg_std": ecg_features["std"],
            "ecg_min": ecg_features["min"],
            "ecg_max": ecg_features["max"],

            "eda_mean": eda_features["mean"],
            "eda_std": eda_features["std"],
            "eda_min": eda_features["min"],
            "eda_max": eda_features["max"],

            "temp_mean": temp_features["mean"],
            "temp_std": temp_features["std"],
            "temp_min": temp_features["min"],
            "temp_max": temp_features["max"],

            "resp_mean": resp_features["mean"],
            "resp_std": resp_features["std"],
            "resp_min": resp_features["min"],
            "resp_max": resp_features["max"],
        }

        rows.append(row)

    return rows


# --------------------------------------------------
# PROCESS ALL AVAILABLE SUBJECTS
# --------------------------------------------------

def main():

    all_rows = []

    subjects = [
        folder
        for folder in os.listdir(WESAD_DIR)
        if folder.startswith("S")
        and os.path.isdir(os.path.join(WESAD_DIR, folder))
    ]

    subjects.sort()

    print("Subjects found:")
    print(subjects)

    for subject in subjects:

        try:
            rows = process_subject(subject)
            all_rows.extend(rows)

        except Exception as error:
            print(f"Could not process {subject}: {error}")

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Save processed data
    output_file = os.path.join(
        PROCESSED_DIR,
        "wesad_features.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nProcessing completed!")
    print("Dataset shape:", df.shape)
    print("Saved to:", output_file)

    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    main()