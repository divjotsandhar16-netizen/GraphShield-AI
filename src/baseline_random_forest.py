import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"

MODEL_PATH = "models/random_forest_baseline.pkl"
RESULTS_DIR = "results"

TRAIN_SAMPLE_SIZE = 250_000
VALIDATION_SAMPLE_SIZE = 100_000

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "Source Port",
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

TARGET_COLUMN = "Label"


# =========================================================
# DATA CLEANING
# =========================================================

def clean_data(df):
    """
    Apply the same data-quality rules to the selected rows.

    Removes physically impossible negative measurements and
    converts known -1 sentinel values to NaN.
    """

    # -----------------------------------------------------
    # Remove physically impossible measurements
    # -----------------------------------------------------

    invalid_mask = (
        (df["Fwd Header Length"] < 0)
        | (df["Bwd Header Length"] < 0)
        | (df["min_seg_size_forward"] < 0)
        | (df["Flow Duration"] < 0)
        | (df["Flow Bytes/s"] < 0)
        | (df["Flow Packets/s"] < 0)
    )

    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        print(
            f"Removing {invalid_count:,} "
            "physically invalid rows."
        )

    df = df.loc[~invalid_mask].copy()

    # -----------------------------------------------------
    # Handle sentinel values
    # -----------------------------------------------------

    sentinel_columns = [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]

    for column in sentinel_columns:
        df.loc[df[column] == -1, column] = np.nan

    return df


# =========================================================
# STRATIFIED SAMPLE FUNCTION
# =========================================================

def get_stratified_sample(file_path, sample_size):
    """
    Creates a stratified sample from a CSV file.

    The target is converted to:
        0 = BENIGN
        1 = ATTACK

    The function first reads only the Label column to choose
    row indices, then reads the CSV in chunks to retrieve those
    rows.
    """

    print(f"\nPreparing stratified sample from:")
    print(file_path)

    # -----------------------------------------------------
    # Pass 1: Read labels
    # -----------------------------------------------------

    labels = pd.read_csv(
        file_path,
        usecols=[TARGET_COLUMN],
        low_memory=False,
    )

    binary_labels = (
        labels[TARGET_COLUMN]
        .astype(int)
        .ne(0)
        .astype(int)
    )

    if sample_size > len(binary_labels):
        raise ValueError(
            f"Requested sample size ({sample_size:,}) is larger "
            f"than dataset size ({len(binary_labels):,})."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=RANDOM_STATE,
    )

    selected_indices, _ = next(
        splitter.split(
            np.zeros(len(binary_labels)),
            binary_labels,
        )
    )

    selected_indices = np.sort(selected_indices)

    print(
        f"Selected {len(selected_indices):,} rows."
    )

    del labels
    del binary_labels

    # -----------------------------------------------------
    # Pass 2: Retrieve selected rows
    # -----------------------------------------------------

    selected_rows = []

    selected_indices_set = set(
        int(index) for index in selected_indices
    )

    current_position = 0

    usecols = MODEL_FEATURES + [TARGET_COLUMN]

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=200_000,
        low_memory=False,
    ):

        chunk_start = current_position
        chunk_end = (
            current_position + len(chunk)
        )

        # Find indices belonging to this chunk
        local_indices = [
            index
            for index in selected_indices_set
            if chunk_start <= index < chunk_end
        ]

        if local_indices:

            relative_indices = [
                index - chunk_start
                for index in local_indices
            ]

            selected_chunk = chunk.iloc[
                relative_indices
            ].copy()

            selected_rows.append(
                selected_chunk
            )

        current_position = chunk_end

    if not selected_rows:
        raise RuntimeError(
            "No rows were retrieved during the second "
            "sampling pass."
        )

    result = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    print(
        f"Retrieved {len(result):,} rows."
    )

    return result


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI - RANDOM FOREST BASELINE")
print("=" * 60)

print("\nCreating training sample...")

train_df = get_stratified_sample(
    TRAIN_FILE,
    TRAIN_SAMPLE_SIZE,
)

print("\nTraining sample before cleaning:")
print(train_df.shape)

train_df = clean_data(train_df)

print("Training sample after cleaning:")
print(train_df.shape)


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

print("\nCreating validation sample...")

validation_df = get_stratified_sample(
    VALIDATION_FILE,
    VALIDATION_SAMPLE_SIZE,
)

print("\nValidation sample before cleaning:")
print(validation_df.shape)

validation_df = clean_data(validation_df)

print("Validation sample after cleaning:")
print(validation_df.shape)


# =========================================================
# CREATE BINARY TARGET
# =========================================================

y_train = (
    train_df[TARGET_COLUMN]
    .astype(int)
    .ne(0)
    .astype(int)
)

y_validation = (
    validation_df[TARGET_COLUMN]
    .astype(int)
    .ne(0)
    .astype(int)
)


# =========================================================
# CREATE FEATURE MATRICES
# =========================================================

X_train = train_df[MODEL_FEATURES].copy()
X_validation = validation_df[MODEL_FEATURES].copy()

print("\nTraining class distribution:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nValidation class distribution:")
print(
    y_validation.value_counts()
    .sort_index()
)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("\nFitting median imputer...")

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)

X_validation = imputer.transform(
    X_validation
)


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train,
    y_train,
)

print("\nRandom Forest training completed.")


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

print("\nGenerating validation predictions...")

y_pred = model.predict(X_validation)

y_probability = model.predict_proba(
    X_validation
)[:, 1]


# =========================================================
# METRICS
# =========================================================

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_validation,
    y_probability,
)

pr_auc = average_precision_score(
    y_validation,
    y_probability,
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("RANDOM FOREST BASELINE RESULTS")
print("=" * 60)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        y_pred,
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        digits=4,
        zero_division=0,
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        y_pred,
    )
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame(
    {
        "feature": MODEL_FEATURES,
        "importance": model.feature_importances_,
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False,
)

importance_file = (
    f"{RESULTS_DIR}/random_forest_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False,
)

print("\nTop 20 important features:")

print(
    importance.head(20).to_string(
        index=False
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

model_bundle = {
    "model": model,
    "imputer": imputer,
    "feature_columns": MODEL_FEATURES,
    "task": "binary_intrusion_detection",
    "random_state": RANDOM_STATE,
}

joblib.dump(
    model_bundle,
    MODEL_PATH,
)


# =========================================================
# SAVE RESULTS
# =========================================================

results = {
    "model": "Random Forest",
    "task": "Binary Intrusion Detection",
    "train_samples": int(len(y_train)),
    "validation_samples": int(len(y_validation)),
    "features": int(len(MODEL_FEATURES)),
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
}

results_file = (
    f"{RESULTS_DIR}/random_forest_baseline.json"
)

with open(
    results_file,
    "w",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nFeature importance saved to:")
print(importance_file)

print("\nMetrics saved to:")
print(results_file)