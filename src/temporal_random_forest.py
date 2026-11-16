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

FILES = [
    "data/processed/train.csv",
    "data/processed/validation.csv",
    "data/processed/test.csv",
]

TRAIN_DATES = [
    "2017-07-03",
    "2017-07-04",
    "2017-07-05",
]

VALIDATION_DATES = [
    "2017-07-06",
]

TEST_DATES = [
    "2017-07-07",
]

TRAIN_SAMPLE_SIZE = 300_000
VALIDATION_SAMPLE_SIZE = 100_000
TEST_SAMPLE_SIZE = 200_000

RANDOM_STATE = 42

MODEL_PATH = (
    "models/random_forest_temporal_baseline.pkl"
)

RESULTS_PATH = (
    "results/random_forest_temporal_baseline.json"
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


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
# LABEL INFORMATION
# =========================================================

LABEL_NAMES = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS",
}


# =========================================================
# DATE PARSING
# =========================================================

def parse_dates(series):
    """
    Parse CIC-IDS2017 timestamps.

    The dataset uses day/month/year ordering.
    """

    return pd.to_datetime(
        series,
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )


# =========================================================
# BUILD METADATA
# =========================================================

def build_metadata():

    print("\n" + "=" * 60)
    print("BUILDING TEMPORAL METADATA")
    print("=" * 60)

    metadata_parts = []

    for file_path in FILES:

        print(f"\nReading metadata from:")
        print(file_path)

        metadata = pd.read_csv(
            file_path,
            usecols=[
                "Timestamp",
                "Label",
            ],
            low_memory=False,
        )

        metadata["Date"] = (
            parse_dates(
                metadata["Timestamp"]
            )
            .dt.strftime("%Y-%m-%d")
        )

        metadata["BinaryLabel"] = (
            metadata["Label"]
            .astype(int)
            .ne(0)
            .astype(int)
        )

        metadata["File"] = file_path

        metadata["RowIndex"] = np.arange(
            len(metadata)
        )

        metadata_parts.append(
            metadata[
                [
                    "Date",
                    "Label",
                    "BinaryLabel",
                    "File",
                    "RowIndex",
                ]
            ]
        )

        print(
            f"Rows: {len(metadata):,}"
        )

    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    print(
        f"\nTotal metadata rows: "
        f"{len(metadata):,}"
    )

    return metadata


# =========================================================
# STRATIFIED TEMPORAL SAMPLING
# =========================================================

def select_temporal_sample(
    metadata,
    dates,
    sample_size,
    split_name,
):

    print("\n" + "-" * 60)
    print(f"SELECTING {split_name.upper()} SAMPLE")
    print("-" * 60)

    subset = metadata[
        metadata["Date"].isin(dates)
    ].copy()

    print(
        f"Available rows for {split_name}: "
        f"{len(subset):,}"
    )

    print("\nAttack distribution:")

    print(
        subset["Label"]
        .value_counts()
        .sort_index()
    )

    if sample_size > len(subset):

        raise ValueError(
            f"{split_name} sample size "
            f"({sample_size:,}) is larger than "
            f"available rows ({len(subset):,})."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=RANDOM_STATE,
    )

    selected_positions, _ = next(
        splitter.split(
            np.zeros(len(subset)),
            subset["BinaryLabel"],
        )
    )

    selected = subset.iloc[
        selected_positions
    ].copy()

    print(
        f"\nSelected rows: "
        f"{len(selected):,}"
    )

    print("\nSelected binary distribution:")

    print(
        selected["BinaryLabel"]
        .value_counts()
        .sort_index()
    )

    return selected


# =========================================================
# RETRIEVE SELECTED FEATURE ROWS
# =========================================================

def retrieve_rows(
    selection,
):

    print("\n" + "-" * 60)
    print("RETRIEVING FEATURE DATA")
    print("-" * 60)

    all_parts = []

    for file_path in FILES:

        file_selection = selection[
            selection["File"] == file_path
        ]

        if len(file_selection) == 0:
            continue

        requested_indices = set(
            file_selection["RowIndex"]
            .astype(int)
            .tolist()
        )

        print(
            f"\n{os.path.basename(file_path)}"
        )

        print(
            f"Requested rows: "
            f"{len(requested_indices):,}"
        )

        chunks = []

        current_position = 0

        usecols = MODEL_FEATURES + [
            TARGET_COLUMN
        ]

        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            chunksize=200_000,
            low_memory=False,
        ):

            chunk_start = current_position

            chunk_end = (
                current_position
                + len(chunk)
            )

            local_indices = [
                index
                for index in requested_indices
                if chunk_start <= index < chunk_end
            ]

            if local_indices:

                relative_indices = [
                    index - chunk_start
                    for index in local_indices
                ]

                selected_chunk = (
                    chunk.iloc[
                        relative_indices
                    ]
                    .copy()
                )

                chunks.append(
                    selected_chunk
                )

            current_position = chunk_end

        if chunks:

            file_result = pd.concat(
                chunks,
                ignore_index=True,
            )

            all_parts.append(
                file_result
            )

            print(
                f"Retrieved: "
                f"{len(file_result):,}"
            )

    if not all_parts:

        raise RuntimeError(
            "No rows were retrieved."
        )

    result = pd.concat(
        all_parts,
        ignore_index=True,
    )

    print(
        f"\nTotal retrieved rows: "
        f"{len(result):,}"
    )

    return result


# =========================================================
# CLEAN SELECTED DATA
# =========================================================

def clean_data(df):

    print("\nCleaning selected data...")

    invalid_mask = (
        (df["Fwd Header Length"] < 0)
        | (df["Bwd Header Length"] < 0)
        | (df["min_seg_size_forward"] < 0)
        | (df["Flow Duration"] < 0)
        | (df["Flow Bytes/s"] < 0)
        | (df["Flow Packets/s"] < 0)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count:

        print(
            f"Removing {invalid_count:,} "
            "physically invalid rows."
        )

        df = df.loc[
            ~invalid_mask
        ].copy()

    sentinel_columns = [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]

    for column in sentinel_columns:

        df.loc[
            df[column] == -1,
            column
        ] = np.nan

    return df


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_features(df):

    y = (
        df[TARGET_COLUMN]
        .astype(int)
        .ne(0)
        .astype(int)
    )

    X = df[
        MODEL_FEATURES
    ].copy()

    return X, y


# =========================================================
# MAIN
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI")
print("TEMPORAL / UNSEEN-ATTACK RANDOM FOREST")
print("=" * 60)

print("\nExperimental protocol:")

print(
    "TRAIN      : July 3, 4, 5, 2017"
)

print(
    "VALIDATION : July 6, 2017"
)

print(
    "TEST       : July 7, 2017"
)


# =========================================================
# METADATA
# =========================================================

metadata = build_metadata()


# =========================================================
# SELECT TRAIN / VALIDATION / TEST
# =========================================================

train_selection = select_temporal_sample(
    metadata,
    TRAIN_DATES,
    TRAIN_SAMPLE_SIZE,
    "training",
)

validation_selection = select_temporal_sample(
    metadata,
    VALIDATION_DATES,
    VALIDATION_SAMPLE_SIZE,
    "validation",
)

test_selection = select_temporal_sample(
    metadata,
    TEST_DATES,
    TEST_SAMPLE_SIZE,
    "test",
)


# =========================================================
# RETRIEVE DATA
# =========================================================

train_df = retrieve_rows(
    train_selection
)

validation_df = retrieve_rows(
    validation_selection
)

test_df = retrieve_rows(
    test_selection
)


# =========================================================
# CLEAN DATA
# =========================================================

train_df = clean_data(
    train_df
)

validation_df = clean_data(
    validation_df
)

test_df = clean_data(
    test_df
)


# =========================================================
# PREPARE MATRICES
# =========================================================

X_train, y_train = prepare_features(
    train_df
)

X_validation, y_validation = (
    prepare_features(
        validation_df
    )
)

X_test, y_test = prepare_features(
    test_df
)


print("\n" + "=" * 60)
print("FINAL TEMPORAL SAMPLE SIZES")
print("=" * 60)

print(
    f"\nTraining    : {len(y_train):,}"
)

print(
    f"Validation  : {len(y_validation):,}"
)

print(
    f"Test        : {len(y_test):,}"
)


print("\nTraining attack distribution:")

print(
    y_train.value_counts()
    .sort_index()
)


print("\nValidation attack distribution:")

print(
    y_validation.value_counts()
    .sort_index()
)


print("\nTest attack distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# =========================================================
# IMPUTATION
# =========================================================

print("\nFitting median imputer on TRAIN only...")

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_validation = imputer.transform(
    X_validation
)

X_test = imputer.transform(
    X_test
)


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

print("\n" + "=" * 60)
print("TRAINING TEMPORAL RANDOM FOREST")
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

print(
    "\nTraining completed."
)


# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_pred = model.predict(
    X_validation
)

validation_probability = (
    model.predict_proba(
        X_validation
    )[:, 1]
)

validation_precision = precision_score(
    y_validation,
    validation_pred,
    zero_division=0,
)

validation_recall = recall_score(
    y_validation,
    validation_pred,
    zero_division=0,
)

validation_f1 = f1_score(
    y_validation,
    validation_pred,
    zero_division=0,
)

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_probability,
)

validation_pr_auc = average_precision_score(
    y_validation,
    validation_probability,
)

print(
    f"\nPrecision : {validation_precision:.4f}"
)

print(
    f"Recall    : {validation_recall:.4f}"
)

print(
    f"F1-score  : {validation_f1:.4f}"
)

print(
    f"ROC-AUC   : {validation_roc_auc:.4f}"
)

print(
    f"PR-AUC    : {validation_pr_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_pred,
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_pred,
    )
)


# =========================================================
# FINAL TEST
# =========================================================

print("\n" + "=" * 60)
print("FINAL TEST - JULY 7")
print("=" * 60)

test_pred = model.predict(
    X_test
)

test_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability,
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability,
)

print(
    f"\nPrecision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_pred,
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

model_bundle = {
    "model": model,
    "imputer": imputer,
    "feature_columns": MODEL_FEATURES,
    "task": "temporal_binary_intrusion_detection",
    "train_dates": TRAIN_DATES,
    "validation_dates": VALIDATION_DATES,
    "test_dates": TEST_DATES,
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
    "protocol": "Temporal / Unseen-Attack Evaluation",
    "train_dates": TRAIN_DATES,
    "validation_dates": VALIDATION_DATES,
    "test_dates": TEST_DATES,
    "train_samples": int(len(y_train)),
    "validation_samples": int(len(y_validation)),
    "test_samples": int(len(y_test)),
    "features": len(MODEL_FEATURES),
    "random_state": RANDOM_STATE,
    "validation": {
        "precision": float(validation_precision),
        "recall": float(validation_recall),
        "f1_score": float(validation_f1),
        "roc_auc": float(validation_roc_auc),
        "pr_auc": float(validation_pr_auc),
    },
    "test": {
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1),
        "roc_auc": float(test_roc_auc),
        "pr_auc": float(test_pr_auc),
    },
}

with open(
    RESULTS_PATH,
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
print("TEMPORAL BASELINE COMPLETE")
print("=" * 60)

print(
    "\nModel saved to:"
)

print(MODEL_PATH)

print(
    "\nResults saved to:"
)

print(RESULTS_PATH)