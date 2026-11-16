import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# GRAPHSHIELD AI
# FULL TEMPORAL RANDOM FOREST BASELINE
# ============================================================

SPLIT_FILES = [
    "data/processed/train.csv",
    "data/processed/validation.csv",
    "data/processed/test.csv",
]

PREPROCESSOR_FILE = (
    "data/processed/graphshield_preprocessor.joblib"
)

MODEL_FILE = (
    "models/temporal_random_forest_full.pkl"
)

VALIDATION_RESULTS_FILE = (
    "results/temporal_random_forest_full_validation.json"
)

TEST_RESULTS_FILE = (
    "results/temporal_random_forest_full_test.json"
)


TRAIN_DATES = {
    "2017-07-03",
    "2017-07-04",
    "2017-07-05",
}

VALIDATION_DATES = {
    "2017-07-06",
}

TEST_DATES = {
    "2017-07-07",
}


SEED = 42
CHUNK_SIZE = 100_000


print("=" * 60)
print("GRAPHSHIELD AI - FULL TEMPORAL RANDOM FOREST")
print("=" * 60)


# ============================================================
# 1. DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 2. LOAD PREPROCESSOR
# ============================================================

print(
    "\nLoading training-only preprocessing metadata..."
)


preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


feature_columns = (
    preprocessor["feature_columns"]
)

imputation_values = (
    preprocessor["imputation_values"]
)


imputation_series = pd.Series(
    imputation_values,
    dtype=np.float64
)


if len(feature_columns) != 66:

    raise ValueError(
        f"Expected 66 features, "
        f"found {len(feature_columns)}"
    )


print(
    "66 model features confirmed."
)


# ============================================================
# 3. CLEANING POLICY
# ============================================================

non_negative_features = [
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

    "Fwd Header Length",
    "Bwd Header Length",

    "Fwd Packets/s",
    "Bwd Packets/s",

    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",

    "Average Packet Size",

    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",

    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
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


sentinel_minus_one_features = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


def clean_features(chunk):

    chunk = chunk.copy()


    for column in feature_columns:

        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )


    chunk[feature_columns] = (
        chunk[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    for column in sentinel_minus_one_features:

        chunk.loc[
            chunk[column] == -1,
            column
        ] = np.nan


    for column in non_negative_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] < 0,
                column
            ] = np.nan


    chunk[feature_columns] = (
        chunk[feature_columns]
        .fillna(imputation_series)
    )


    return chunk


# ============================================================
# 4. LOAD ALL TEMPORAL TRAINING DATA
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "LOADING ALL JULY 3-5 TRAINING FLOWS"
)

print(
    "-" * 60
)


X_parts = []
y_parts = []

total_training_rows = 0


use_columns = [
    "Timestamp",
    "Label",
] + feature_columns


for split_file in SPLIT_FILES:

    print(
        f"\nReading: {split_file}"
    )


    for chunk in pd.read_csv(
        split_file,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        # ----------------------------------------------------
        # Parse timestamps
        # ----------------------------------------------------

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        # ----------------------------------------------------
        # July 3-5 only
        # ----------------------------------------------------

        mask = (
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(TRAIN_DATES)
        )


        chunk = chunk[
            mask
        ].copy()


        if len(chunk) == 0:

            continue


        # ----------------------------------------------------
        # Clean features
        # ----------------------------------------------------

        chunk = clean_features(
            chunk
        )


        # ----------------------------------------------------
        # Binary labels
        # ----------------------------------------------------

        y = (
            chunk["Label"]
            .astype(int)
            >
            0
        ).astype(
            np.int8
        ).to_numpy()


        X = chunk[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )


        X_parts.append(
            X
        )

        y_parts.append(
            y
        )


        total_training_rows += len(
            chunk
        )


        print(
            f"Training rows loaded: "
            f"{total_training_rows:,}"
        )


# ============================================================
# 5. COMBINE TRAINING DATA
# ============================================================

if not X_parts:

    raise RuntimeError(
        "No temporal training data found."
    )


X_train = np.concatenate(
    X_parts,
    axis=0
)


y_train = np.concatenate(
    y_parts,
    axis=0
)


# Free temporary lists

del X_parts
del y_parts


print(
    "\nFinal training matrix:"
)

print(
    "X:",
    X_train.shape
)

print(
    "y:",
    y_train.shape
)


if len(X_train) != 1_668_475:

    raise ValueError(
        f"Expected 1,668,475 training rows, "
        f"found {len(X_train)}"
    )


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

unique, counts = np.unique(
    y_train,
    return_counts=True
)


print(
    "\nTraining class distribution:"
)


for label, count in zip(
    unique,
    counts
):

    name = (
        "BENIGN"
        if label == 0
        else
        "ATTACK"
    )


    print(
        f"  {name}: {count:,}"
    )


# ============================================================
# 7. TRAIN FULL RANDOM FOREST
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TRAINING FULL RANDOM FOREST"
)

print(
    "-" * 60
)


model = RandomForestClassifier(

    n_estimators=100,

    max_depth=20,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced_subsample",

    n_jobs=-1,

    random_state=SEED,
)


print(
    "\nConfiguration:"
)

print(
    "  Training rows:",
    f"{len(X_train):,}"
)

print(
    "  Trees: 100"
)

print(
    "  Max depth: 20"
)

print(
    "  Min samples leaf: 2"
)

print(
    "  Max features: sqrt"
)

print(
    "  Class weight: balanced_subsample"
)


model.fit(
    X_train,
    y_train
)


print(
    "\nFull Random Forest training complete."
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


print(
    "Saved:"
)

print(
    MODEL_FILE
)


# ============================================================
# 9. VALIDATION INFERENCE
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "VALIDATION INFERENCE - JULY 6"
)

print(
    "-" * 60
)


all_val_labels = []
all_val_probabilities = []

total_val_rows = 0


for split_file in SPLIT_FILES:

    print(
        f"\nReading: {split_file}"
    )


    for chunk in pd.read_csv(
        split_file,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        mask = (
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(VALIDATION_DATES)
        )


        chunk = chunk[
            mask
        ].copy()


        if len(chunk) == 0:

            continue


        chunk = clean_features(
            chunk
        )


        y = (
            chunk["Label"]
            .astype(int)
            >
            0
        ).astype(
            np.int8
        ).to_numpy()


        X = chunk[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )


        probabilities = model.predict_proba(
            X
        )[:, 1]


        all_val_labels.append(
            y
        )

        all_val_probabilities.append(
            probabilities
        )


        total_val_rows += len(y)


        print(
            f"Validation rows processed: "
            f"{total_val_rows:,}"
        )


y_val = np.concatenate(
    all_val_labels
)


val_probabilities = np.concatenate(
    all_val_probabilities
)


print(
    "\nValidation rows:",
    f"{len(y_val):,}"
)


# ============================================================
# 10. VALIDATION AUC METRICS
# ============================================================

val_roc_auc = roc_auc_score(
    y_val,
    val_probabilities
)


val_pr_auc = average_precision_score(
    y_val,
    val_probabilities
)


print(
    "\nValidation ROC-AUC:",
    f"{val_roc_auc:.4f}"
)


print(
    "Validation PR-AUC:",
    f"{val_pr_auc:.4f}"
)


# ============================================================
# 11. VALIDATION THRESHOLD SELECTION
# ============================================================

thresholds = np.linspace(
    0.01,
    0.99,
    197
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        val_probabilities
        >=
        threshold
    ).astype(
        np.int8
    )


    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )


    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()


    threshold_results.append(
        {
            "threshold":
                float(threshold),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),
        }
    )


best_result = max(
    threshold_results,
    key=lambda item: (
        item["f1"],
        item["precision"]
    )
)


THRESHOLD = (
    best_result["threshold"]
)


print(
    "\n" + "-" * 60
)

print(
    "SELECTED VALIDATION THRESHOLD"
)

print(
    "-" * 60
)


print(
    "Threshold:",
    f"{THRESHOLD:.3f}"
)

print(
    "Precision:",
    f"{best_result['precision']:.4f}"
)

print(
    "Recall:",
    f"{best_result['recall']:.4f}"
)

print(
    "F1:",
    f"{best_result['f1']:.4f}"
)


# ============================================================
# 12. SAVE VALIDATION RESULTS
# ============================================================

validation_output = {

    "experiment":
        "Full temporal Random Forest",

    "train_dates":
        sorted(TRAIN_DATES),

    "validation_dates":
        sorted(VALIDATION_DATES),

    "training_rows":
        int(len(X_train)),

    "validation_rows":
        int(len(y_val)),

    "roc_auc":
        float(val_roc_auc),

    "pr_auc":
        float(val_pr_auc),

    "selected_threshold":
        float(THRESHOLD),

    "selected_precision":
        float(
            best_result["precision"]
        ),

    "selected_recall":
        float(
            best_result["recall"]
        ),

    "selected_f1":
        float(
            best_result["f1"]
        ),

    "threshold_selection_metric":
        "F1",

    "test_used":
        False,
}


with open(
    VALIDATION_RESULTS_FILE,
    "w"
) as f:

    json.dump(
        validation_output,
        f,
        indent=4
    )


print(
    "\nSaved validation results:"
)

print(
    VALIDATION_RESULTS_FILE
)


# ============================================================
# 13. FREE MEMORY
# ============================================================

del X_train
del y_train


# ============================================================
# 14. FROZEN TEST EVALUATION
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "FROZEN JULY 7 TEST EVALUATION"
)

print(
    "-" * 60
)

print(
    "Frozen threshold:",
    f"{THRESHOLD:.3f}"
)


all_original_labels = []
all_binary_labels = []
all_test_probabilities = []

total_test_rows = 0


for split_file in SPLIT_FILES:

    print(
        f"\nReading: {split_file}"
    )


    for chunk in pd.read_csv(
        split_file,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        mask = (
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(TEST_DATES)
        )


        chunk = chunk[
            mask
        ].copy()


        if len(chunk) == 0:

            continue


        chunk = clean_features(
            chunk
        )


        original_labels = (
            chunk["Label"]
            .astype(int)
            .to_numpy()
        )


        binary_labels = (
            original_labels > 0
        ).astype(
            np.int8
        )


        X = chunk[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )


        probabilities = model.predict_proba(
            X
        )[:, 1]


        all_original_labels.append(
            original_labels
        )

        all_binary_labels.append(
            binary_labels
        )

        all_test_probabilities.append(
            probabilities
        )


        total_test_rows += len(
            chunk
        )


        print(
            f"Test rows processed: "
            f"{total_test_rows:,}"
        )


# ============================================================
# 15. COMBINE TEST RESULTS
# ============================================================

y_original = np.concatenate(
    all_original_labels
)

y_test = np.concatenate(
    all_binary_labels
)

test_probabilities = np.concatenate(
    all_test_probabilities
)


test_predictions = (
    test_probabilities
    >=
    THRESHOLD
).astype(
    np.int8
)


# ============================================================
# 16. FINAL TEST METRICS
# ============================================================

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)


pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()


specificity = (
    tn /
    (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "FULL-DATA RF - JULY 7 TEST RESULTS"
)

print(
    "-" * 60
)


print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall:",
    f"{recall:.4f}"
)

print(
    "F1:",
    f"{f1:.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
)

print(
    "Specificity:",
    f"{specificity:.4f}"
)


print(
    "\nConfusion matrix:"
)

print(
    "TN:",
    f"{tn:,}"
)

print(
    "FP:",
    f"{fp:,}"
)

print(
    "FN:",
    f"{fn:,}"
)

print(
    "TP:",
    f"{tp:,}"
)


# ============================================================
# 17. PER-ATTACK-FAMILY RESULTS
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "ATTACK-FAMILY RESULTS"
)

print(
    "-" * 60
)


label_names = {
    1: "Bot",
    2: "DDoS",
    10: "PortScan",
}


per_class_results = {}


for label in sorted(
    np.unique(
        y_original
    )
):

    label = int(label)


    if label == 0:
        continue


    mask = (
        y_original
        ==
        label
    )


    flows = int(
        mask.sum()
    )


    detected = int(
        test_predictions[
            mask
        ].sum()
    )


    class_recall = (
        detected / flows
        if flows > 0
        else 0.0
    )


    name = label_names.get(
        label,
        f"Class {label}"
    )


    per_class_results[
        str(label)
    ] = {

        "name": name,

        "flows": flows,

        "detected": detected,

        "missed":
            flows - detected,

        "recall":
            float(class_recall),
    }


    print(
        f"{name:20s} | "
        f"Flows: {flows:7,} | "
        f"Detected: {detected:7,} | "
        f"Recall: {class_recall:.4f}"
    )


# ============================================================
# 18. SAVE TEST RESULTS
# ============================================================

test_output = {

    "experiment":
        "Full-data temporal Random Forest",

    "test_date":
        ["2017-07-07"],

    "train_dates":
        sorted(TRAIN_DATES),

    "validation_dates":
        sorted(VALIDATION_DATES),

    "training_rows":
        1_668_475,

    "test_rows":
        int(len(y_test)),

    "frozen_threshold":
        float(THRESHOLD),

    "test_metrics": {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "specificity":
            float(specificity),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    },

    "per_attack_family":
        per_class_results,

    "test_used_for_training":
        False,

    "test_used_for_threshold_selection":
        False,
}


with open(
    TEST_RESULTS_FILE,
    "w"
) as f:

    json.dump(
        test_output,
        f,
        indent=4
    )


print(
    "\nSaved full RF test results:"
)

print(
    TEST_RESULTS_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "FULL TEMPORAL RANDOM FOREST COMPLETE"
)

print("=" * 60)