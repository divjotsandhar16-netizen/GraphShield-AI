import os
import json
import joblib
import numpy as np
import pandas as pd

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
# FINAL FROZEN RANDOM FOREST TEST EVALUATION
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
    "models/temporal_random_forest_fair.pkl"
)

VALIDATION_RESULTS_FILE = (
    "results/temporal_random_forest_fair_validation.json"
)

OUTPUT_FILE = (
    "results/temporal_random_forest_fair_test.json"
)

TEST_DATES = {
    "2017-07-07",
}


# ============================================================
# LABEL NAMES
# ============================================================

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
    12: "Brute Force",
    13: "Sql Injection",
    14: "XSS",
}


print("=" * 60)
print("GRAPHSHIELD AI - FINAL TEMPORAL RF TEST")
print("=" * 60)


# ============================================================
# 1. LOAD PREPROCESSOR
# ============================================================

print("\nLoading training-only preprocessing artifact...")

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
        f"Expected 66 features, found "
        f"{len(feature_columns)}"
    )


# ============================================================
# 2. SAME CLEANING POLICY
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
# 3. LOAD FROZEN MODEL
# ============================================================

print(
    "\nLoading trained Random Forest..."
)

model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)


# ============================================================
# 4. LOAD FROZEN VALIDATION THRESHOLD
# ============================================================

print(
    "\nLoading frozen validation threshold..."
)


with open(
    VALIDATION_RESULTS_FILE,
    "r"
) as f:

    validation_results = json.load(f)


THRESHOLD = float(
    validation_results[
        "selected_threshold"
    ]
)


print(
    "Frozen threshold:",
    f"{THRESHOLD:.3f}"
)


print(
    "Validation F1 used for selection:",
    f"{validation_results['selected_f1']:.4f}"
)


# ============================================================
# 5. TEST INFERENCE
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "RUNNING FROZEN JULY 7 RF INFERENCE"
)

print(
    "-" * 60
)

print(
    "No threshold tuning will be performed."
)


use_columns = [
    "Timestamp",
    "Label",
] + feature_columns


all_original_labels = []
all_binary_labels = []
all_probabilities = []

total_test_rows = 0


for split_file in SPLIT_FILES:

    print(
        f"\nReading: {split_file}"
    )


    for chunk_index, chunk in enumerate(
        pd.read_csv(
            split_file,
            usecols=use_columns,
            chunksize=100_000,
            low_memory=False
        ),
        start=1
    ):

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        # ----------------------------------------------------
        # Only July 7
        # ----------------------------------------------------

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


        # Original 15-class labels

        original_labels = (
            chunk["Label"]
            .astype(int)
            .to_numpy()
        )


        # Binary labels

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

        all_probabilities.append(
            probabilities
        )


        total_test_rows += len(
            chunk
        )


        print(
            f"  Test rows processed: "
            f"{total_test_rows:,}"
        )


# ============================================================
# 6. COMBINE
# ============================================================

y_original = np.concatenate(
    all_original_labels
)

y_true = np.concatenate(
    all_binary_labels
)

probabilities = np.concatenate(
    all_probabilities
)


y_pred = (
    probabilities
    >=
    THRESHOLD
).astype(
    np.int8
)


print(
    "\nTotal July 7 test flows:",
    f"{len(y_true):,}"
)


# ============================================================
# 7. TEST CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TEST CLASS DISTRIBUTION"
)

print(
    "-" * 60
)


unique_labels, counts = np.unique(
    y_original,
    return_counts=True
)


test_class_distribution = {}


for label, count in zip(
    unique_labels,
    counts
):

    label = int(label)
    count = int(count)

    name = LABEL_NAMES.get(
        label,
        f"Class {label}"
    )


    test_class_distribution[
        str(label)
    ] = {

        "name": name,
        "count": count,
    }


    print(
        f"{label:2d} | "
        f"{name:20s} | "
        f"{count:,}"
    )


# ============================================================
# 8. OVERALL METRICS
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_true,
    probabilities
)


pr_auc = average_precision_score(
    y_true,
    probabilities
)


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "FINAL JULY 7 RANDOM FOREST RESULTS"
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
# 9. PER-ATTACK FAMILY
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


    class_predictions = (
        y_pred[mask]
    )


    flows = int(
        mask.sum()
    )


    detected = int(
        class_predictions.sum()
    )


    class_recall = (
        detected / flows
        if flows > 0
        else 0.0
    )


    name = LABEL_NAMES.get(
        label,
        f"Class {label}"
    )


    per_class_results[
        str(label)
    ] = {

        "name": name,

        "flows": flows,

        "detected": detected,

        "missed": flows - detected,

        "recall": float(
            class_recall
        ),
    }


    print(
        f"{name:20s} | "
        f"Flows: {flows:7,} | "
        f"Detected: {detected:7,} | "
        f"Recall: {class_recall:.4f}"
    )


# ============================================================
# 10. UNSEEN ATTACKS
# ============================================================

unseen_labels = [
    1,   # Bot
    2,   # DDoS
    10,  # PortScan
]


print(
    "\n" + "-" * 60
)

print(
    "UNSEEN ATTACK FAMILIES"
)

print(
    "-" * 60
)


unseen_total = 0
unseen_detected = 0


for label in unseen_labels:

    key = str(label)


    if key not in per_class_results:

        continue


    result = per_class_results[key]


    print(
        f"{result['name']:20s} | "
        f"Flows: {result['flows']:,} | "
        f"Recall: {result['recall']:.4f}"
    )


    unseen_total += result[
        "flows"
    ]

    unseen_detected += result[
        "detected"
    ]


unseen_recall = (
    unseen_detected
    /
    unseen_total
    if unseen_total > 0
    else 0.0
)


print(
    "\nAggregate unseen-family recall:",
    f"{unseen_recall:.4f}"
)


# ============================================================
# 11. SAVE FINAL RESULTS
# ============================================================

results = {

    "experiment":
        "Frozen temporal Random Forest test",

    "test_date":
        ["2017-07-07"],

    "model_file":
        MODEL_FILE,

    "threshold":
        float(THRESHOLD),

    "threshold_selection":
        {
            "dataset":
                "July 6 validation",

            "metric":
                validation_results[
                    "threshold_selection_metric"
                ],

            "validation_f1":
                float(
                    validation_results[
                        "selected_f1"
                    ]
                ),
        },


    "test_metrics":
        {

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


    "test_class_distribution":
        test_class_distribution,


    "per_attack_family":
        per_class_results,


    "unseen_attack_families":
        {

            "labels":
                unseen_labels,

            "total_flows":
                int(unseen_total),

            "detected":
                int(unseen_detected),

            "aggregate_recall":
                float(unseen_recall),
        },


    "test_was_frozen":
        True,

    "test_used_for_threshold_selection":
        False,
}


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nSaved final RF test results:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "FINAL RANDOM FOREST TEST COMPLETE"
)

print("=" * 60)

print(
    "\nJuly 7 was evaluated using the "
    "frozen validation threshold."
)

print(
    "No test-based tuning was performed."
)

print("=" * 60)