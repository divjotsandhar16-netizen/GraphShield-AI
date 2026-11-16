import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = (
    "models/random_forest_temporal_baseline.pkl"
)

TEST_FILE = (
    "data/processed/test.csv"
)

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

ATTACK_LABELS = {
    1: "Bot",
    2: "DDoS",
    10: "PortScan",
}


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI - TEMPORAL RF ANALYSIS")
print("=" * 60)

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]
imputer = bundle["imputer"]

print("\nModel loaded successfully.")


# =========================================================
# LOAD JULY 7 TEST DATA
# =========================================================

print("\nLoading July 7 test traffic...")

chunks = []

usecols = MODEL_FEATURES + [
    "Label",
    "Timestamp",
]

for chunk in pd.read_csv(
    TEST_FILE,
    usecols=usecols,
    chunksize=200_000,
    low_memory=False,
):

    timestamp = pd.to_datetime(
        chunk["Timestamp"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    date = timestamp.dt.strftime(
        "%Y-%m-%d"
    )

    july7 = chunk.loc[
        date == "2017-07-07"
    ].copy()

    if len(july7) > 0:
        chunks.append(july7)


test_df = pd.concat(
    chunks,
    ignore_index=True,
)

print(
    f"Total July 7 rows available: "
    f"{len(test_df):,}"
)


# =========================================================
# CLEAN TEST DATA
# =========================================================

invalid_mask = (
    (test_df["Fwd Header Length"] < 0)
    | (test_df["Bwd Header Length"] < 0)
    | (test_df["min_seg_size_forward"] < 0)
    | (test_df["Flow Duration"] < 0)
    | (test_df["Flow Bytes/s"] < 0)
    | (test_df["Flow Packets/s"] < 0)
)

test_df = test_df.loc[
    ~invalid_mask
].copy()

for column in [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]:

    test_df.loc[
        test_df[column] == -1,
        column,
    ] = np.nan


# =========================================================
# PREDICTION
# =========================================================

X_test = test_df[
    MODEL_FEATURES
].copy()

X_test = imputer.transform(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# PER-ATTACK PROBABILITY ANALYSIS
# =========================================================

print("\n" + "-" * 60)
print("PER-ATTACK SCORE DISTRIBUTION")
print("-" * 60)

for label, name in ATTACK_LABELS.items():

    mask = (
        test_df["Label"].astype(int)
        == label
    )

    scores = probabilities[mask]

    print(
        f"\n{name}"
    )

    print(
        f"Flows: {len(scores):,}"
    )

    if len(scores) > 0:

        print(
            f"Mean score: "
            f"{scores.mean():.4f}"
        )

        print(
            f"Median score: "
            f"{np.median(scores):.4f}"
        )

        print(
            f"90th percentile: "
            f"{np.percentile(scores, 90):.4f}"
        )

        print(
            f"Maximum score: "
            f"{scores.max():.4f}"
        )


# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

print("\n" + "-" * 60)
print("THRESHOLD ANALYSIS ON JULY 7")
print("-" * 60)

y_test = (
    test_df["Label"]
    .astype(int)
    .ne(0)
    .astype(int)
)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

print(
    "\nThreshold | Precision | Recall | F1"
)

print("-" * 45)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )


# =========================================================
# DEFAULT THRESHOLD PER-ATTACK RECALL
# =========================================================

print("\n" + "-" * 60)
print("PER-ATTACK RECALL AT THRESHOLD 0.50")
print("-" * 60)

default_predictions = (
    probabilities >= 0.50
).astype(int)

for label, name in ATTACK_LABELS.items():

    mask = (
        test_df["Label"].astype(int)
        == label
    )

    true_values = np.ones(
        mask.sum(),
        dtype=int,
    )

    predicted_values = (
        default_predictions[mask]
    )

    recall = recall_score(
        true_values,
        predicted_values,
        zero_division=0,
    )

    print(
        f"{name:10s}: "
        f"{recall:.4f}"
    )


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("TEMPORAL RF ANALYSIS COMPLETE")
print("=" * 60)