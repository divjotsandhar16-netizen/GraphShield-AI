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
# FAIR TEMPORAL RANDOM FOREST BASELINE
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

RESULTS_FILE = (
    "results/temporal_random_forest_fair_validation.json"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

TRAIN_SAMPLE_SIZE = 300_000

CHUNK_SIZE = 100_000

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


print("=" * 60)
print("GRAPHSHIELD AI - FAIR TEMPORAL RANDOM FOREST")
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
# 2. LOAD PREPROCESSOR METADATA
# ============================================================

print(
    "\nLoading training preprocessing metadata..."
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
# 3. DATA QUALITY POLICY
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


# ============================================================
# 4. CLEAN FEATURES
# ============================================================

def clean_features(chunk):

    chunk = chunk.copy()


    for column in feature_columns:

        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )


    # Infinities → missing

    chunk[feature_columns] = (
        chunk[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # Sentinel -1 values

    for column in sentinel_minus_one_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] == -1,
                column
            ] = np.nan


    # Physically invalid negatives

    for column in non_negative_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] < 0,
                column
            ] = np.nan


    # Training-only median imputation

    chunk[feature_columns] = (
        chunk[feature_columns]
        .fillna(imputation_series)
    )


    return chunk


# ============================================================
# 5. LOAD A TRAINING SAMPLE
# ============================================================

def build_training_sample():

    print(
        "\n" + "-" * 60
    )

    print(
        "BUILDING TEMPORAL TRAINING SAMPLE"
    )

    print(
        "-" * 60
    )


    # Approximate sampling fraction based on the known
    # July 3-5 temporal training population.

    known_training_rows = 1_668_475

    sample_fraction = (
        TRAIN_SAMPLE_SIZE
        /
        known_training_rows
    )


    print(
        "Target sample:",
        f"{TRAIN_SAMPLE_SIZE:,}"
    )

    print(
        "Sampling fraction:",
        f"{sample_fraction:.6f}"
    )


    parts = []

    rows_selected = 0

    chunk_counter = 0


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

            chunk_counter += 1


            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            chunk["Timestamp"] = pd.to_datetime(
                chunk["Timestamp"],
                format="mixed",
                dayfirst=True,
                errors="coerce"
            )


            # ------------------------------------------------
            # Temporal training period
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Clean
            # ------------------------------------------------

            chunk = clean_features(
                chunk
            )


            # ------------------------------------------------
            # Binary label
            # ------------------------------------------------

            chunk["binary_label"] = (
                chunk["Label"]
                .astype(int)
                >
                0
            ).astype(
                np.int8
            )


            # ------------------------------------------------
            # Stratified within-chunk sampling
            # ------------------------------------------------

            remaining = (
                TRAIN_SAMPLE_SIZE
                -
                rows_selected
            )


            if remaining <= 0:

                break


            chunk_sample_size = int(
                min(
                    len(chunk),
                    max(
                        1,
                        round(
                            len(chunk)
                            *
                            sample_fraction
                        )
                    )
                )
            )


            if chunk_sample_size > remaining:

                chunk_sample_size = (
                    remaining
                )


            if (
                chunk_sample_size
                <
                len(chunk)
            ):

                sampled_parts = []


                # Preserve benign/attack proportion
                for _, subgroup in chunk.groupby(
                    "binary_label"
                ):

                    subgroup_n = int(
                        round(
                            len(subgroup)
                            /
                            len(chunk)
                            *
                            chunk_sample_size
                        )
                    )


                    if subgroup_n > 0:

                        subgroup_n = min(
                            subgroup_n,
                            len(subgroup)
                        )


                        sampled_parts.append(
                            subgroup.sample(
                                n=subgroup_n,
                                random_state=SEED
                                +
                                chunk_counter
                                +
                                int(
                                    subgroup[
                                        "binary_label"
                                    ].iloc[0]
                                )
                            )
                        )


                if sampled_parts:

                    sampled = pd.concat(
                        sampled_parts,
                        ignore_index=True
                    )

                else:

                    sampled = chunk.sample(
                        n=chunk_sample_size,
                        random_state=SEED
                        +
                        chunk_counter
                    )

            else:

                sampled = chunk


            # If rounding produced slightly too many rows,
            # trim deterministically.

            if len(sampled) > remaining:

                sampled = sampled.sample(
                    n=remaining,
                    random_state=SEED + chunk_counter
                )


            parts.append(
                sampled
            )


            rows_selected += len(
                sampled
            )


            print(
                f"  Selected training rows: "
                f"{rows_selected:,}"
            )


        if rows_selected >= TRAIN_SAMPLE_SIZE:

            break


    if not parts:

        raise RuntimeError(
            "No temporal training rows found."
        )


    train_df = pd.concat(
        parts,
        ignore_index=True
    )


    # Final exact-size correction

    if len(train_df) > TRAIN_SAMPLE_SIZE:

        train_df = train_df.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=SEED
        ).reset_index(
            drop=True
        )


    print(
        "\nFinal training sample:",
        f"{len(train_df):,}"
    )


    return train_df


# ============================================================
# 6. BUILD TRAINING DATA
# ============================================================

train_df = build_training_sample()


X_train = train_df[
    feature_columns
].to_numpy(
    dtype=np.float32
)


y_train = train_df[
    "binary_label"
].to_numpy(
    dtype=np.int8
)


print(
    "\nTraining class distribution:"
)


train_unique, train_counts = np.unique(
    y_train,
    return_counts=True
)


for label, count in zip(
    train_unique,
    train_counts
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
# 7. TRAIN RANDOM FOREST
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TRAINING RANDOM FOREST"
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
    "\nRandom Forest training complete."
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


print(
    "Saved model:"
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
    "RUNNING FROZEN VALIDATION INFERENCE"
)

print(
    "-" * 60
)


all_val_labels = []

all_val_probabilities = []

total_val_edges = 0


use_columns = [
    "Timestamp",
    "Label",
] + feature_columns


for split_file in SPLIT_FILES:

    print(
        f"\nReading validation data from: "
        f"{split_file}"
    )


    for chunk_index, chunk in enumerate(
        pd.read_csv(
            split_file,
            usecols=use_columns,
            chunksize=CHUNK_SIZE,
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


        total_val_edges += len(y)


        print(
            f"  Validation rows processed: "
            f"{total_val_edges:,}"
        )


y_val = np.concatenate(
    all_val_labels
)


val_probabilities = np.concatenate(
    all_val_probabilities
)


print(
    "\nTotal validation flows:",
    f"{len(y_val):,}"
)


# ============================================================
# 10. THRESHOLD-INDEPENDENT METRICS
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
    "\n" + "-" * 60
)

print(
    "VALIDATION THRESHOLD-INDEPENDENT METRICS"
)

print(
    "-" * 60
)


print(
    "ROC-AUC:",
    f"{val_roc_auc:.4f}"
)


print(
    "PR-AUC:",
    f"{val_pr_auc:.4f}"
)


# ============================================================
# 11. THRESHOLD SWEEP
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
            "threshold": float(
                threshold
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "tn": int(tn),

            "fp": int(fp),

            "fn": int(fn),

            "tp": int(tp),
        }
    )


# ============================================================
# 12. SELECT VALIDATION THRESHOLD
# ============================================================

best_threshold_result = max(
    threshold_results,
    key=lambda item: (
        item["f1"],
        item["precision"]
    )
)


best_threshold = (
    best_threshold_result[
        "threshold"
    ]
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
    f"{best_threshold:.3f}"
)


print(
    "Precision:",
    f"{best_threshold_result['precision']:.4f}"
)


print(
    "Recall:",
    f"{best_threshold_result['recall']:.4f}"
)


print(
    "F1:",
    f"{best_threshold_result['f1']:.4f}"
)


print(
    "TN:",
    f"{best_threshold_result['tn']:,}"
)


print(
    "FP:",
    f"{best_threshold_result['fp']:,}"
)


print(
    "FN:",
    f"{best_threshold_result['fn']:,}"
)


print(
    "TP:",
    f"{best_threshold_result['tp']:,}"
)


# ============================================================
# 13. TOP VALIDATION THRESHOLDS
# ============================================================

print(
    "\nTop validation thresholds:"
)


print(
    "\nThreshold | Precision | Recall | F1"
)

print(
    "-" * 42
)


for result in sorted(
    threshold_results,
    key=lambda item: item["f1"],
    reverse=True
)[:10]:

    print(
        f"{result['threshold']:9.3f} | "
        f"{result['precision']:9.4f} | "
        f"{result['recall']:6.4f} | "
        f"{result['f1']:6.4f}"
    )


# ============================================================
# 14. SAVE VALIDATION RESULTS
# ============================================================

output = {

    "experiment":
        "Fair temporal Random Forest baseline",

    "model_file":
        MODEL_FILE,

    "train_dates":
        sorted(TRAIN_DATES),

    "validation_dates":
        sorted(VALIDATION_DATES),

    "test_dates":
        sorted(TEST_DATES),

    "train_sample_size":
        int(len(train_df)),

    "validation_size":
        int(len(y_val)),

    "hyperparameters": {

        "n_estimators": 100,

        "max_depth": 20,

        "min_samples_leaf": 2,

        "max_features": "sqrt",

        "class_weight":
            "balanced_subsample",

        "random_state": SEED,
    },

    "validation_roc_auc":
        float(val_roc_auc),

    "validation_pr_auc":
        float(val_pr_auc),

    "selected_threshold":
        float(best_threshold),

    "selected_precision":
        float(
            best_threshold_result[
                "precision"
            ]
        ),

    "selected_recall":
        float(
            best_threshold_result[
                "recall"
            ]
        ),

    "selected_f1":
        float(
            best_threshold_result[
                "f1"
            ]
        ),

    "confusion_matrix": {

        "TN":
            best_threshold_result["tn"],

        "FP":
            best_threshold_result["fp"],

        "FN":
            best_threshold_result["fn"],

        "TP":
            best_threshold_result["tp"],
    },

    "threshold_selection_metric":
        "F1",

    "test_used":
        False,

    "test_used_for_threshold_selection":
        False,

    "threshold_results":
        threshold_results,
}


with open(
    RESULTS_FILE,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print(
    "\nSaved validation results:"
)

print(
    RESULTS_FILE
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TEMPORAL RANDOM FOREST BASELINE COMPLETE"
)

print("=" * 60)


print(
    "\nTraining:"
)

print(
    "July 3-5"
)

print(
    f"{len(train_df):,} sampled flows"
)


print(
    "\nValidation:"
)

print(
    "July 6"
)

print(
    f"{len(y_val):,} flows"
)


print(
    "\nTest:"
)

print(
    "July 7"
)

print(
    "NOT USED"
)


print(
    "\nSelected validation threshold:",
    f"{best_threshold:.3f}"
)


print("=" * 60)