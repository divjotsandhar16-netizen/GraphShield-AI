import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# GRAPHSHIELD AI
# TEMPORAL TRAINING-ONLY PREPROCESSOR
# ============================================================

SPLIT_FILES = [
    "data/processed/train.csv",
    "data/processed/validation.csv",
    "data/processed/test.csv",
]

OUTPUT_FILE = (
    "data/processed/graphshield_preprocessor.joblib"
)

CHUNK_SIZE = 100_000

# Approximately 15% of the ~1.67M temporal-training rows.
# Used only to estimate medians while keeping memory reasonable.
MEDIAN_SAMPLE_FRACTION = 0.15

TRAIN_DATES = {
    "2017-07-03",
    "2017-07-04",
    "2017-07-05",
}


print("=" * 60)
print("GRAPHSHIELD AI - TEMPORAL TRAINING PREPROCESSOR")
print("=" * 60)


# ============================================================
# 1. IDENTIFY THE 66 MODEL FEATURES
# ============================================================

metadata_columns = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
}


header = pd.read_csv(
    SPLIT_FILES[0],
    nrows=0
)

all_columns = list(header.columns)

feature_columns = [
    column
    for column in all_columns
    if column not in metadata_columns
]


print(
    f"\nTotal columns: {len(all_columns)}"
)

print(
    f"Model features: {len(feature_columns)}"
)


if len(feature_columns) != 66:

    raise ValueError(
        f"Expected 66 model features, "
        f"found {len(feature_columns)}"
    )


print("66 model features confirmed.")


# ============================================================
# 2. DATA-QUALITY POLICY
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


print(
    f"\nNon-negative feature rules: "
    f"{len(non_negative_features)}"
)

print(
    "Sentinel -1 features:",
    sentinel_minus_one_features
)


# ============================================================
# 3. CLEANING FUNCTION
# ============================================================

def clean_feature_chunk(chunk):

    chunk = chunk.copy()

    # Convert all model features to numeric
    for column in feature_columns:

        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )


    # Convert infinities to missing values
    chunk[feature_columns] = (
        chunk[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # --------------------------------------------------------
    # Sentinel -1 handling
    # --------------------------------------------------------

    for column in sentinel_minus_one_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] == -1,
                column
            ] = np.nan


    # --------------------------------------------------------
    # Physically invalid negative values
    # --------------------------------------------------------

    for column in non_negative_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] < 0,
                column
            ] = np.nan


    return chunk


# ============================================================
# 4. PASS 1
#    ESTIMATE MEDIANS FROM JULY 3-5 ONLY
# ============================================================

print("\n" + "-" * 60)
print("PASS 1 - TEMPORAL TRAINING-ONLY MEDIAN ESTIMATION")
print("-" * 60)

print(
    "Training period:",
    sorted(TRAIN_DATES)
)

print(
    f"Median sample fraction: "
    f"{MEDIAN_SAMPLE_FRACTION}"
)


median_sample_parts = []

total_source_rows = 0
total_training_rows = 0

chunk_counter = 0


use_columns = [
    "Timestamp"
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

        total_source_rows += len(chunk)


        # ----------------------------------------------------
        # Parse timestamp
        # ----------------------------------------------------

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        # ----------------------------------------------------
        # Keep ONLY July 3-5
        # ----------------------------------------------------

        chunk = chunk[
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(TRAIN_DATES)
        ].copy()


        if len(chunk) == 0:
            continue


        total_training_rows += len(chunk)


        # ----------------------------------------------------
        # Clean values
        # ----------------------------------------------------

        chunk = clean_feature_chunk(chunk)


        # ----------------------------------------------------
        # Deterministic random sample from this chunk
        # ----------------------------------------------------

        sample_size = max(
            1,
            int(
                len(chunk)
                *
                MEDIAN_SAMPLE_FRACTION
            )
        )


        if sample_size < len(chunk):

            sampled = chunk.sample(
                n=sample_size,
                random_state=42 + chunk_counter
            )

        else:

            sampled = chunk


        median_sample_parts.append(
            sampled[feature_columns]
        )


        if chunk_counter % 5 == 0:

            print(
                f"Source rows processed: "
                f"{total_source_rows:,} | "
                f"Temporal training rows: "
                f"{total_training_rows:,}"
            )


print(
    f"\nTotal temporal training rows found: "
    f"{total_training_rows:,}"
)


if total_training_rows == 0:

    raise RuntimeError(
        "No July 3-5 training-period rows found."
    )


# Combine sampled rows

median_sample = pd.concat(
    median_sample_parts,
    ignore_index=True
)


print(
    f"Median sample rows: "
    f"{len(median_sample):,}"
)


# ============================================================
# 5. CALCULATE TRAINING-ONLY MEDIANS
# ============================================================

print(
    "\nCalculating training-only medians..."
)


imputation_values = {}


for column in feature_columns:

    value = median_sample[
        column
    ].median()


    if pd.isna(value):

        value = 0.0


    imputation_values[column] = float(
        value
    )


imputation_series = pd.Series(
    imputation_values,
    dtype=np.float64
)


print(
    "Median calculation complete."
)


# ============================================================
# 6. PASS 2
#    FIT SCALER ON ALL JULY 3-5 ROWS
# ============================================================

print("\n" + "-" * 60)
print("PASS 2 - TEMPORAL TRAINING-ONLY STANDARD SCALER")
print("-" * 60)


scaler = StandardScaler()

rows_used_for_scaler = 0


for split_file in SPLIT_FILES:

    print(
        f"\nReading for scaler: {split_file}"
    )


    for chunk in pd.read_csv(
        split_file,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        # ----------------------------------------------------
        # Parse timestamp
        # ----------------------------------------------------

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )


        # ----------------------------------------------------
        # Keep ONLY July 3-5
        # ----------------------------------------------------

        chunk = chunk[
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(TRAIN_DATES)
        ].copy()


        if len(chunk) == 0:
            continue


        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        chunk = clean_feature_chunk(chunk)


        # ----------------------------------------------------
        # Impute using training-only medians
        # ----------------------------------------------------

        chunk[feature_columns] = (
            chunk[feature_columns]
            .fillna(imputation_series)
        )


        # ----------------------------------------------------
        # Convert to numeric matrix
        # ----------------------------------------------------

        values = chunk[
            feature_columns
        ].to_numpy(
            dtype=np.float64
        )


        # ----------------------------------------------------
        # Incremental StandardScaler fitting
        # ----------------------------------------------------

        scaler.partial_fit(
            values
        )


        rows_used_for_scaler += len(chunk)


        print(
            f"Scaler rows fitted: "
            f"{rows_used_for_scaler:,}"
        )


# ============================================================
# 7. VERIFY SCALER
# ============================================================

print(
    "\nVerifying scaler..."
)


if not np.isfinite(
    scaler.mean_
).all():

    raise ValueError(
        "Scaler mean contains invalid values."
    )


if not np.isfinite(
    scaler.scale_
).all():

    raise ValueError(
        "Scaler scale contains invalid values."
    )


if np.any(
    scaler.scale_ <= 0
):

    bad_columns = [
        feature_columns[i]
        for i, value in enumerate(
            scaler.scale_
        )
        if value <= 0
    ]

    raise ValueError(
        "Non-positive scale found for: "
        f"{bad_columns}"
    )


print(
    "Scaler statistics are finite and valid."
)


# ============================================================
# 8. BUILD PREPROCESSOR ARTIFACT
# ============================================================

preprocessor = {

    "feature_columns": feature_columns,

    "imputation_values": imputation_values,

    "scaler": scaler,

    "fit_dates": sorted(
        TRAIN_DATES
    ),

    "source_files": SPLIT_FILES,

    "temporal_training_rows": total_training_rows,

    "scaler_fit_rows": rows_used_for_scaler,

    "median_sample_rows": len(
        median_sample
    ),

    "invalid_negative_policy": (
        "Negative values in predefined "
        "non-negative flow features "
        "converted to NaN before imputation."
    ),

    "sentinel_policy": (
        "-1 in Init_Win_bytes_forward "
        "and Init_Win_bytes_backward "
        "converted to NaN before imputation."
    ),
}


# ============================================================
# 9. SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


joblib.dump(
    preprocessor,
    OUTPUT_FILE
)


print(
    "\nSaved temporal training preprocessor:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL PREPROCESSOR BUILD COMPLETE")
print("=" * 60)

print(
    "\nModel features:",
    len(feature_columns)
)

print(
    "Training dates:",
    sorted(TRAIN_DATES)
)

print(
    "Temporal training rows:",
    f"{total_training_rows:,}"
)

print(
    "Scaler fit rows:",
    f"{rows_used_for_scaler:,}"
)

print(
    "Median sample rows:",
    f"{len(median_sample):,}"
)

print("\nPreprocessor contains:")

print("  ✓ 66 feature names")
print("  ✓ training-only imputation values")
print("  ✓ training-only StandardScaler")
print("  ✓ July 3-5 temporal training period")
print("  ✓ invalid-value policy")
print("  ✓ source split files used")

print("\n" + "=" * 60)