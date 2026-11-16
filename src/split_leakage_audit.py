import numpy as np
import pandas as pd


TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

CHUNK_SIZE = 200_000

TARGET_COLUMN = "Label"

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


# =========================================================
# DATE DISTRIBUTION
# =========================================================

def get_date_distribution(file_path):

    date_counts = {}
    invalid_timestamps = 0
    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        total_rows += len(chunk)

        # IMPORTANT:
        # CIC-IDS2017 timestamps are day/month/year.
        # format="mixed" handles the mixed timestamp strings.
        timestamps = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        invalid_timestamps += int(
            timestamps.isna().sum()
        )

        valid_dates = timestamps.dropna().dt.date

        counts = valid_dates.value_counts()

        for date_value, count in counts.items():

            date_counts[date_value] = (
                date_counts.get(date_value, 0)
                + int(count)
            )

    distribution = pd.Series(
        date_counts
    ).sort_index()

    return (
        total_rows,
        distribution,
        invalid_timestamps,
    )


# =========================================================
# BUILD TRAINING HASHES
# =========================================================

def build_train_hashes(file_path):

    print("\nBuilding training signatures...")

    feature_hashes = set()
    feature_label_hashes = set()

    usecols = MODEL_FEATURES + [TARGET_COLUMN]

    rows_processed = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        feature_hash = pd.util.hash_pandas_object(
            chunk[MODEL_FEATURES],
            index=False,
        )

        feature_label_hash = pd.util.hash_pandas_object(
            chunk,
            index=False,
        )

        feature_hashes.update(
            feature_hash.to_numpy(
                dtype=np.uint64
            )
        )

        feature_label_hashes.update(
            feature_label_hash.to_numpy(
                dtype=np.uint64
            )
        )

        rows_processed += len(chunk)

        print(
            f"Processed: {rows_processed:,}",
            end="\r",
        )

    print()

    return feature_hashes, feature_label_hashes


# =========================================================
# COUNT OVERLAP
# =========================================================

def count_overlap(
    file_path,
    train_feature_hashes,
    train_feature_label_hashes,
):

    usecols = MODEL_FEATURES + [TARGET_COLUMN]

    total_rows = 0
    feature_matches = 0
    feature_label_matches = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        feature_hash = pd.util.hash_pandas_object(
            chunk[MODEL_FEATURES],
            index=False,
        )

        feature_label_hash = pd.util.hash_pandas_object(
            chunk,
            index=False,
        )

        feature_values = feature_hash.to_numpy(
            dtype=np.uint64
        )

        feature_label_values = feature_label_hash.to_numpy(
            dtype=np.uint64
        )

        feature_matches += int(
            np.isin(
                feature_values,
                list(train_feature_hashes),
            ).sum()
        )

        feature_label_matches += int(
            np.isin(
                feature_label_values,
                list(train_feature_label_hashes),
            ).sum()
        )

        total_rows += len(chunk)

        print(
            f"Processed: {total_rows:,}",
            end="\r",
        )

    print()

    feature_percentage = (
        feature_matches / total_rows * 100
    )

    feature_label_percentage = (
        feature_label_matches / total_rows * 100
    )

    return (
        total_rows,
        feature_matches,
        feature_percentage,
        feature_label_matches,
        feature_label_percentage,
    )


# =========================================================
# MAIN
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI - CORRECTED SPLIT LEAKAGE AUDIT")
print("=" * 60)


# =========================================================
# 1. CORRECT DATE DISTRIBUTION
# =========================================================

print("\n" + "-" * 60)
print("CORRECTED DATE DISTRIBUTION")
print("-" * 60)

for name, file_path in [
    ("TRAIN", TRAIN_FILE),
    ("VALIDATION", VALIDATION_FILE),
    ("TEST", TEST_FILE),
]:

    print(f"\n{name}")

    total, distribution, invalid = (
        get_date_distribution(file_path)
    )

    print(f"Total rows: {total:,}")

    print("\nParsed dates:")

    print(distribution)

    print(
        f"\nInvalid/unparsed timestamps: "
        f"{invalid:,}"
    )


# =========================================================
# 2. TRAINING SIGNATURES
# =========================================================

print("\n" + "-" * 60)
print("TRAINING SIGNATURES")
print("-" * 60)

train_feature_hashes, train_feature_label_hashes = (
    build_train_hashes(TRAIN_FILE)
)

print(
    f"\nUnique feature signatures: "
    f"{len(train_feature_hashes):,}"
)

print(
    f"Unique feature+label signatures: "
    f"{len(train_feature_label_hashes):,}"
)


# =========================================================
# 3. VALIDATION OVERLAP
# =========================================================

print("\n" + "-" * 60)
print("VALIDATION ↔ TRAIN")
print("-" * 60)

(
    total,
    feature_matches,
    feature_pct,
    feature_label_matches,
    feature_label_pct,
) = count_overlap(
    VALIDATION_FILE,
    train_feature_hashes,
    train_feature_label_hashes,
)

print(f"\nValidation rows: {total:,}")

print(
    f"Feature-only exact matches: "
    f"{feature_matches:,}"
)

print(
    f"Feature-only overlap: "
    f"{feature_pct:.4f}%"
)

print(
    f"Feature+label exact matches: "
    f"{feature_label_matches:,}"
)

print(
    f"Feature+label overlap: "
    f"{feature_label_pct:.4f}%"
)


# =========================================================
# 4. TEST OVERLAP
# =========================================================

print("\n" + "-" * 60)
print("TEST ↔ TRAIN")
print("-" * 60)

(
    total,
    feature_matches,
    feature_pct,
    feature_label_matches,
    feature_label_pct,
) = count_overlap(
    TEST_FILE,
    train_feature_hashes,
    train_feature_label_hashes,
)

print(f"\nTest rows: {total:,}")

print(
    f"Feature-only exact matches: "
    f"{feature_matches:,}"
)

print(
    f"Feature-only overlap: "
    f"{feature_pct:.4f}%"
)

print(
    f"Feature+label exact matches: "
    f"{feature_label_matches:,}"
)

print(
    f"Feature+label overlap: "
    f"{feature_label_pct:.4f}%"
)


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("CORRECTED LEAKAGE AUDIT COMPLETE")
print("=" * 60)