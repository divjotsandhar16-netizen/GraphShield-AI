import pandas as pd
import numpy as np

FILE = "data/processed/train.csv"

print("=" * 60)
print("GRAPHSHIELD AI - DEEP DATA QUALITY AUDIT")
print("=" * 60)

# Columns required for investigation
use_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Flow Duration",
    "Flow IAT Min",
    "Flow IAT Mean",
    "Flow IAT Max"
]

chunk_size = 200_000

invalid_rows = []
negative_rate_rows = []
sentinel_counts = {
    "Init_Win_bytes_forward": 0,
    "Init_Win_bytes_backward": 0,
    "Flow IAT Min": 0,
    "Flow IAT Mean": 0,
    "Flow IAT Max": 0,
    "Flow Duration": 0
}

total_rows = 0

for chunk in pd.read_csv(
    FILE,
    usecols=use_columns,
    chunksize=chunk_size,
    low_memory=False
):

    total_rows += len(chunk)

    # -----------------------------------------------------
    # Clearly invalid physical values
    # -----------------------------------------------------

    invalid_mask = (
        (chunk["Fwd Header Length"] < 0)
        | (chunk["Bwd Header Length"] < 0)
        | (chunk["min_seg_size_forward"] < 0)
    )

    current_invalid = chunk[invalid_mask]

    if len(current_invalid) > 0:
        invalid_rows.append(current_invalid)

    # -----------------------------------------------------
    # Negative traffic rates
    # -----------------------------------------------------

    rate_mask = (
        (chunk["Flow Bytes/s"] < 0)
        | (chunk["Flow Packets/s"] < 0)
    )

    current_rates = chunk[rate_mask]

    if len(current_rates) > 0:
        negative_rate_rows.append(current_rates)

    # -----------------------------------------------------
    # Sentinel -1 values
    # -----------------------------------------------------

    sentinel_features = [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Flow IAT Min",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow Duration"
    ]

    for feature in sentinel_features:
        sentinel_counts[feature] += int(
            (chunk[feature] == -1).sum()
        )

# ---------------------------------------------------------
# Combine suspicious rows
# ---------------------------------------------------------

print("\nTotal training rows scanned:")
print(f"{total_rows:,}")

if invalid_rows:

    invalid_df = pd.concat(
        invalid_rows,
        ignore_index=True
    )

    print("\n" + "-" * 60)
    print("INVALID HEADER / SEGMENT ROWS")
    print("-" * 60)

    print("Count:", len(invalid_df))

    print("\nLabels:")
    print(invalid_df["Label"].value_counts())

    print("\nProtocols:")
    print(invalid_df["Protocol"].value_counts())

    print("\nRows:")
    print(invalid_df.to_string(index=False))

else:
    print("\nNo invalid header/segment rows found.")

# ---------------------------------------------------------
# Negative rate rows
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("NEGATIVE RATE VALUES")
print("-" * 60)

if negative_rate_rows:

    rate_df = pd.concat(
        negative_rate_rows,
        ignore_index=True
    )

    print("Count:", len(rate_df))

    print("\nLabels:")
    print(rate_df["Label"].value_counts())

    print("\nRows:")
    print(rate_df.to_string(index=False))

else:
    print("No negative rate values found.")

# ---------------------------------------------------------
# Sentinel statistics
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("SENTINEL -1 COUNTS")
print("-" * 60)

for feature, count in sentinel_counts.items():
    print(f"{feature}: {count:,}")

print("\n" + "=" * 60)
print("DEEP DATA QUALITY AUDIT COMPLETE")
print("=" * 60)