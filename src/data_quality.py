import pandas as pd
import numpy as np

FILE = "data/processed/train.csv"

print("=" * 60)
print("GRAPHSHIELD AI - DATA QUALITY AUDIT")
print("=" * 60)

df = pd.read_csv(
    FILE,
    nrows=200000,
    low_memory=False
)

identifier_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

features = [
    column
    for column in df.columns
    if column not in identifier_columns
]

numeric = df[features].select_dtypes(include=np.number)

# ---------------------------------------------------------
# Negative values
# ---------------------------------------------------------

print("\nNegative values by feature:")
negative_counts = (numeric < 0).sum()

negative_counts = (
    negative_counts[negative_counts > 0]
    .sort_values(ascending=False)
)

print(negative_counts)

# ---------------------------------------------------------
# Extremely large absolute values
# ---------------------------------------------------------

print("\nExtremely large values (|value| > 1e8):")

extreme_counts = (
    numeric.abs() > 1e8
).sum()

extreme_counts = (
    extreme_counts[extreme_counts > 0]
    .sort_values(ascending=False)
)

print(extreme_counts)

# ---------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------

print("\nDetailed statistics for suspicious features:")

suspicious_features = [
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow Bytes/s",
    "Flow Packets/s"
]

for feature in suspicious_features:

    if feature in numeric.columns:

        print("\n" + "-" * 40)
        print(feature)

        print("Minimum:", numeric[feature].min())
        print("Maximum:", numeric[feature].max())
        print("Mean:", numeric[feature].mean())
        print("Negative count:", (numeric[feature] < 0).sum())
        print(
            "|value| > 1e8 count:",
            (numeric[feature].abs() > 1e8).sum()
        )

print("\n" + "=" * 60)
print("DATA QUALITY AUDIT COMPLETE")
print("=" * 60)