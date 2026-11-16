import pandas as pd
import numpy as np
import joblib

TRAIN_FILE = "data/processed/train.csv"
SCALER_FILE = "data/processed/scaler.pkl"

print("=" * 60)
print("GRAPHSHIELD AI - MODEL PRECHECK")
print("=" * 60)

# ---------------------------------------------------------
# Load a sample from training data
# ---------------------------------------------------------

df = pd.read_csv(
    TRAIN_FILE,
    nrows=10000,
    low_memory=False
)

identifier_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns
]

numeric_features = df[feature_columns].select_dtypes(
    include=np.number
)

print("\nNumber of model features:")
print(len(feature_columns))

print("\nNumber of numerical features:")
print(len(numeric_features.columns))

print("\nOverall feature range:")
print("Minimum:", numeric_features.min().min())
print("Maximum:", numeric_features.max().max())

print("\nOverall mean:")
print(numeric_features.mean().mean())

print("\nOverall standard deviation:")
print(numeric_features.std().mean())

# ---------------------------------------------------------
# Check -1 values
# ---------------------------------------------------------

negative_one_counts = (
    numeric_features == -1
).sum()

negative_one_features = (
    negative_one_counts[
        negative_one_counts > 0
    ]
    .sort_values(ascending=False)
)

print("\nFeatures containing -1 values:")

if len(negative_one_features) == 0:
    print("None")

else:
    print(negative_one_features.head(20))

# ---------------------------------------------------------
# Inspect scaler
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("SCALER INFORMATION")
print("-" * 60)

scaler = joblib.load(SCALER_FILE)

print("\nScaler type:")
print(type(scaler).__name__)

if hasattr(scaler, "n_features_in_"):
    print("Scaler expects features:", scaler.n_features_in_)

if hasattr(scaler, "mean_"):
    print("Scaler mean values:", len(scaler.mean_))

if hasattr(scaler, "scale_"):
    print("Scaler scale values:", len(scaler.scale_))

print("\nScaler inspection completed.")

print("\n" + "=" * 60)
print("MODEL PRECHECK COMPLETE")
print("=" * 60)