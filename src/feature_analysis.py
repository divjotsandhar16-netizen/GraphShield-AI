import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/train.csv"

print("=" * 60)
print("GRAPHSHIELD AI - TRAINING FEATURE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# Load a representative sample for feature analysis
# We do not need to load the entire 909 MB file just to
# inspect correlations and redundant features.
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    nrows=200_000,
    low_memory=False
)

print("\nSample shape:")
print(df.shape)

# ---------------------------------------------------------
# Separate identifiers, graph information and target
# ---------------------------------------------------------

identifier_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

graph_columns = [
    "Source IP",
    "Destination IP"
]

target_column = "Label"

# ---------------------------------------------------------
# Determine model features
# ---------------------------------------------------------

model_features = [
    column
    for column in df.columns
    if column not in identifier_columns + [target_column]
]

numeric_features = df[model_features].select_dtypes(
    include=np.number
)

print("\nNumber of candidate model features:")
print(len(model_features))

print("\nNumber of numerical model features:")
print(len(numeric_features.columns))

# ---------------------------------------------------------
# Check constant features
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("CONSTANT FEATURES")
print("-" * 60)

constant_features = [
    column
    for column in numeric_features.columns
    if numeric_features[column].nunique() <= 1
]

if constant_features:
    for feature in constant_features:
        print("-", feature)
else:
    print("None found.")

# ---------------------------------------------------------
# Check perfectly correlated features
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("PERFECTLY CORRELATED FEATURES")
print("-" * 60)

correlation = numeric_features.corr()

duplicate_pairs = []

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if np.isclose(abs(value), 1.0):
            duplicate_pairs.append(
                (columns[i], columns[j], value)
            )

if duplicate_pairs:

    for feature1, feature2, value in duplicate_pairs:
        print(
            f"{feature1} <--> {feature2} "
            f"(correlation = {value:.4f})"
        )

else:
    print("No perfectly correlated feature pairs found.")

# ---------------------------------------------------------
# Show model features
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("MODEL FEATURES")
print("-" * 60)

for feature in model_features:
    print("-", feature)

print("\n" + "=" * 60)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 60)