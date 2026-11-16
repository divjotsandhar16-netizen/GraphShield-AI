import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/cic_ids2017_clean.csv"
OUTPUT_PATH = "data/processed"

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("GRAPHSHIELD AI - FEATURE SELECTION")
print("=" * 60)

print("\nOriginal shape:")
print(df.shape)

constant_features = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if df[column].nunique() <= 1
]

print("\nConstant features:")
for feature in constant_features:
    print("-", feature)

df = df.drop(columns=constant_features)

print("\nRemoved constant features:", len(constant_features))
print("Shape after removing constants:", df.shape)

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

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns + [target_column]
]

print("\nIdentifier columns:")
print(identifier_columns)

print("\nGraph columns:")
print(graph_columns)

print("\nTarget column:")
print(target_column)

print("\nNumber of model features:")
print(len(feature_columns))

print("\nModel features:")
for feature in feature_columns:
    print("-", feature)

output_file = os.path.join(
    OUTPUT_PATH,
    "cic_ids2017_features.csv"
)

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE")
print("=" * 60)

print("Saved to:")
print(output_file)