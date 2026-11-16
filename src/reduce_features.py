import pandas as pd
import numpy as np

print("=" * 60)
print("GRAPHSHIELD AI - FEATURE REDUCTION")
print("=" * 60)

input_file = "data/processed/cic_ids2017_features.csv"
output_file = "data/processed/cic_ids2017_clean.csv"

df = pd.read_csv(input_file)

print("\nOriginal shape:")
print(df.shape)

redundant_features = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Avg Fwd Segment Size",
    "SYN Flag Count",
    "CWE Flag Count",
    "Fwd Header Length.1"
]

print("\nRemoving confirmed redundant features:")

for feature in redundant_features:
    if feature in df.columns:
        print("-", feature)
        df.drop(columns=[feature], inplace=True)

print("\nShape after feature reduction:")
print(df.shape)

numeric_columns = df.select_dtypes(include=np.number).columns

print("\nChecking for infinite values...")

inf_count = np.isinf(df[numeric_columns]).sum().sum()

print("Infinite values:", inf_count)

if inf_count > 0:
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

print("\nChecking for missing values...")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

    for column in missing.index:
        if column in numeric_columns:
            df[column] = df[column].fillna(df[column].median())

print("\nFinal missing values:")
print(df.isnull().sum().sum())

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("FEATURE REDUCTION COMPLETE")
print("=" * 60)

print("\nFinal shape:")
print(df.shape)

print("\nSaved to:")
print(output_file)