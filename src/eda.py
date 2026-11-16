import pandas as pd
import numpy as np
import os

DATA_FILE = "data/processed/cic_ids2017_clean.csv"

print("=" * 60)
print("GRAPHSHIELD AI - DATASET EXPLORATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.dtypes.value_counts())

print("\nMissing values:")
print(df.isna().sum().sum())

print("\nInfinite values:")

numeric = df.select_dtypes(include=np.number)

print(np.isinf(numeric).sum().sum())

print("\nUnique values per numerical feature:")

unique_counts = numeric.nunique().sort_values()

print(unique_counts.head(20))

print("\nConstant numerical features:")

constant_features = unique_counts[unique_counts == 1]

if len(constant_features) == 0:
    print("None")
else:
    for feature in constant_features.index:
        print(feature)

print("\nLabel distribution:")

print(df["Label"].value_counts())

print("\nProtocol distribution:")

print(df["Protocol"].value_counts())

print("\nUnique network entities:")

print("Source IPs:", df["Source IP"].nunique())

print("Destination IPs:", df["Destination IP"].nunique())

print("\nTop Source IPs:")

print(df["Source IP"].value_counts().head(10))

print("\nTop Destination IPs:")

print(df["Destination IP"].value_counts().head(10))

print("\nTop Destination Ports:")

print(df["Destination Port"].value_counts().head(10))

print("\nDataset exploration completed.")