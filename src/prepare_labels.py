import pandas as pd

print("=" * 60)
print("GRAPHSHIELD AI - LABEL PREPARATION")
print("=" * 60)

input_file = "data/processed/cic_ids2017_clean.csv"
output_file = "data/processed/cic_ids2017_labels.csv"

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

print("\nChecking labels...")

print("\nOriginal labels:")
print(df["Label"].value_counts())

labels = sorted(df["Label"].unique())

label_mapping = {label: index for index, label in enumerate(labels)}

print("\nLabel mapping:")

for label, number in label_mapping.items():
    print(f"{number}: {label}")

df["Label"] = df["Label"].map(label_mapping)

print("\nEncoded labels:")
print(df["Label"].value_counts().sort_index())

print("\nNumber of classes:")
print(df["Label"].nunique())

print("\nChecking missing labels:")

missing_labels = df["Label"].isnull().sum()

print("Missing labels:", missing_labels)

if missing_labels > 0:
    print("ERROR: Some labels could not be encoded.")
    exit()

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("LABEL PREPARATION COMPLETE")
print("=" * 60)

print("\nFinal shape:")
print(df.shape)

print("\nSaved to:")
print(output_file)