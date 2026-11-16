import pandas as pd
import numpy as np
import glob
import os

DATA_PATH = "data/TrafficLabelling"
OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

print(f"Found {len(files)} CSV files")

dataframes = []
total_rows_loaded = 0

for file in files:
    print(f"\nLoading: {os.path.basename(file)}")

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    total_rows_loaded += len(df)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\n" + "=" * 60)
print("DATASET COMBINED")
print("=" * 60)

print(f"Total rows loaded: {len(data):,}")
print(f"Total columns: {len(data.columns)}")

required_columns = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label"
]

missing_required = [
    column for column in required_columns
    if column not in data.columns
]

if missing_required:
    print("\nMissing required columns:")
    print(missing_required)
    raise ValueError("Required columns are missing.")

print("\nRequired columns verified.")

identifier_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

for column in identifier_columns:
    data[column] = data[column].astype("string").str.strip()

print("\nIdentifier columns cleaned.")

numeric_columns = data.select_dtypes(include=np.number).columns

infinite_values = np.isinf(data[numeric_columns]).sum().sum()

print(f"\nInfinite numerical values found: {infinite_values:,}")

data[numeric_columns] = data[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

nan_before = data.isna().sum().sum()

print(f"NaN values after infinity conversion: {nan_before:,}")

essential_columns = [
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

rows_before_essential = len(data)

data.dropna(
    subset=essential_columns,
    inplace=True
)

rows_removed_essential = rows_before_essential - len(data)

print(
    f"Rows removed due to missing essential information: "
    f"{rows_removed_essential:,}"
)

numeric_columns = data.select_dtypes(include=np.number).columns

missing_numeric = data[numeric_columns].isna().sum().sum()

print(f"Missing numerical values to be imputed: {missing_numeric:,}")

for column in numeric_columns:
    if data[column].isna().any():
        data[column] = data[column].fillna(
            data[column].median()
        )

remaining_nan = data.isna().sum().sum()

print(f"Remaining missing values: {remaining_nan:,}")

duplicates = data.duplicated().sum()

print(f"\nDuplicate rows found: {duplicates:,}")

data.drop_duplicates(inplace=True)

data["Label"] = data["Label"].astype("string").str.strip()

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(data["Label"].value_counts())

output_file = os.path.join(
    OUTPUT_PATH,
    "cic_ids2017_clean.csv"
)

data.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Final rows: {len(data):,}")
print(f"Final columns: {len(data.columns)}")

print(f"\nClean dataset saved to:")
print(output_file)