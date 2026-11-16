import pandas as pd
import numpy as np

FILES = {
    "Train": "data/processed/train.csv",
    "Validation": "data/processed/validation.csv",
    "Test": "data/processed/test.csv"
}

print("=" * 60)
print("GRAPHSHIELD AI - SPLIT VERIFICATION")
print("=" * 60)

for name, file in FILES.items():

    print(f"\n{'-' * 60}")
    print(name.upper())
    print(f"{'-' * 60}")

    # Read only the Label column for the complete split.
    labels = pd.read_csv(
        file,
        usecols=["Label"],
        low_memory=False
    )

    print("Total rows:", len(labels))
    print("\nClass distribution:")
    print(labels["Label"].value_counts())

    print("\nClass percentages:")
    percentages = (
        labels["Label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    print(percentages)

    del labels

    # Small sample to inspect numerical feature ranges.
    sample = pd.read_csv(
        file,
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

    model_features = [
        column
        for column in sample.columns
        if column not in identifier_columns
    ]

    numeric = sample[model_features].select_dtypes(
        include=np.number
    )

    print("\nNumber of model features:", len(model_features))

    print(
        "Sample numerical range:",
        round(float(numeric.min().min()), 4),
        "to",
        round(float(numeric.max().max()), 4)
    )

print("\n" + "=" * 60)
print("SPLIT VERIFICATION COMPLETE")
print("=" * 60)