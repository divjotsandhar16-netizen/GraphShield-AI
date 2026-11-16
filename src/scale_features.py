import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("GRAPHSHIELD AI - FEATURE SCALER")
print("=" * 60)

train_file = "data/processed/train.csv"
scaler_output = "data/processed/scaler.pkl"

print("\nLoading training data...")

train = pd.read_csv(train_file)

print("Training shape:")
print(train.shape)

columns_to_exclude = [
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

feature_columns = [
    column for column in train.columns
    if column not in columns_to_exclude
]

print("\nNumber of features:")
print(len(feature_columns))

print("\nFitting StandardScaler...")

scaler = StandardScaler()

scaler.fit(train[feature_columns])

joblib.dump(
    {
        "scaler": scaler,
        "feature_columns": feature_columns
    },
    scaler_output
)

print("\nScaler fitted successfully.")

print("\nSaved to:")
print(scaler_output)

print("\n" + "=" * 60)
print("FEATURE SCALER COMPLETE")
print("=" * 60)