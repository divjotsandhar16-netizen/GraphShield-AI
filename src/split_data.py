import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("GRAPHSHIELD AI - DATA SPLITTING")
print("=" * 60)

input_file = "data/processed/cic_ids2017_labels.csv"

train_file = "data/processed/train.csv"
val_file = "data/processed/validation.csv"
test_file = "data/processed/test.csv"

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

X = df.drop(columns=["Label"])
y = df["Label"]

print("\nPerforming stratified split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

train = X_train.copy()
train["Label"] = y_train

validation = X_val.copy()
validation["Label"] = y_val

test = X_test.copy()
test["Label"] = y_test

print("\nSplit sizes:")
print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", test.shape)

print("\nTraining class distribution:")
print(train["Label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(validation["Label"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test["Label"].value_counts().sort_index())

train.to_csv(train_file, index=False)
validation.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)

print("\n" + "=" * 60)
print("DATA SPLITTING COMPLETE")
print("=" * 60)

print("\nSaved files:")
print(train_file)
print(val_file)
print(test_file)