import pandas as pd
import numpy as np
import joblib
import torch

print("=" * 60)
print("GRAPHSHIELD AI - GRAPH CONSTRUCTION PROTOTYPE")
print("=" * 60)

input_file = "data/processed/train.csv"
scaler_file = "data/processed/scaler.pkl"

SAMPLE_SIZE = 10000

print("\nLoading sample data...")

df = pd.read_csv(input_file, nrows=SAMPLE_SIZE)

print("Sample shape:")
print(df.shape)

print("\nLoading scaler...")

scaler_data = joblib.load(scaler_file)

scaler = scaler_data["scaler"]
feature_columns = scaler_data["feature_columns"]

print("Features:", len(feature_columns))

print("\nExtracting graph nodes...")

source_ips = df["Source IP"].astype(str)
destination_ips = df["Destination IP"].astype(str)

all_ips = pd.concat(
    [source_ips, destination_ips]
).unique()

node_mapping = {
    ip: index
    for index, ip in enumerate(all_ips)
}

print("Number of unique nodes:")
print(len(node_mapping))

print("\nCreating edges...")

source_nodes = source_ips.map(node_mapping).values
destination_nodes = destination_ips.map(node_mapping).values

edge_index = torch.tensor(
    np.array([source_nodes, destination_nodes]),
    dtype=torch.long
)

print("Edge index shape:")
print(edge_index.shape)

print("\nScaling flow features...")

edge_features = scaler.transform(
    df[feature_columns]
)

edge_features = torch.tensor(
    edge_features,
    dtype=torch.float32
)

print("Edge feature shape:")
print(edge_features.shape)

print("\nCreating labels...")

labels = torch.tensor(
    df["Label"].values,
    dtype=torch.long
)

print("Label shape:")
print(labels.shape)

print("\nChecking graph...")

print("Nodes:", len(node_mapping))
print("Edges:", edge_index.shape[1])
print("Features per edge:", edge_features.shape[1])
print("Labels:", labels.shape[0])

print("\nFirst 5 edges:")
print(edge_index[:, :5])

print("\nFirst 5 labels:")
print(labels[:5])

print("\n" + "=" * 60)
print("GRAPH PROTOTYPE COMPLETE")
print("=" * 60)