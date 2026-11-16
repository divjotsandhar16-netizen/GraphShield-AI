import pandas as pd
import numpy as np
import joblib
import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score

print("=" * 60)
print("GRAPHSHIELD AI - GNN PROTOTYPE")
print("=" * 60)

INPUT_FILE = "data/processed/train.csv"
SCALER_FILE = "data/processed/scaler.pkl"

SAMPLE_SIZE = 10000

print("\nLoading data...")

df = pd.read_csv(INPUT_FILE, nrows=SAMPLE_SIZE)

scaler_data = joblib.load(SCALER_FILE)

scaler = scaler_data["scaler"]
feature_columns = scaler_data["feature_columns"]

source_ips = df["Source IP"].astype(str)
destination_ips = df["Destination IP"].astype(str)

all_ips = pd.concat(
    [source_ips, destination_ips]
).unique()

node_mapping = {
    ip: index for index, ip in enumerate(all_ips)
}

source_nodes = source_ips.map(node_mapping).values
destination_nodes = destination_ips.map(node_mapping).values

edge_index = torch.tensor(
    np.array([source_nodes, destination_nodes]),
    dtype=torch.long
)

edge_features = scaler.transform(df[feature_columns])

edge_attr = torch.tensor(
    edge_features,
    dtype=torch.float32
)

labels = torch.tensor(
    df["Label"].values,
    dtype=torch.long
)

num_nodes = len(node_mapping)
num_edges = edge_index.shape[1]
num_features = edge_attr.shape[1]
num_classes = 15

print("\nGraph information:")
print("Nodes:", num_nodes)
print("Edges:", num_edges)
print("Edge features:", num_features)
print("Classes:", num_classes)

node_features = torch.ones(
    (num_nodes, 1),
    dtype=torch.float32
)

graph = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=labels
)

print("\nPyG graph:")
print(graph)

class GraphShieldGNN(torch.nn.Module):

    def __init__(self, hidden_channels, num_classes):

        super().__init__()

        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, edge_attr):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        source = edge_index[0]
        destination = edge_index[1]

        edge_embedding = (
            x[source] + x[destination]
        ) / 2

        combined = torch.cat(
            [edge_embedding, edge_attr],
            dim=1
        )

        return self.classifier(combined)


print("\nInitializing GNN...")

model = GraphShieldGNN(
    hidden_channels=32,
    num_classes=num_classes
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

criterion = torch.nn.CrossEntropyLoss()

print("\nStarting training...")

model.train()

for epoch in range(1, 11):

    optimizer.zero_grad()

    output = model(
        graph.x,
        graph.edge_index,
        graph.edge_attr
    )

    loss = criterion(
        output,
        graph.y
    )

    loss.backward()

    optimizer.step()

    predictions = output.argmax(dim=1)

    accuracy = accuracy_score(
        graph.y.numpy(),
        predictions.detach().numpy()
    )

    macro_f1 = f1_score(
        graph.y.numpy(),
        predictions.detach().numpy(),
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {loss.item():.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

print("\n" + "=" * 60)
print("GNN PROTOTYPE TRAINING COMPLETE")
print("=" * 60)