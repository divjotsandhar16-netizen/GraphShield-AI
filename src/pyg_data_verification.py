import os
import torch
from torch_geometric.data import Data


# ============================================================
# GRAPHSHIELD AI - PYTORCH GEOMETRIC DATA VERIFICATION
# ============================================================

INPUT_FILE = "results/temporal_graph_prototype.pt"


print("=" * 60)
print("GRAPHSHIELD AI - PYTORCH GEOMETRIC DATA VERIFICATION")
print("=" * 60)


# ============================================================
# 1. LOAD TEMPORAL GRAPH SNAPSHOTS
# ============================================================

print("\nLoading temporal graph snapshots...")

snapshots = torch.load(
    INPUT_FILE,
    map_location="cpu",
    weights_only=False
)

print(
    f"Loaded {len(snapshots)} temporal snapshots."
)


# ============================================================
# 2. SELECT ONE SNAPSHOT
# ============================================================

snapshot = snapshots[0]

print("\nUsing first snapshot:")
print(
    "Window:",
    snapshot["window_start"]
)


# ============================================================
# 3. EXTRACT GRAPH COMPONENTS
# ============================================================

x = snapshot["x"]

edge_index = snapshot["edge_index"]

edge_attr = snapshot["edge_attr"]

y = snapshot["y"]


print("\nRaw tensor shapes:")

print(
    "Node features x:",
    tuple(x.shape)
)

print(
    "Edge index:",
    tuple(edge_index.shape)
)

print(
    "Edge features:",
    tuple(edge_attr.shape)
)

print(
    "Labels:",
    tuple(y.shape)
)


# ============================================================
# 4. CREATE PYTORCH GEOMETRIC DATA OBJECT
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y
)


# ============================================================
# 5. BASIC GRAPH INFORMATION
# ============================================================

print("\n" + "-" * 60)
print("PYTORCH GEOMETRIC DATA OBJECT")
print("-" * 60)

print(data)


print("\nNumber of nodes:")
print(data.num_nodes)


print("\nNumber of edges:")
print(data.num_edges)


print("\nNumber of node features:")
print(data.num_node_features)


print("\nNumber of edge features:")
print(data.edge_attr.shape[1])


# ============================================================
# 6. VERIFY GRAPH SHAPES
# ============================================================

print("\n" + "-" * 60)
print("STRUCTURAL VERIFICATION")
print("-" * 60)


# Node feature matrix
node_shape_valid = (
    data.x.shape[0]
    ==
    data.num_nodes
)


# Edge index
edge_index_valid = (
    data.edge_index.shape[0]
    ==
    2
)


# Edge feature alignment
edge_attr_valid = (
    data.edge_attr.shape[0]
    ==
    data.num_edges
)


# Edge label alignment
label_alignment_valid = (
    data.y.shape[0]
    ==
    data.num_edges
)


print(
    "Node feature shape valid:",
    node_shape_valid
)

print(
    "Edge index shape valid:",
    edge_index_valid
)

print(
    "Edge-feature alignment:",
    edge_attr_valid
)

print(
    "Edge-label alignment:",
    label_alignment_valid
)


# ============================================================
# 7. CHECK EDGE INDICES
# ============================================================

print("\n" + "-" * 60)
print("EDGE INDEX VERIFICATION")
print("-" * 60)


min_index = int(
    data.edge_index.min().item()
)

max_index = int(
    data.edge_index.max().item()
)


print(
    "Minimum node index:",
    min_index
)

print(
    "Maximum node index:",
    max_index
)


edge_indices_valid = (
    min_index >= 0
    and
    max_index < data.num_nodes
)


print(
    "All edge indices reference valid nodes:",
    edge_indices_valid
)


# ============================================================
# 8. SELF LOOP CHECK
# ============================================================

self_loops = int(
    (
        data.edge_index[0]
        ==
        data.edge_index[1]
    ).sum().item()
)


print(
    "\nSelf-loops:",
    self_loops
)


# ============================================================
# 9. LABEL VERIFICATION
# ============================================================

unique_labels = torch.unique(
    data.y
).tolist()


print(
    "\nLabels in first snapshot:",
    unique_labels
)


# ============================================================
# 10. DISPLAY SAMPLE DATA
# ============================================================

print("\n" + "-" * 60)
print("SAMPLE GRAPH DATA")
print("-" * 60)


print("\nFirst 5 node feature vectors:")

print(
    data.x[:5]
)


print("\nFirst 5 edges:")

print(
    data.edge_index[:, :5]
)


print("\nFirst 5 edge feature vectors:")

print(
    data.edge_attr[:5]
)


print("\nFirst 5 labels:")

print(
    data.y[:5]
)


# ============================================================
# 11. FINAL PASS/FAIL
# ============================================================

all_checks_passed = all(
    [
        node_shape_valid,
        edge_index_valid,
        edge_attr_valid,
        label_alignment_valid,
        edge_indices_valid,
        self_loops == 0
    ]
)


print("\n" + "=" * 60)

if all_checks_passed:

    print(
        "ALL PYTORCH GEOMETRIC CHECKS PASSED"
    )

else:

    print(
        "SOME PYTORCH GEOMETRIC CHECKS FAILED"
    )

print("=" * 60)