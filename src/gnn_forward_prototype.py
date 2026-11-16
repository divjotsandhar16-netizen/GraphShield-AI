import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# ============================================================
# GRAPHSHIELD AI
# GNN FORWARD-PASS PROTOTYPE
# ============================================================

INPUT_FILE = "results/temporal_graph_prototype.pt"

# Dataset has 15 attack/benign classes
NUM_CLASSES = 15

# Graph input dimensions
NODE_FEATURES = 11
EDGE_FEATURES = 66


print("=" * 60)
print("GRAPHSHIELD AI - GNN FORWARD PASS PROTOTYPE")
print("=" * 60)


# ============================================================
# 1. LOAD TEMPORAL SNAPSHOTS
# ============================================================

print("\nLoading temporal graph snapshots...")

snapshots = torch.load(
    INPUT_FILE,
    map_location="cpu",
    weights_only=False
)

print(
    f"Loaded {len(snapshots)} snapshots."
)


# ============================================================
# 2. SELECT A SNAPSHOT
# ============================================================

# Use the first snapshot only for architecture verification.
snapshot = snapshots[0]

print("\nUsing snapshot:")

print(
    "Window:",
    snapshot["window_start"]
)


# ============================================================
# 3. CREATE PYTORCH GEOMETRIC DATA OBJECT
# ============================================================

data = Data(
    x=snapshot["x"],
    edge_index=snapshot["edge_index"],
    edge_attr=snapshot["edge_attr"],
    y=snapshot["y"]
)


print("\nGraph:")

print(data)

print(
    "Nodes:",
    data.num_nodes
)

print(
    "Edges:",
    data.num_edges
)

print(
    "Node features:",
    data.num_node_features
)

print(
    "Edge features:",
    data.edge_attr.shape[1]
)


# ============================================================
# 4. NUMERICAL SAFETY FOR THIS PROTOTYPE
# ============================================================

# IMPORTANT:
#
# This is ONLY a forward-pass prototype.
#
# The final model will use a scaler fitted only on
# training data.
#
# Here we simply normalize each feature dimension
# so that extremely large raw CIC-IDS2017 values
# do not dominate this architecture smoke test.

def normalize_features(x):
    """
    Simple column-wise normalization for prototype testing.
    This is NOT the final training preprocessing.
    """

    x = x.float()

    mean = x.mean(dim=0, keepdim=True)

    std = x.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std
    )

    return (x - mean) / std


x = normalize_features(data.x)

edge_attr = normalize_features(
    data.edge_attr
)


# ============================================================
# 5. EDGE-LEVEL GNN MODEL
# ============================================================

class GraphShieldGNN(nn.Module):

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_classes
    ):

        super().__init__()


        # ----------------------------------------------------
        # GATv2 Layer 1
        # ----------------------------------------------------

        self.gat1 = GATv2Conv(
            in_channels=node_features,
            out_channels=hidden_dim,
            heads=2,
            concat=True,
            edge_dim=edge_features,
            dropout=0.0,
            add_self_loops=False
        )


        # Output dimension:
        #
        # hidden_dim × number_of_heads
        #
        # 32 × 2 = 64

        gat1_output_dim = (
            hidden_dim * 2
        )


        # ----------------------------------------------------
        # GATv2 Layer 2
        # ----------------------------------------------------

        self.gat2 = GATv2Conv(
            in_channels=gat1_output_dim,
            out_channels=hidden_dim,
            heads=2,
            concat=False,
            edge_dim=edge_features,
            dropout=0.0,
            add_self_loops=False
        )


        # Because concat=False:
        #
        # 32 × 2 heads
        # becomes
        # 32-dimensional output

        self.node_embedding_dim = hidden_dim


        # ----------------------------------------------------
        # Edge classifier
        # ----------------------------------------------------

        # We combine:
        #
        # source node embedding
        # destination node embedding
        # original edge features
        #
        # 32 + 32 + 66
        # = 130

        edge_input_dim = (
            hidden_dim
            +
            hidden_dim
            +
            edge_features
        )


        self.edge_classifier = nn.Sequential(

            nn.Linear(
                edge_input_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        # ----------------------------------------------------
        # Graph message passing - layer 1
        # ----------------------------------------------------

        x = self.gat1(
            x,
            edge_index,
            edge_attr
        )

        x = F.elu(x)


        # ----------------------------------------------------
        # Graph message passing - layer 2
        # ----------------------------------------------------

        x = self.gat2(
            x,
            edge_index,
            edge_attr
        )

        x = F.elu(x)


        # ----------------------------------------------------
        # Identify source and destination node
        # for every flow edge
        # ----------------------------------------------------

        source_nodes = edge_index[0]

        destination_nodes = edge_index[1]


        source_embedding = x[
            source_nodes
        ]

        destination_embedding = x[
            destination_nodes
        ]


        # ----------------------------------------------------
        # Construct edge representation
        # ----------------------------------------------------

        edge_representation = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_attr
            ],
            dim=1
        )


        # ----------------------------------------------------
        # Edge classification
        # ----------------------------------------------------

        logits = self.edge_classifier(
            edge_representation
        )


        return logits, x


# ============================================================
# 6. CREATE MODEL
# ============================================================

print("\n" + "-" * 60)
print("CREATING GRAPHSHIELD GNN")
print("-" * 60)


model = GraphShieldGNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
    hidden_dim=32,
    num_classes=NUM_CLASSES
)


print(model)


# ============================================================
# 7. PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "\nTotal parameters:",
    total_parameters
)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# 8. RUN FORWARD PASS
# ============================================================

print("\n" + "-" * 60)
print("RUNNING GNN FORWARD PASS")
print("-" * 60)


model.eval()


with torch.no_grad():

    logits, node_embeddings = model(
        x=x,
        edge_index=data.edge_index,
        edge_attr=edge_attr
    )


# ============================================================
# 9. OUTPUT SHAPES
# ============================================================

print(
    "\nNode embedding shape:"
)

print(
    tuple(
        node_embeddings.shape
    )
)


print(
    "\nEdge logits shape:"
)

print(
    tuple(
        logits.shape
    )
)


# ============================================================
# 10. EXPECTED SHAPE CHECKS
# ============================================================

expected_node_shape = (
    data.num_nodes,
    32
)

expected_edge_shape = (
    data.num_edges,
    NUM_CLASSES
)


node_shape_valid = (
    tuple(node_embeddings.shape)
    ==
    expected_node_shape
)


edge_shape_valid = (
    tuple(logits.shape)
    ==
    expected_edge_shape
)


print("\n" + "-" * 60)
print("GNN OUTPUT VERIFICATION")
print("-" * 60)


print(
    "Expected node embedding shape:",
    expected_node_shape
)

print(
    "Actual node embedding shape:",
    tuple(node_embeddings.shape)
)

print(
    "Node embedding shape valid:",
    node_shape_valid
)


print(
    "\nExpected edge logits shape:",
    expected_edge_shape
)

print(
    "Actual edge logits shape:",
    tuple(logits.shape)
)

print(
    "Edge logits shape valid:",
    edge_shape_valid
)


# ============================================================
# 11. NUMERICAL STABILITY CHECK
# ============================================================

logits_finite = torch.isfinite(
    logits
).all().item()


embeddings_finite = torch.isfinite(
    node_embeddings
).all().item()


print(
    "\nEdge logits finite:",
    logits_finite
)

print(
    "Node embeddings finite:",
    embeddings_finite
)


# ============================================================
# 12. CONVERT LOGITS TO PREDICTIONS
# ============================================================

predicted_classes = torch.argmax(
    logits,
    dim=1
)


print(
    "\nFirst 10 predicted classes:"
)

print(
    predicted_classes[:10]
)


print(
    "\nFirst 10 actual labels:"
)

print(
    data.y[:10]
)


# ============================================================
# 13. PREDICTION DISTRIBUTION
# ============================================================

unique_predictions, counts = torch.unique(
    predicted_classes,
    return_counts=True
)


print(
    "\nPredicted class distribution:"
)

for label, count in zip(
    unique_predictions.tolist(),
    counts.tolist()
):

    print(
        f"Class {label}: {count} edges"
    )


# ============================================================
# 14. FINAL VERIFICATION
# ============================================================

all_checks_passed = all(
    [
        node_shape_valid,
        edge_shape_valid,
        logits_finite,
        embeddings_finite
    ]
)


print("\n" + "=" * 60)

if all_checks_passed:

    print(
        "ALL GNN FORWARD-PASS CHECKS PASSED"
    )

else:

    print(
        "SOME GNN CHECKS FAILED"
    )

print("=" * 60)