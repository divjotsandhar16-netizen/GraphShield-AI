import os
import json
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from torch_geometric.nn import GATv2Conv


# ============================================================
# GRAPHSHIELD AI
# TOPOLOGY-ISOLATED GNN
# ============================================================
#
# Node features retained:
#
# 0  out_flow_count
# 1  in_flow_count
# 4  out_bytes
# 5  in_bytes
# 6  out_packets
# 7  in_packets
# 10 unique_protocols
#
# Removed:
#
# 2  unique_destinations
# 3  unique_sources
# 8  out_unique_ports
# 9  in_unique_ports
#
# Therefore explicit topology summaries are removed.
# The model still has graph connectivity and GATv2 message passing.
# ============================================================


TRAIN_DIR = "results/temporal_graphs/train"
VAL_DIR = "results/temporal_graphs/validation"
TEST_DIR = "results/temporal_graphs/test"

MODEL_FILE = (
    "models/graphshield_gnn_topology_isolated.pt"
)

RESULTS_FILE = (
    "results/graphshield_gnn_topology_isolated.json"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

EPOCHS = 3

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 32

HEADS = 2

DROPOUT = 0.20

GRADIENT_CLIP = 2.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print(
    "GRAPHSHIELD AI - TOPOLOGY-ISOLATED GNN"
)
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. LOAD GRAPH FILES
# ============================================================

def get_batch_files(directory):

    return sorted(
        glob.glob(
            os.path.join(
                directory,
                "*.pt"
            )
        )
    )


train_files = get_batch_files(
    TRAIN_DIR
)

val_files = get_batch_files(
    VAL_DIR
)

test_files = get_batch_files(
    TEST_DIR
)


print(
    "\nTrain batches:",
    len(train_files)
)

print(
    "Validation batches:",
    len(val_files)
)

print(
    "Test batches:",
    len(test_files)
)


if not train_files:
    raise RuntimeError(
        "Training graph files not found."
    )

if not val_files:
    raise RuntimeError(
        "Validation graph files not found."
    )

if not test_files:
    raise RuntimeError(
        "Test graph files not found."
    )


# ============================================================
# 2. CLASS BALANCE
# ============================================================

with open(
    "results/temporal_graphs/metadata.json",
    "r"
) as f:

    metadata = json.load(f)


train_labels = (
    metadata["splits"]
    ["train"]
    ["label_counts"]
)


benign_count = int(
    train_labels["0"]
)


attack_count = sum(
    int(count)

    for label, count
    in train_labels.items()

    if int(label) != 0
)


positive_weight = (
    benign_count
    /
    attack_count
)


POS_WEIGHT = torch.tensor(
    positive_weight,
    dtype=torch.float32,
    device=DEVICE
)


print(
    "\nTraining benign flows:",
    f"{benign_count:,}"
)

print(
    "Training attack flows:",
    f"{attack_count:,}"
)

print(
    "Positive class weight:",
    f"{positive_weight:.4f}"
)


# ============================================================
# 3. NODE FEATURE INDICES
# ============================================================

# Original 11 node features:
#
# 0 out_flow_count
# 1 in_flow_count
# 2 unique_destinations       ← REMOVE
# 3 unique_sources            ← REMOVE
# 4 out_bytes
# 5 in_bytes
# 6 out_packets
# 7 in_packets
# 8 out_unique_ports          ← REMOVE
# 9 in_unique_ports           ← REMOVE
# 10 unique_protocols

NODE_FEATURE_INDICES = [
    0,
    1,
    4,
    5,
    6,
    7,
    10,
]


NODE_FEATURE_COUNT = 7


print(
    "\nRetained node-history features:",
    NODE_FEATURE_COUNT
)

print(
    "Removed topology-summary features: 4"
)


# ============================================================
# 4. MODEL
# ============================================================

class TopologyIsolatedGNN(
    nn.Module
):

    """
    GNN using only non-topological historical endpoint
    behavior plus the actual graph structure.

    This allows us to test whether message passing itself
    contributes beyond explicit topology-summary features.
    """


    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # GATv2 Layer 1
        # ----------------------------------------------------

        self.gat1 = GATv2Conv(

            in_channels=NODE_FEATURE_COUNT,

            out_channels=HIDDEN_DIM,

            heads=HEADS,

            concat=True,

            edge_dim=66,

            dropout=DROPOUT,

            add_self_loops=False,
        )


        gat1_dim = (
            HIDDEN_DIM
            *
            HEADS
        )


        # ----------------------------------------------------
        # GATv2 Layer 2
        # ----------------------------------------------------

        self.gat2 = GATv2Conv(

            in_channels=gat1_dim,

            out_channels=HIDDEN_DIM,

            heads=HEADS,

            concat=False,

            edge_dim=66,

            dropout=DROPOUT,

            add_self_loops=False,
        )


        self.node_norm = nn.LayerNorm(
            HIDDEN_DIM
        )


        # ----------------------------------------------------
        # Edge classifier
        # ----------------------------------------------------

        self.edge_classifier = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM
                +
                HIDDEN_DIM
                +
                66,

                64
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        x = self.gat1(
            x,
            edge_index,
            edge_attr
        )

        x = F.elu(x)


        x = self.gat2(
            x,
            edge_index,
            edge_attr
        )

        x = F.elu(x)


        x = self.node_norm(
            x
        )


        source_nodes = (
            edge_index[0]
        )

        destination_nodes = (
            edge_index[1]
        )


        source_embedding = x[
            source_nodes
        ]


        destination_embedding = x[
            destination_nodes
        ]


        edge_representation = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_attr
            ],
            dim=1
        )


        logits = self.edge_classifier(
            edge_representation
        ).squeeze(-1)


        return logits


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = TopologyIsolatedGNN().to(
    DEVICE
)


print(
    "\n" + "-" * 60
)

print(
    "MODEL"
)

print(
    "-" * 60
)

print(model)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "\nTrainable parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# 6. LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=POS_WEIGHT
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 7. PREPARE SNAPSHOT
# ============================================================

def prepare_snapshot(
    snapshot
):

    # --------------------------------------------------------
    # Select ONLY 7 retained node-history features
    # --------------------------------------------------------

    x = snapshot["x"][
        :,
        NODE_FEATURE_INDICES
    ].to(
        DEVICE
    )


    edge_index = snapshot[
        "edge_index"
    ].to(
        DEVICE
    )


    edge_attr = snapshot[
        "edge_attr"
    ].to(
        DEVICE
    )


    y = (
        snapshot["y"] > 0
    ).float().to(
        DEVICE
    )


    return (
        x,
        edge_index,
        edge_attr,
        y
    )


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int8
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )


    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    return {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),
    }


# ============================================================
# 9. TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    epoch
):

    model.train()


    total_loss = 0.0

    total_edges = 0

    processed_snapshots = 0


    for batch_file in train_files:

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            (
                x,
                edge_index,
                edge_attr,
                y
            ) = prepare_snapshot(
                snapshot
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = model(
                x,
                edge_index,
                edge_attr
            )


            loss = criterion(
                logits,
                y
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )


            optimizer.step()


            n = y.numel()


            total_loss += (
                loss.item()
                *
                n
            )


            total_edges += n

            processed_snapshots += 1


            if (
                processed_snapshots % 50
                == 0
            ):

                print(
                    f"  Epoch {epoch} | "
                    f"Snapshots: "
                    f"{processed_snapshots:,} | "
                    f"Edges: "
                    f"{total_edges:,}"
                )


    return (
        total_loss
        /
        total_edges
    )


# ============================================================
# 10. INFERENCE
# ============================================================

@torch.no_grad()
def infer(
    files
):

    model.eval()


    all_labels = []

    all_probabilities = []


    for batch_file in files:

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            (
                x,
                edge_index,
                edge_attr,
                y
            ) = prepare_snapshot(
                snapshot
            )


            logits = model(
                x,
                edge_index,
                edge_attr
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_labels.append(
                y.cpu().numpy()
            )


            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )


    return (
        np.concatenate(
            all_labels
        ),

        np.concatenate(
            all_probabilities
        )
    )


# ============================================================
# 11. TRAINING LOOP
# ============================================================

history = []

best_pr_auc = -np.inf

best_epoch = None


print(
    "\n" + "=" * 60
)

print(
    "STARTING TOPOLOGY-ISOLATED TRAINING"
)

print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        f"{'=' * 60}"
    )


    train_loss = train_epoch(
        epoch
    )


    y_val, val_probabilities = infer(
        val_files
    )


    val_metrics = calculate_metrics(
        y_val,
        val_probabilities
    )


    print(
        "\nTraining loss:",
        f"{train_loss:.6f}"
    )


    print(
        "Validation ROC-AUC:",
        f"{val_metrics['roc_auc']:.4f}"
    )


    print(
        "Validation PR-AUC:",
        f"{val_metrics['pr_auc']:.4f}"
    )


    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                float(train_loss),

            "validation":
                val_metrics,
        }
    )


    if (
        val_metrics["pr_auc"]
        >
        best_pr_auc
    ):

        best_pr_auc = (
            val_metrics["pr_auc"]
        )

        best_epoch = epoch


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "validation_pr_auc":
                    best_pr_auc,

                "model_config":
                    {
                        "node_features":
                            NODE_FEATURE_COUNT,

                        "retained_indices":
                            NODE_FEATURE_INDICES,

                        "edge_features":
                            66,

                        "hidden_dim":
                            HIDDEN_DIM,

                        "heads":
                            HEADS,

                        "dropout":
                            DROPOUT,
                    },

                "ablation":
                    "topology_isolated_gnn",
            },

            MODEL_FILE
        )


        print(
            "\n*** New best model saved ***"
        )


# ============================================================
# 12. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


# ============================================================
# 13. VALIDATION THRESHOLD
# ============================================================

y_val, val_probabilities = infer(
    val_files
)


thresholds = np.linspace(
    0.01,
    0.99,
    197
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        val_probabilities
        >=
        threshold
    ).astype(
        np.int8
    )


    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )


    threshold_results.append(
        {
            "threshold":
                float(threshold),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),
        }
    )


best_threshold_result = max(
    threshold_results,
    key=lambda item: (
        item["f1"],
        item["precision"]
    )
)


THRESHOLD = (
    best_threshold_result[
        "threshold"
    ]
)


print(
    "\n" + "-" * 60
)

print(
    "VALIDATION THRESHOLD"
)

print(
    "-" * 60
)


print(
    "Threshold:",
    f"{THRESHOLD:.3f}"
)

print(
    "Precision:",
    f"{best_threshold_result['precision']:.4f}"
)

print(
    "Recall:",
    f"{best_threshold_result['recall']:.4f}"
)

print(
    "F1:",
    f"{best_threshold_result['f1']:.4f}"
)


# ============================================================
# 14. JULY 7 EXPLORATORY TEST
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "JULY 7 EXPLORATORY EVALUATION"
)

print("=" * 60)

print(
    "NOTE: July 7 has already been inspected "
    "for earlier model variants."
)

print(
    "This result is therefore exploratory, "
    "not a new pristine test."
)


y_test, test_probabilities = infer(
    test_files
)


test_predictions = (
    test_probabilities
    >=
    THRESHOLD
).astype(
    np.int8
)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)


test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()


specificity = (
    tn
    /
    (tn + fp)
    if (
        tn + fp
    ) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "TOPOLOGY-ISOLATED TEST RESULTS"
)

print(
    "-" * 60
)


print(
    "Threshold:",
    f"{THRESHOLD:.3f}"
)


print(
    "Precision:",
    f"{test_precision:.4f}"
)


print(
    "Recall:",
    f"{test_recall:.4f}"
)


print(
    "F1:",
    f"{test_f1:.4f}"
)


print(
    "ROC-AUC:",
    f"{test_roc_auc:.4f}"
)


print(
    "PR-AUC:",
    f"{test_pr_auc:.4f}"
)


print(
    "Specificity:",
    f"{specificity:.4f}"
)


print(
    "\nConfusion matrix:"
)

print(
    "TN:",
    f"{tn:,}"
)

print(
    "FP:",
    f"{fp:,}"
)

print(
    "FN:",
    f"{fn:,}"
)

print(
    "TP:",
    f"{tp:,}"
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "Topology-isolated GNN",

    "description":
        (
            "GNN retains actual graph connectivity and "
            "message passing, but removes explicit "
            "topology-summary node features."
        ),

    "removed_node_features":
        [
            "unique_destinations",
            "unique_sources",
            "out_unique_ports",
            "in_unique_ports",
        ],

    "retained_node_features":
        [
            "out_flow_count",
            "in_flow_count",
            "out_bytes",
            "in_bytes",
            "out_packets",
            "in_packets",
            "unique_protocols",
        ],

    "node_features":
        7,

    "edge_features":
        66,

    "train_edges":
        1_668_475,

    "validation_edges":
        458_825,

    "test_edges":
        703_240,

    "best_epoch":
        best_epoch,

    "best_validation_pr_auc":
        float(best_pr_auc),

    "selected_threshold":
        float(THRESHOLD),

    "validation_f1":
        float(
            best_threshold_result[
                "f1"
            ]
        ),

    "test_metrics_exploratory":
        {
            "precision":
                float(test_precision),

            "recall":
                float(test_recall),

            "f1":
                float(test_f1),

            "roc_auc":
                float(test_roc_auc),

            "pr_auc":
                float(test_pr_auc),

            "specificity":
                float(specificity),

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),
        },

    "july_7_pristine_test":
        False,

    "history":
        history,
}


with open(
    RESULTS_FILE,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nSaved model:"
)

print(
    MODEL_FILE
)


print(
    "\nSaved results:"
)

print(
    RESULTS_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "TOPOLOGY-ISOLATED GNN COMPLETE"
)

print("=" * 60)