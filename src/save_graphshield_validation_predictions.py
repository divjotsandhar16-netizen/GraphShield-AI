import os
import json
import glob

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
# VALIDATION THRESHOLD SELECTION
# ============================================================

VAL_DIR = "results/temporal_graphs/validation"

MODEL_FILE = (
    "models/graphshield_gnn_binary_best.pt"
)

OUTPUT_FILE = (
    "results/graphshield_validation_threshold.json"
)

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and
    torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("GRAPHSHIELD AI - VALIDATION THRESHOLD SELECTION")
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. MODEL
# ============================================================

class GraphShieldGNN(nn.Module):

    def __init__(
        self,
        node_features=11,
        edge_features=66,
        hidden_dim=32,
        heads=2,
        dropout=0.20,
    ):

        super().__init__()


        self.gat1 = GATv2Conv(
            in_channels=node_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_features,
            dropout=dropout,
            add_self_loops=False,
        )


        gat1_dim = (
            hidden_dim * heads
        )


        self.gat2 = GATv2Conv(
            in_channels=gat1_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_features,
            dropout=dropout,
            add_self_loops=False,
        )


        self.node_norm = nn.LayerNorm(
            hidden_dim
        )


        self.edge_classifier = nn.Sequential(

            nn.Linear(
                hidden_dim
                +
                hidden_dim
                +
                edge_features,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1,
            ),
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr,
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


        source_nodes = edge_index[0]

        destination_nodes = edge_index[1]


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
                edge_attr,
            ],
            dim=1
        )


        logits = self.edge_classifier(
            edge_representation
        ).squeeze(-1)


        return logits


# ============================================================
# 2. LOAD BEST CHECKPOINT
# ============================================================

print(
    "\nLoading best validation checkpoint..."
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model = GraphShieldGNN(
    node_features=11,
    edge_features=66,
    hidden_dim=32,
    heads=2,
    dropout=0.20,
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Saved validation PR-AUC:",
    checkpoint["validation_pr_auc"]
)


# ============================================================
# 3. GET VALIDATION BATCHES
# ============================================================

batch_files = sorted(
    glob.glob(
        os.path.join(
            VAL_DIR,
            "*.pt"
        )
    )
)


if not batch_files:

    raise RuntimeError(
        "No validation graph batches found."
    )


print(
    "\nValidation batches:",
    len(batch_files)
)


# ============================================================
# 4. COLLECT VALIDATION PROBABILITIES
# ============================================================

all_labels = []

all_probabilities = []

total_edges = 0


print(
    "\nRunning frozen validation inference..."
)


with torch.no_grad():

    for batch_number, batch_file in enumerate(
        batch_files,
        start=1
    ):

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            x = snapshot[
                "x"
            ].to(DEVICE)

            edge_index = snapshot[
                "edge_index"
            ].to(DEVICE)

            edge_attr = snapshot[
                "edge_attr"
            ].to(DEVICE)


            # Binary target:
            # 0 = benign
            # 1 = attack

            y = (
                snapshot["y"] > 0
            ).long()


            logits = model(
                x,
                edge_index,
                edge_attr
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_labels.append(
                y.numpy()
            )


            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )


            total_edges += (
                len(y)
            )


        print(
            f"Processed validation batch "
            f"{batch_number}/{len(batch_files)}"
        )


y_true = np.concatenate(
    all_labels
)

probabilities = np.concatenate(
    all_probabilities
)


# ------------------------------------------------------------
# Save per-flow validation predictions for ROC/PR curves
# ------------------------------------------------------------

np.savez_compressed(
    "results/graphshield_validation_predictions_5min.npz",
    y_true=y_true,
    probabilities=probabilities
)

print(
    "\nSaved validation prediction arrays:"
)
print(
    "results/graphshield_validation_predictions_5min.npz"
)


print(
    "\nValidation edges:",
    f"{total_edges:,}"
)


# ============================================================
# 5. THRESHOLD-INDEPENDENT METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    probabilities
)


pr_auc = average_precision_score(
    y_true,
    probabilities
)


print(
    "\n" + "-" * 60
)

print(
    "THRESHOLD-INDEPENDENT VALIDATION METRICS"
)

print(
    "-" * 60
)


print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)


print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
)


# ============================================================
# 6. THRESHOLD SWEEP
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "THRESHOLD SWEEP"
)

print(
    "-" * 60
)


thresholds = np.linspace(
    0.01,
    0.99,
    197
)


results = []


for threshold in thresholds:

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


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()


    results.append(
        {
            "threshold": float(threshold),

            "precision": float(precision),

            "recall": float(recall),

            "f1": float(f1),

            "tn": int(tn),

            "fp": int(fp),

            "fn": int(fn),

            "tp": int(tp),
        }
    )


# ============================================================
# 7. SELECT BEST F1 THRESHOLD
# ============================================================

best_result = max(
    results,
    key=lambda item: (
        item["f1"],
        item["precision"]
    )
)


best_threshold = (
    best_result["threshold"]
)


print(
    "\nBest validation threshold:",
    f"{best_threshold:.3f}"
)


print(
    "Precision:",
    f"{best_result['precision']:.4f}"
)


print(
    "Recall:",
    f"{best_result['recall']:.4f}"
)


print(
    "F1:",
    f"{best_result['f1']:.4f}"
)


print(
    "TN:",
    best_result["tn"]
)


print(
    "FP:",
    best_result["fp"]
)


print(
    "FN:",
    best_result["fn"]
)


print(
    "TP:",
    best_result["tp"]
)


# ============================================================
# 8. PRINT TOP THRESHOLDS
# ============================================================

sorted_results = sorted(
    results,
    key=lambda item: item["f1"],
    reverse=True
)


print(
    "\nTop validation thresholds:"
)


print(
    "\nThreshold | Precision | Recall | F1"
)

print(
    "-" * 42
)


for result in sorted_results[:10]:

    print(
        f"{result['threshold']:9.3f} | "
        f"{result['precision']:9.4f} | "
        f"{result['recall']:6.4f} | "
        f"{result['f1']:6.4f}"
    )


# ============================================================
# 9. SAVE THRESHOLD
# ============================================================

output = {

    "model_file": MODEL_FILE,

    "checkpoint_epoch": int(
        checkpoint["epoch"]
    ),

    "validation_edges": int(
        total_edges
    ),

    "validation_roc_auc": float(
        roc_auc
    ),

    "validation_pr_auc": float(
        pr_auc
    ),

    "selection_metric":
        "F1",

    "selected_threshold":
        float(best_threshold),

    "selected_precision":
        float(
            best_result["precision"]
        ),

    "selected_recall":
        float(
            best_result["recall"]
        ),

    "selected_f1":
        float(
            best_result["f1"]
        ),

    "confusion_matrix": {

        "TN": best_result["tn"],

        "FP": best_result["fp"],

        "FN": best_result["fn"],

        "TP": best_result["tp"],
    },

    "threshold_results": results,

    "test_used_for_selection":
        False,
}


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print(
    "\nSaved validation threshold:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 60)

print(
    "VALIDATION THRESHOLD SELECTION COMPLETE"
)

print("=" * 60)