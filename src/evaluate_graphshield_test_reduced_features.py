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
# FINAL FROZEN TEST EVALUATION
# ============================================================

TEST_DIR = (
    "results/temporal_graphs/test"
)

MODEL_FILE = (
    "models/graphshield_gnn_reduced_features_best.pt"
)

THRESHOLD_FILE = (
    "results/graphshield_validation_threshold_reduced_features.json"
)

OUTPUT_FILE = (
    "results/graphshield_final_test_results_reduced_features.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


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
print("GRAPHSHIELD AI - FINAL FROZEN TEST EVALUATION")
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. LOAD FROZEN THRESHOLD
# ============================================================

print(
    "\nLoading frozen validation threshold..."
)


with open(
    THRESHOLD_FILE,
    "r"
) as f:

    threshold_data = json.load(f)


THRESHOLD = float(
    threshold_data[
        "selected_threshold"
    ]
)


print(
    "Frozen threshold:",
    f"{THRESHOLD:.3f}"
)


print(
    "Threshold selected using:",
    threshold_data[
        "selection_metric"
    ]
)


print(
    "Test used for threshold selection:",
    threshold_data[
        "test_used_for_selection"
    ]
)


if threshold_data[
    "test_used_for_selection"
]:

    raise RuntimeError(
        "ERROR: Test data was marked as "
        "used for threshold selection."
    )


# ============================================================
# 2. MODEL DEFINITION
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


        # ----------------------------------------------------
        # GATv2 Layer 1
        # ----------------------------------------------------

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
            hidden_dim
            *
            heads
        )


        # ----------------------------------------------------
        # GATv2 Layer 2
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Edge classifier
        # ----------------------------------------------------

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
            edge_attr,
        )

        x = F.elu(x)


        x = self.gat2(
            x,
            edge_index,
            edge_attr,
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
# 3. LOAD FROZEN MODEL
# ============================================================

print(
    "\nLoading frozen GNN checkpoint..."
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model_config = checkpoint.get(
    "model_config",
    {}
)


model = GraphShieldGNN(
    node_features=model_config.get(
        "node_features",
        11
    ),

    edge_features=model_config.get(
        "edge_features",
        66
    ),

    hidden_dim=model_config.get(
        "hidden_dim",
        32
    ),

    heads=model_config.get(
        "heads",
        2
    ),

    dropout=model_config.get(
        "dropout",
        0.20
    ),
).to(DEVICE)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)


print(
    "Validation PR-AUC at selection:",
    f'{checkpoint["validation_pr_auc"]:.4f}'
)


# ============================================================
# 4. TEST BATCH FILES
# ============================================================

test_files = sorted(
    glob.glob(
        os.path.join(
            TEST_DIR,
            "*.pt"
        )
    )
)


if not test_files:

    raise RuntimeError(
        "No test graph batch files found."
    )


print(
    "\nTest batch files:",
    len(test_files)
)


# ============================================================
# 5. ATTACK LABEL NAMES
# ============================================================

label_names = {

    0: "BENIGN",

    1: "Bot",

    2: "DDoS",

    3: "DoS GoldenEye",

    4: "DoS Hulk",

    5: "DoS Slowhttptest",

    6: "DoS slowloris",

    7: "FTP-Patator",

    8: "Heartbleed",

    9: "Infiltration",

    10: "PortScan",

    11: "SSH-Patator",

    12: "Brute Force",

    13: "Sql Injection",

    14: "XSS",
}


# ============================================================
# 6. FROZEN TEST INFERENCE
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "RUNNING FROZEN JULY 7 TEST INFERENCE"
)

print(
    "-" * 60
)

print(
    "Threshold:",
    f"{THRESHOLD:.3f}"
)

print(
    "NO threshold tuning is performed on TEST."
)


all_binary_labels = []

all_original_labels = []

all_probabilities = []

total_edges = 0


with torch.no_grad():

    for batch_number, test_file in enumerate(
        test_files,
        start=1
    ):

        print(
            f"\nLoading test batch "
            f"{batch_number}/{len(test_files)}"
        )


        snapshots = torch.load(
            test_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            x = snapshot[
                "x"
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

            # Experiment 11:
            # Remove the two verified redundant features:
            # 52. Avg Bwd Segment Size
            # 53. Subflow Fwd Bytes
            edge_attr = torch.cat(
                [
                    edge_attr[:, :51],
                    edge_attr[:, 53:],
                ],
                dim=1
            )


            # Original 15-class labels
            original_labels = snapshot[
                "y"
            ].long()


            # Binary evaluation label
            #
            # 0 = BENIGN
            # 1 = ATTACK

            binary_labels = (
                original_labels > 0
            ).long()


            logits = model(
                x,
                edge_index,
                edge_attr
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_original_labels.append(
                original_labels.numpy()
            )


            all_binary_labels.append(
                binary_labels.numpy()
            )


            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )


            total_edges += (
                len(
                    original_labels
                )
            )


        print(
            f"Completed batch "
            f"{batch_number}/{len(test_files)}"
        )


# ============================================================
# 7. COMBINE TEST OUTPUTS
# ============================================================

y_original = np.concatenate(
    all_original_labels
)


y_true = np.concatenate(
    all_binary_labels
)


probabilities = np.concatenate(
    all_probabilities
)


y_pred = (
    probabilities
    >=
    THRESHOLD
).astype(
    np.int8
)


print(
    "\nTotal test flows:",
    f"{len(y_true):,}"
)


# ============================================================
# 8. TEST CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TEST CLASS DISTRIBUTION"
)

print(
    "-" * 60
)


unique_labels, counts = np.unique(
    y_original,
    return_counts=True
)


test_class_distribution = {}


for label, count in zip(
    unique_labels,
    counts
):

    label = int(label)
    count = int(count)

    name = label_names.get(
        label,
        f"Class {label}"
    )


    test_class_distribution[
        str(label)
    ] = {

        "name": name,

        "count": count,
    }


    print(
        f"{label:2d} | "
        f"{name:20s} | "
        f"{count:,}"
    )


# ============================================================
# 9. OVERALL BINARY METRICS
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
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


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


specificity = (
    tn
    /
    (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "FINAL JULY 7 BINARY TEST RESULTS"
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
    f"{precision:.4f}"
)


print(
    "Recall:",
    f"{recall:.4f}"
)


print(
    "F1:",
    f"{f1:.4f}"
)


print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)


print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
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
# 10. PER-ATTACK-FAMILY RECALL
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "UNSEEN ATTACK-FAMILY ANALYSIS"
)

print(
    "-" * 60
)


per_class_results = {}


for label in sorted(
    np.unique(
        y_original
    )
):

    label = int(label)


    # Benign is handled in the binary metrics above.
    if label == 0:

        continue


    class_mask = (
        y_original
        ==
        label
    )


    class_true = np.ones(
        class_mask.sum(),
        dtype=np.int8
    )


    class_pred = y_pred[
        class_mask
    ]


    class_recall = recall_score(
        class_true,
        class_pred,
        zero_division=0
    )


    class_count = int(
        class_mask.sum()
    )


    correctly_detected = int(
        class_pred.sum()
    )


    name = label_names.get(
        label,
        f"Class {label}"
    )


    per_class_results[
        str(label)
    ] = {

        "name": name,

        "flows": class_count,

        "detected": correctly_detected,

        "missed": (
            class_count
            -
            correctly_detected
        ),

        "recall": float(
            class_recall
        ),
    }


    print(
        f"{name:20s} | "
        f"Flows: {class_count:7,} | "
        f"Detected: {correctly_detected:7,} | "
        f"Recall: {class_recall:.4f}"
    )


# ============================================================
# 11. UNSEEN ATTACK SUMMARY
# ============================================================

unseen_attack_labels = [
    1,   # Bot
    2,   # DDoS
    10,  # PortScan
]


print(
    "\n" + "-" * 60
)

print(
    "UNSEEN ATTACK FAMILIES"
)

print(
    "-" * 60
)


unseen_available = []


for label in unseen_attack_labels:

    if label in [
        int(x)
        for x in np.unique(
            y_original
        )
    ]:

        unseen_available.append(
            label
        )


        result = per_class_results[
            str(label)
        ]


        print(
            f"{result['name']:20s} | "
            f"Flows: {result['flows']:,} | "
            f"Recall: {result['recall']:.4f}"
        )


if unseen_available:

    unseen_correct = sum(
        per_class_results[
            str(label)
        ]["detected"]
        for label
        in unseen_available
    )


    unseen_total = sum(
        per_class_results[
            str(label)
        ]["flows"]
        for label
        in unseen_available
    )


    unseen_recall = (
        unseen_correct
        /
        unseen_total
    )


else:

    unseen_recall = 0.0
    unseen_total = 0
    unseen_correct = 0


print(
    "\nAggregate unseen-family recall:",
    f"{unseen_recall:.4f}"
)


# ============================================================
# 12. TEST SCORE DISTRIBUTION
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TEST PROBABILITY DISTRIBUTION"
)

print(
    "-" * 60
)


print(
    "Min:",
    f"{probabilities.min():.6f}"
)


print(
    "Mean:",
    f"{probabilities.mean():.6f}"
)


print(
    "Median:",
    f"{np.median(probabilities):.6f}"
)


print(
    "90th percentile:",
    f"{np.percentile(probabilities, 90):.6f}"
)


print(
    "99th percentile:",
    f"{np.percentile(probabilities, 99):.6f}"
)


print(
    "Max:",
    f"{probabilities.max():.6f}"
)


# ============================================================
# 13. FINAL ATTACK RATE
# ============================================================

actual_attack_rate = (
    y_true.mean()
)


predicted_attack_rate = (
    y_pred.mean()
)


print(
    "\nActual attack rate:",
    f"{actual_attack_rate:.4%}"
)


print(
    "Predicted attack rate:",
    f"{predicted_attack_rate:.4%}"
)


# ============================================================
# 14. SAVE FINAL RESULTS
# ============================================================

final_results = {

    "experiment": (
        "Frozen temporal test evaluation"
    ),

    "test_date": [
        "2017-07-07"
    ],

    "model_file": MODEL_FILE,

    "checkpoint_epoch": int(
        checkpoint["epoch"]
    ),

    "threshold_file": THRESHOLD_FILE,

    "threshold": float(
        THRESHOLD
    ),

    "threshold_selection": {
        "dataset": "July 6 validation",
        "metric": threshold_data[
            "selection_metric"
        ],

        "validation_f1": float(
            threshold_data[
                "selected_f1"
            ]
        ),

        "validation_precision": float(
            threshold_data[
                "selected_precision"
            ]
        ),

        "validation_recall": float(
            threshold_data[
                "selected_recall"
            ]
        ),
    },


    "test_metrics": {

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "specificity": float(
            specificity
        ),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),
    },


    "test_class_distribution":
        test_class_distribution,


    "per_attack_family":
        per_class_results,


    "unseen_attack_families": {

        "labels":
            unseen_available,

        "total_flows":
            int(unseen_total),

        "correctly_detected":
            int(unseen_correct),

        "aggregate_recall":
            float(unseen_recall),
    },


    "actual_attack_rate":
        float(actual_attack_rate),

    "predicted_attack_rate":
        float(predicted_attack_rate),


    "test_was_frozen":
        True,

    "test_used_for_model_selection":
        False,

    "test_used_for_threshold_selection":
        False,
}


os.makedirs(
    "results",
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


print(
    "\nSaved final test results:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL FROZEN TEST EVALUATION COMPLETE"
)

print("=" * 60)

print(
    "\nJuly 7 was evaluated exactly once "
    "with the frozen model and threshold."
)

print(
    "No test-based tuning was performed."
)

print("=" * 60)