import os
import json
import glob
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# GRAPHSHIELD AI
# ABLATION 3
# NODE HISTORY + FLOW FEATURES
# WITHOUT GNN MESSAGE PASSING
# ============================================================

TRAIN_DIR = "results/temporal_graphs/train"
VAL_DIR = "results/temporal_graphs/validation"
TEST_DIR = "results/temporal_graphs/test"

MODEL_FILE = (
    "models/graphshield_node_history_edge_mlp.pt"
)

RESULTS_FILE = (
    "results/graphshield_node_history_edge_mlp.json"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

EPOCHS = 3

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 64

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
    hasattr(
        torch.backends,
        "mps"
    )
    and
    torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print(
    "GRAPHSHIELD AI - NODE HISTORY + EDGE MLP"
)
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. GET GRAPH FILES
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
# 2. LOAD CLASS BALANCE
# ============================================================

with open(
    "results/temporal_graphs/metadata.json",
    "r"
) as f:

    metadata = json.load(f)


train_labels = (
    metadata[
        "splits"
    ][
        "train"
    ][
        "label_counts"
    ]
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


POS_WEIGHT = torch.tensor(
    positive_weight,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 3. MODEL
# ============================================================

class NodeHistoryEdgeMLP(nn.Module):

    """
    Ablation model.

    Uses:

        source node historical features = 11
        destination node historical features = 11
        current flow features = 66

    Total:

        11 + 11 + 66 = 88

    DOES NOT USE:

        edge_index
        graph connectivity
        GATv2
        message passing
        neighborhood aggregation
    """


    def __init__(
        self
    ):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                88,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                1
            )
        )


    def forward(
        self,
        edge_input
    ):

        return self.classifier(
            edge_input
        ).squeeze(-1)


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = NodeHistoryEdgeMLP().to(
    DEVICE
)


print(
    "\n" + "-" * 60
)

print(
    "ABLATION MODEL"
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
# 5. LOSS / OPTIMIZER
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
# 6. PREPARE EDGE INPUT
# ============================================================

def prepare_snapshot(
    snapshot
):

    # --------------------------------------------------------
    # Node historical features
    # --------------------------------------------------------

    x = snapshot[
        "x"
    ].to(
        DEVICE
    )


    # --------------------------------------------------------
    # Edge connectivity
    #
    # We use edge_index ONLY to select the two endpoint
    # historical feature vectors.
    #
    # There is NO message passing.
    # --------------------------------------------------------

    edge_index = snapshot[
        "edge_index"
    ]


    source_nodes = (
        edge_index[0]
    )

    destination_nodes = (
        edge_index[1]
    )


    source_history = x[
        source_nodes
    ]


    destination_history = x[
        destination_nodes
    ]


    # --------------------------------------------------------
    # Current 66-dimensional flow features
    # --------------------------------------------------------

    edge_attr = snapshot[
        "edge_attr"
    ].to(
        DEVICE
    )


    # --------------------------------------------------------
    # Combine:
    #
    # 11 source history
    # 11 destination history
    # 66 flow features
    #
    # = 88
    # --------------------------------------------------------

    edge_input = torch.cat(
        [
            source_history,
            destination_history,
            edge_attr
        ],
        dim=1
    )


    # --------------------------------------------------------
    # Binary target
    # --------------------------------------------------------

    y = (
        snapshot["y"] > 0
    ).float().to(
        DEVICE
    )


    return (
        edge_input,
        y
    )


# ============================================================
# 7. METRIC FUNCTION
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
# 8. TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    epoch_number
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

            edge_input, y = prepare_snapshot(
                snapshot
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = model(
                edge_input
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


            number_of_edges = (
                y.numel()
            )


            total_loss += (
                loss.item()
                *
                number_of_edges
            )


            total_edges += (
                number_of_edges
            )


            processed_snapshots += 1


            if (
                processed_snapshots % 50
                ==
                0
            ):

                print(
                    f"  Epoch {epoch_number} | "
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
# 9. INFERENCE
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

            edge_input, y = prepare_snapshot(
                snapshot
            )


            logits = model(
                edge_input
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


    y_true = np.concatenate(
        all_labels
    )


    probabilities = np.concatenate(
        all_probabilities
    )


    return (
        y_true,
        probabilities
    )


# ============================================================
# 10. TRAINING
# ============================================================

history = []

best_pr_auc = -np.inf

best_epoch = None


print(
    "\n" + "=" * 60
)

print(
    "STARTING TRAINING"
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


    # --------------------------------------------------------
    # Best epoch selected using validation PR-AUC
    # --------------------------------------------------------

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

                "model_config":
                    {
                        "input_features": 88,

                        "hidden_dim":
                            HIDDEN_DIM,

                        "dropout":
                            DROPOUT,
                    },

                "epoch":
                    epoch,

                "validation_pr_auc":
                    best_pr_auc,

                "ablation":
                    "node_history_plus_edge_features_without_message_passing",
            },

            MODEL_FILE
        )


        print(
            "\n*** New best model saved ***"
        )


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# ============================================================
# 12. VALIDATION THRESHOLD
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
# 13. FROZEN JULY 7 TEST
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FROZEN JULY 7 TEST"
)

print("=" * 60)


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
    if (tn + fp) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "NODE-HISTORY + EDGE MLP TEST RESULTS"
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
# 14. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        (
            "Node-history + edge-feature MLP "
            "without graph message passing"
        ),

    "description":
        (
            "Uses source historical node features, "
            "destination historical node features, "
            "and 66 current flow features. "
            "No GATv2 or neighborhood aggregation."
        ),

    "train_dates":
        [
            "2017-07-03",
            "2017-07-04",
            "2017-07-05"
        ],

    "validation_dates":
        [
            "2017-07-06"
        ],

    "test_dates":
        [
            "2017-07-07"
        ],

    "input_features":
        88,

    "training_edges":
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

    "validation_threshold_f1":
        float(
            best_threshold_result[
                "f1"
            ]
        ),

    "test_metrics":
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

    "test_was_used_for_tuning":
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
    "NODE-HISTORY + EDGE MLP ABLATION COMPLETE"
)

print("=" * 60)