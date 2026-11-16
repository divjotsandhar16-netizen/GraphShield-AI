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
# PARAMETER-MATCHED NODE-HISTORY + EDGE MLP
# ============================================================

TRAIN_DIR = "results/temporal_graphs/train"
VAL_DIR = "results/temporal_graphs/validation"
TEST_DIR = "results/temporal_graphs/test"

MODEL_FILE = (
    "models/graphshield_node_history_edge_mlp_matched.pt"
)

RESULTS_FILE = (
    "results/graphshield_node_history_edge_mlp_matched.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 3

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

DROPOUT = 0.20

# 88 -> 176 -> 64 -> 1
#
# Parameter count:
#
# 88*176 + 176
# + 176*64 + 64
# + 64*1 + 1
#
# = 27,057
#
# Very close to GraphShield GNN:
# 27,041 parameters

HIDDEN_1 = 176

HIDDEN_2 = 64


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

    DEVICE = torch.device(
        "cuda"
    )

elif (
    hasattr(
        torch.backends,
        "mps"
    )
    and
    torch.backends.mps.is_available()
):

    DEVICE = torch.device(
        "mps"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print("=" * 60)
print(
    "GRAPHSHIELD AI - PARAMETER-MATCHED "
    "NODE HISTORY + EDGE MLP"
)
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. GRAPH FILES
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

    metadata = json.load(
        f
    )


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
# 3. PARAMETER-MATCHED MLP
# ============================================================

class MatchedNodeHistoryEdgeMLP(
    nn.Module
):

    """
    Parameter-matched non-graph baseline.

    Inputs:
        11 source historical features
        11 destination historical features
        66 current flow features

    Total:
        88 features

    Architecture:
        88 -> 176 -> 64 -> 1

    This has approximately the same number of trainable
    parameters as the 27,041-parameter GraphShield GNN.

    No graph message passing is performed.
    """


    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                88,
                HIDDEN_1
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_2,
                1
            )
        )


    def forward(
        self,
        edge_input
    ):

        return self.network(
            edge_input
        ).squeeze(-1)


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = (
    MatchedNodeHistoryEdgeMLP()
    .to(DEVICE)
)


print(
    "\n" + "-" * 60
)

print(
    "PARAMETER-MATCHED MODEL"
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


print(
    "GraphShield GNN parameters:",
    "27,041"
)


print(
    "Parameter difference:",
    parameter_count - 27_041
)


# ============================================================
# 5. LOSS AND OPTIMIZER
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
# 6. BUILD 88-DIMENSIONAL EDGE INPUT
# ============================================================

def prepare_snapshot(
    snapshot
):

    x = snapshot[
        "x"
    ].to(
        DEVICE
    )


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


    edge_attr = snapshot[
        "edge_attr"
    ].to(
        DEVICE
    )


    edge_input = torch.cat(
        [
            source_history,
            destination_history,
            edge_attr
        ],
        dim=1
    )


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
# 7. METRICS
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

            edge_input, y = (
                prepare_snapshot(
                    snapshot
                )
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
                2.0
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
                ==
                0
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

            edge_input, y = (
                prepare_snapshot(
                    snapshot
                )
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


    return (
        np.concatenate(
            all_labels
        ),

        np.concatenate(
            all_probabilities
        )
    )


# ============================================================
# 10. TRAINING LOOP
# ============================================================

history = []

best_pr_auc = -np.inf

best_epoch = None


print(
    "\n" + "=" * 60
)

print(
    "STARTING PARAMETER-MATCHED TRAINING"
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


    y_val, val_probabilities = (
        infer(
            val_files
        )
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
    # Best epoch based on validation PR-AUC
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

                "epoch":
                    epoch,

                "validation_pr_auc":
                    best_pr_auc,

                "input_features":
                    88,

                "hidden_1":
                    HIDDEN_1,

                "hidden_2":
                    HIDDEN_2,

                "dropout":
                    DROPOUT,

                "ablation":
                    "parameter_matched_history_edge_mlp",
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


print(
    "Test will be evaluated once "
    "with this frozen threshold."
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
    if (tn + fp) > 0
    else 0.0
)


print(
    "\n" + "-" * 60
)

print(
    "PARAMETER-MATCHED MLP TEST RESULTS"
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
        "Parameter-matched node-history + edge MLP",

    "description":
        (
            "Same 88 input features as the smaller "
            "history+edge MLP, but model capacity is "
            "matched closely to GraphShield."
        ),

    "input_features":
        88,

    "hidden_layers":
        [
            HIDDEN_1,
            HIDDEN_2
        ],

    "trainable_parameters":
        parameter_count,

    "graphshield_parameters":
        27_041,

    "parameter_difference":
        parameter_count - 27_041,

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
    "PARAMETER-MATCHED ABLATION COMPLETE"
)

print("=" * 60)