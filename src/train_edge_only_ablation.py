import os
import json
import glob

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
# EDGE-ONLY ABLATION
# ============================================================

TRAIN_DIR = (
    "results/temporal_graphs/train"
)

VAL_DIR = (
    "results/temporal_graphs/validation"
)

TEST_DIR = (
    "results/temporal_graphs/test"
)

OUTPUT_MODEL = (
    "models/graphshield_edge_only_ablation.pt"
)

OUTPUT_RESULTS = (
    "results/graphshield_edge_only_ablation.json"
)


SEED = 42

EPOCHS = 3

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

DROPOUT = 0.20

HIDDEN_DIM = 64


# ============================================================
# DEVICE
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


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
print("GRAPHSHIELD AI - EDGE-ONLY ABLATION")
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 1. GRAPH FILES
# ============================================================

def get_batch_files(
    directory
):

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


# ============================================================
# 2. CLASS BALANCE
# ============================================================

with open(
    "results/temporal_graphs/metadata.json",
    "r"
) as f:

    metadata = json.load(f)


train_labels = metadata[
    "splits"
]["train"]["label_counts"]


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
# 3. EDGE-ONLY MODEL
# ============================================================

class EdgeOnlyMLP(
    nn.Module
):

    """
    Ablation model.

    IMPORTANT:
    It receives only the 66 flow-level edge features.

    It does NOT use:

    - node features
    - edge_index
    - graph message passing
    - network topology
    - historical node embeddings
    """


    def __init__(
        self
    ):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                66,
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
        edge_attr
    ):

        return self.classifier(
            edge_attr
        ).squeeze(-1)


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = EdgeOnlyMLP().to(
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
# 5. LOSS
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
# 6. TRAINING
# ============================================================

def train_epoch():

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


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = model(
                edge_attr
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


            num_edges = y.numel()


            total_loss += (
                loss.item()
                *
                num_edges
            )


            total_edges += (
                num_edges
            )


            processed_snapshots += 1


            if (
                processed_snapshots % 50
                == 0
            ):

                print(
                    f"  Snapshots: "
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
# 7. INFERENCE FUNCTION
# ============================================================

@torch.no_grad()
def infer(
    files
):

    model.eval()


    all_labels = []

    all_probabilities = []


    total_edges = 0


    for batch_file in files:

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            edge_attr = snapshot[
                "edge_attr"
            ].to(
                DEVICE
            )


            y = (
                snapshot["y"] > 0
            ).long()


            logits = model(
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
# 8. TRAINING LOOP
# ============================================================

history = []


print(
    "\n" + "=" * 60
)

print(
    "STARTING EDGE-ONLY TRAINING"
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


    train_loss = train_epoch()


    print(
        "\nTraining loss:",
        f"{train_loss:.6f}"
    )


    y_val, val_probabilities = infer(
        val_files
    )


    val_roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )


    val_pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )


    print(
        "\nValidation ROC-AUC:",
        f"{val_roc_auc:.4f}"
    )


    print(
        "Validation PR-AUC:",
        f"{val_pr_auc:.4f}"
    )


    history.append(
        {
            "epoch": epoch,

            "train_loss":
                float(train_loss),

            "validation_roc_auc":
                float(val_roc_auc),

            "validation_pr_auc":
                float(val_pr_auc),
        }
    )


# ============================================================
# 9. VALIDATION THRESHOLD SELECTION
# ============================================================

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


best_result = max(
    threshold_results,
    key=lambda item: (
        item["f1"],
        item["precision"]
    )
)


THRESHOLD = (
    best_result[
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


# ============================================================
# 10. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "model_config":
            {
                "edge_features": 66,

                "hidden_dim":
                    HIDDEN_DIM,

                "dropout":
                    DROPOUT,
            },

        "task":
            "edge_only_binary_ablation",

        "threshold":
            float(THRESHOLD),

        "seed":
            SEED,
    },
    OUTPUT_MODEL
)


print(
    "\nSaved ablation model:"
)

print(
    OUTPUT_MODEL
)


# ============================================================
# 11. FINAL TEST
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FROZEN JULY 7 TEST"
)

print("=" * 60)

print(
    "Threshold:",
    f"{THRESHOLD:.3f}"
)

print(
    "No test-based threshold tuning."
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
    "EDGE-ONLY ABLATION TEST RESULTS"
)

print(
    "-" * 60
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
# 12. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "Edge-only graph ablation",

    "description":
        (
            "66 scaled flow features only; "
            "no node history, topology, or "
            "GNN message passing."
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

    "training_edges":
        1_668_475,

    "validation_edges":
        458_825,

    "test_edges":
        703_240,

    "threshold":
        float(THRESHOLD),

    "validation_metrics":
        {
            "roc_auc":
                float(val_roc_auc),

            "pr_auc":
                float(val_pr_auc),

            "precision":
                float(
                    best_result[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    best_result[
                        "recall"
                    ]
                ),

            "f1":
                float(
                    best_result[
                        "f1"
                    ]
                ),
        },


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


    "history":
        history,

    "test_was_used_for_tuning":
        False,
}


with open(
    OUTPUT_RESULTS,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nSaved ablation results:"
)

print(
    OUTPUT_RESULTS
)


print(
    "\n" + "=" * 60
)

print(
    "EDGE-ONLY ABLATION COMPLETE"
)

print("=" * 60)