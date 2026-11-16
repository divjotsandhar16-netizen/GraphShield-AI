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
)

from torch_geometric.nn import GATv2Conv


# ============================================================
# GRAPHSHIELD AI
# ACTUAL TEMPORAL GNN TRAINING
# ============================================================

TRAIN_DIR = "results/temporal_graphs_1min/train"
VAL_DIR = "results/temporal_graphs_1min/validation"

METADATA_FILE = "results/temporal_graphs_1min/metadata.json"

MODEL_DIR = "models"
RESULTS_DIR = "results"

BEST_MODEL_FILE = (
    "models/graphshield_gnn_1min_best.pt"
)

RESULTS_FILE = (
    "results/graphshield_gnn_1min_training.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 3

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 32

HEADS = 2

DROPOUT = 0.20

GRADIENT_CLIP = 2.0

# None = use every training snapshot.
#
# During debugging you may temporarily set this to 20,
# but the final research run must use None.

MAX_TRAIN_SNAPSHOTS = None

# Validation always uses every validation snapshot.

MAX_VAL_SNAPSHOTS = None


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

elif hasattr(torch.backends, "mps") and (
    torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("GRAPHSHIELD AI - TEMPORAL GNN TRAINING")
print("=" * 60)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD METADATA
# ============================================================

print("\nLoading graph metadata...")

with open(
    METADATA_FILE,
    "r"
) as f:

    metadata = json.load(f)


TRAIN_EXPECTED_EDGES = metadata[
    "splits"
]["train"]["edges"]

VAL_EXPECTED_EDGES = metadata[
    "splits"
]["validation"]["edges"]


print(
    "Expected training edges:",
    f"{TRAIN_EXPECTED_EDGES:,}"
)

print(
    "Expected validation edges:",
    f"{VAL_EXPECTED_EDGES:,}"
)


# ============================================================
# 2. CALCULATE BINARY CLASS BALANCE
# ============================================================

print(
    "\nCalculating binary class balance..."
)


train_label_counts = metadata[
    "splits"
]["train"]["label_counts"]


benign_count = int(
    train_label_counts.get(
        "0",
        0
    )
)


attack_count = sum(
    int(count)
    for label, count
    in train_label_counts.items()
    if int(label) != 0
)


total_count = (
    benign_count
    +
    attack_count
)


print(
    "Benign training flows:",
    f"{benign_count:,}"
)

print(
    "Attack training flows:",
    f"{attack_count:,}"
)

print(
    "Total training flows:",
    f"{total_count:,}"
)


if attack_count == 0:

    raise RuntimeError(
        "No attack examples found in training data."
    )


pos_weight_value = (
    benign_count
    /
    attack_count
)


print(
    "Positive-class weight:",
    f"{pos_weight_value:.4f}"
)


POS_WEIGHT = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 3. GET GRAPH BATCH FILES
# ============================================================

def get_batch_files(directory):

    files = glob.glob(
        os.path.join(
            directory,
            "*.pt"
        )
    )

    return sorted(files)


train_batch_files = get_batch_files(
    TRAIN_DIR
)

val_batch_files = get_batch_files(
    VAL_DIR
)


print(
    "\nTraining batch files:",
    len(train_batch_files)
)

print(
    "Validation batch files:",
    len(val_batch_files)
)


if len(train_batch_files) == 0:

    raise RuntimeError(
        "No training graph batches found."
    )


if len(val_batch_files) == 0:

    raise RuntimeError(
        "No validation graph batches found."
    )


# ============================================================
# 4. MODEL
# ============================================================

class GraphShieldGNN(nn.Module):

    def __init__(
        self,
        node_features=11,
        edge_features=66,
        hidden_dim=32,
        heads=2,
        dropout=0.20
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

        # source embedding:      32
        # destination embedding: 32
        # edge features:         66
        #
        # total = 130

        self.edge_classifier = nn.Sequential(

            nn.Linear(
                hidden_dim
                +
                hidden_dim
                +
                edge_features,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
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

        # ----------------------------------------------------
        # First message-passing layer
        # ----------------------------------------------------

        x1 = self.gat1(
            x,
            edge_index,
            edge_attr
        )

        x1 = F.elu(x1)


        # ----------------------------------------------------
        # Second message-passing layer
        # ----------------------------------------------------

        x2 = self.gat2(
            x1,
            edge_index,
            edge_attr
        )

        x2 = F.elu(x2)


        x2 = self.node_norm(
            x2
        )


        # ----------------------------------------------------
        # Source / destination embeddings
        # ----------------------------------------------------

        source_nodes = (
            edge_index[0]
        )

        destination_nodes = (
            edge_index[1]
        )


        source_embedding = x2[
            source_nodes
        ]

        destination_embedding = x2[
            destination_nodes
        ]


        # ----------------------------------------------------
        # Edge representation
        # ----------------------------------------------------

        edge_representation = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_attr,
            ],
            dim=1
        )


        # ----------------------------------------------------
        # Binary attack logit
        # ----------------------------------------------------

        logits = self.edge_classifier(
            edge_representation
        ).squeeze(-1)


        return logits, x2


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = GraphShieldGNN(
    node_features=11,
    edge_features=66,
    hidden_dim=HIDDEN_DIM,
    heads=HEADS,
    dropout=DROPOUT
).to(DEVICE)


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
# 6. LOSS AND OPTIMIZER
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
# 7. HELPER FUNCTIONS
# ============================================================

def prepare_snapshot(
    snapshot
):
    """
    Move one temporal graph snapshot to the selected device.
    """

    x = snapshot["x"].to(
        DEVICE,
        non_blocking=True
    )

    edge_index = snapshot[
        "edge_index"
    ].to(
        DEVICE,
        non_blocking=True
    )

    edge_attr = snapshot[
        "edge_attr"
    ].to(
        DEVICE,
        non_blocking=True
    )


    # Multiclass original labels:
    #
    # 0 = benign
    # >0 = attack
    #
    # Convert them to binary.

    y = (
        snapshot["y"] > 0
    ).float().to(
        DEVICE,
        non_blocking=True
    )


    return (
        x,
        edge_index,
        edge_attr,
        y
    )


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


    # These require both classes.
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")


    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    epoch_number
):

    model.train()


    running_loss = 0.0

    processed_edges = 0

    all_labels = []

    all_probabilities = []


    processed_snapshots = 0


    for batch_file in train_batch_files:

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            if (
                MAX_TRAIN_SNAPSHOTS
                is not None
                and
                processed_snapshots
                >=
                MAX_TRAIN_SNAPSHOTS
            ):

                break


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


            logits, _ = model(
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


            probabilities = torch.sigmoid(
                logits
            )


            num_edges = y.numel()


            running_loss += (
                loss.item()
                *
                num_edges
            )


            processed_edges += (
                num_edges
            )


            all_labels.append(
                y.detach().cpu().numpy()
            )


            all_probabilities.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )


            processed_snapshots += 1


            if (
                processed_snapshots % 25
                == 0
            ):

                print(
                    f"  Epoch {epoch_number} | "
                    f"Snapshots: "
                    f"{processed_snapshots:,} | "
                    f"Edges: "
                    f"{processed_edges:,}"
                )


        if (
            MAX_TRAIN_SNAPSHOTS
            is not None
            and
            processed_snapshots
            >=
            MAX_TRAIN_SNAPSHOTS
        ):

            break


    y_true = np.concatenate(
        all_labels
    )

    probabilities = np.concatenate(
        all_probabilities
    )


    metrics = calculate_metrics(
        y_true,
        probabilities
    )


    metrics["loss"] = (
        running_loss
        /
        processed_edges
    )

    metrics["edges"] = int(
        processed_edges
    )

    metrics["snapshots"] = int(
        processed_snapshots
    )


    return metrics


# ============================================================
# 9. VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()


    running_loss = 0.0

    processed_edges = 0

    all_labels = []

    all_probabilities = []

    processed_snapshots = 0


    for batch_file in val_batch_files:

        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        for snapshot in snapshots:

            if (
                MAX_VAL_SNAPSHOTS
                is not None
                and
                processed_snapshots
                >=
                MAX_VAL_SNAPSHOTS
            ):

                break


            (
                x,
                edge_index,
                edge_attr,
                y
            ) = prepare_snapshot(
                snapshot
            )


            logits, _ = model(
                x,
                edge_index,
                edge_attr
            )


            loss = criterion(
                logits,
                y
            )


            probabilities = torch.sigmoid(
                logits
            )


            num_edges = y.numel()


            running_loss += (
                loss.item()
                *
                num_edges
            )


            processed_edges += (
                num_edges
            )


            all_labels.append(
                y.cpu().numpy()
            )


            all_probabilities.append(
                probabilities.cpu().numpy()
            )


            processed_snapshots += 1


        if (
            MAX_VAL_SNAPSHOTS
            is not None
            and
            processed_snapshots
            >=
            MAX_VAL_SNAPSHOTS
        ):

            break


    y_true = np.concatenate(
        all_labels
    )

    probabilities = np.concatenate(
        all_probabilities
    )


    metrics = calculate_metrics(
        y_true,
        probabilities
    )


    metrics["loss"] = (
        running_loss
        /
        processed_edges
    )

    metrics["edges"] = int(
        processed_edges
    )

    metrics["snapshots"] = int(
        processed_snapshots
    )


    return metrics


# ============================================================
# 10. TRAINING LOOP
# ============================================================

history = []

best_validation_pr_auc = -np.inf

best_epoch = None


print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"EPOCH {epoch} / {EPOCHS}"
    )

    print(
        f"{'=' * 60}"
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_metrics = train_one_epoch(
        epoch
    )


    print(
        "\nTRAIN"
    )

    print(
        f"Loss:      "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"Precision: "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{train_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{train_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{train_metrics['pr_auc']:.4f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics = validate()


    print(
        "\nVALIDATION"
    )

    print(
        f"Loss:      "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{val_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{val_metrics['pr_auc']:.4f}"
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    epoch_record = {

        "epoch": epoch,

        "train": train_metrics,

        "validation": val_metrics,
    }


    history.append(
        epoch_record
    )


    # --------------------------------------------------------
    # Model selection
    #
    # We choose the best epoch using VALIDATION PR-AUC.
    #
    # TEST is never used here.
    # --------------------------------------------------------

    current_pr_auc = (
        val_metrics["pr_auc"]
    )


    if (
        np.isfinite(current_pr_auc)
        and
        current_pr_auc
        >
        best_validation_pr_auc
    ):

        best_validation_pr_auc = (
            current_pr_auc
        )

        best_epoch = epoch


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "model_config": {

                "node_features": 11,

                "edge_features": 66,

                "hidden_dim": HIDDEN_DIM,

                "heads": HEADS,

                "dropout": DROPOUT,
            },

            "epoch": epoch,

            "validation_pr_auc":
                current_pr_auc,

            "seed": SEED,

            "task":
                "binary_edge_classification",

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
        }


        torch.save(
            checkpoint,
            BEST_MODEL_FILE
        )


        print(
            "\n*** New best model saved ***"
        )

        print(
            "Best validation PR-AUC:",
            f"{current_pr_auc:.4f}"
        )


# ============================================================
# 11. SAVE TRAINING RESULTS
# ============================================================

results = {

    "model": "GraphShieldGNN",

    "task":
        "binary_edge_classification",

    "seed": SEED,

    "device": str(DEVICE),

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "hidden_dim": HIDDEN_DIM,

    "heads": HEADS,

    "dropout": DROPOUT,

    "gradient_clip": GRADIENT_CLIP,

    "positive_class_weight":
        pos_weight_value,

    "best_epoch":
        best_epoch,

    "best_validation_pr_auc":
        (
            None
            if best_epoch is None
            else best_validation_pr_auc
        ),

    "train_dates": [
        "2017-07-03",
        "2017-07-04",
        "2017-07-05"
    ],

    "validation_dates": [
        "2017-07-06"
    ],

    "test_dates": [
        "2017-07-07"
    ],

    "test_used_during_training":
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


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GRAPHSHIELD GNN TRAINING COMPLETE")
print("=" * 60)


print(
    "\nBest epoch:",
    best_epoch
)


if best_epoch is not None:

    print(
        "Best validation PR-AUC:",
        f"{best_validation_pr_auc:.4f}"
    )


print(
    "\nBest model:",
    BEST_MODEL_FILE
)


print(
    "Training results:",
    RESULTS_FILE
)


print(
    "\nTEST SET:"
)

print(
    "July 7 was NOT used during training "
    "or model selection."
)


print("\n" + "=" * 60)