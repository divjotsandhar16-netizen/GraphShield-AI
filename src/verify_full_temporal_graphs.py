import os
import json
import joblib
import torch
import numpy as np


# ============================================================
# GRAPHSHIELD AI
# FULL TEMPORAL GRAPH DATASET VERIFICATION
# ============================================================

ROOT = "results/temporal_graphs_10min"

NODE_SCALER_FILE = (
    "data/processed/graphshield_node_scaler.joblib"
)


EXPECTED = {
    "train": {
        "windows": 152,
        "edges": 1_668_475,
    },

    "validation": {
        "windows": 50,
        "edges": 458_825,
    },

    "test": {
        "windows": 50,
        "edges": 703_240,
    },
}


print("=" * 60)
print("GRAPHSHIELD AI - FULL TEMPORAL GRAPH VERIFICATION")
print("=" * 60)


# ============================================================
# 1. LOAD METADATA
# ============================================================

metadata_file = os.path.join(
    ROOT,
    "metadata.json"
)


with open(
    metadata_file,
    "r"
) as f:

    metadata = json.load(f)


print("\nMetadata loaded successfully.")


# ============================================================
# 2. LOAD NODE SCALER
# ============================================================

print(
    "\nLoading node scaler..."
)


node_scaler_package = joblib.load(
    NODE_SCALER_FILE
)

node_scaler = (
    node_scaler_package["scaler"]
)


print(
    "Node scaler loaded successfully."
)


# ============================================================
# 3. VERIFY SPLIT METADATA
# ============================================================

print("\n" + "-" * 60)
print("SPLIT METADATA")
print("-" * 60)


metadata_checks = True


for split_name, expected in EXPECTED.items():

    actual = metadata[
        "splits"
    ][split_name]


    windows = actual[
        "windows"
    ]

    edges = actual[
        "edges"
    ]


    print(
        f"\n{split_name.upper()}"
    )


    print(
        "Expected windows:",
        expected["windows"]
    )


    print(
        "Actual windows:",
        windows
    )


    print(
        "Expected edges:",
        f'{expected["edges"]:,}'
    )


    print(
        "Actual edges:",
        f"{edges:,}"
    )


    if windows != expected["windows"]:

        metadata_checks = False


    if edges != expected["edges"]:

        metadata_checks = False


print(
    "\nSplit metadata valid:",
    metadata_checks
)


# ============================================================
# 4. GET BATCH FILES
# ============================================================

def get_batch_files(
    split_name
):

    directory = os.path.join(
        ROOT,
        split_name
    )


    files = [
        os.path.join(
            directory,
            filename
        )

        for filename
        in os.listdir(directory)

        if filename.endswith(".pt")
    ]


    return sorted(files)


# ============================================================
# 5. VERIFY EACH SPLIT
# ============================================================

all_checks = metadata_checks


for split_name in [
    "train",
    "validation",
    "test"
]:

    print("\n" + "=" * 60)

    print(
        f"VERIFYING {split_name.upper()}"
    )

    print("=" * 60)


    batch_files = get_batch_files(
        split_name
    )


    print(
        "Batch files:",
        len(batch_files)
    )


    total_snapshots = 0
    total_edges = 0
    total_nodes = 0

    first_snapshot = None

    last_window = None

    split_valid = True


    # ========================================================
    # LOAD EVERY BATCH
    # ========================================================

    for batch_file in batch_files:

        print(
            "\nLoading:",
            batch_file
        )


        snapshots = torch.load(
            batch_file,
            map_location="cpu",
            weights_only=False
        )


        print(
            "Snapshots:",
            len(snapshots)
        )


        for snapshot in snapshots:

            total_snapshots += 1


            x = snapshot["x"]

            edge_index = (
                snapshot["edge_index"]
            )

            edge_attr = (
                snapshot["edge_attr"]
            )

            y = snapshot["y"]


            # ------------------------------------------------
            # Save first snapshot
            # ------------------------------------------------

            if first_snapshot is None:

                first_snapshot = snapshot


            # ------------------------------------------------
            # Node features
            # ------------------------------------------------

            if x.ndim != 2:

                print(
                    "ERROR: Node features "
                    "are not a 2D matrix."
                )

                split_valid = False


            elif x.shape[1] != 11:

                print(
                    "ERROR: Node feature "
                    "count != 11"
                )

                split_valid = False


            # ------------------------------------------------
            # Edge features
            # ------------------------------------------------

            if edge_attr.ndim != 2:

                print(
                    "ERROR: Edge features "
                    "are not a 2D matrix."
                )

                split_valid = False


            elif edge_attr.shape[1] != 66:

                print(
                    "ERROR: Edge feature "
                    "count != 66"
                )

                split_valid = False


            # ------------------------------------------------
            # Edge index
            # ------------------------------------------------

            if edge_index.shape[0] != 2:

                print(
                    "ERROR: edge_index "
                    "shape is not [2, E]"
                )

                split_valid = False


            # ------------------------------------------------
            # Edge / feature alignment
            # ------------------------------------------------

            if (
                edge_index.shape[1]
                !=
                edge_attr.shape[0]
            ):

                print(
                    "ERROR: Edge / "
                    "edge-feature mismatch"
                )

                split_valid = False


            # ------------------------------------------------
            # Edge / label alignment
            # ------------------------------------------------

            if (
                edge_index.shape[1]
                !=
                y.shape[0]
            ):

                print(
                    "ERROR: Edge / "
                    "label mismatch"
                )

                split_valid = False


            # ------------------------------------------------
            # Edge index validity
            # ------------------------------------------------

            if edge_index.numel() > 0:

                min_index = int(
                    edge_index.min().item()
                )

                max_index = int(
                    edge_index.max().item()
                )


                if min_index < 0:

                    print(
                        "ERROR: Negative "
                        "node index found."
                    )

                    split_valid = False


                if max_index >= x.shape[0]:

                    print(
                        "ERROR: Edge references "
                        "nonexistent node."
                    )

                    split_valid = False


            # ------------------------------------------------
            # Self-loop check
            # ------------------------------------------------

            self_loops = (
                edge_index[0]
                ==
                edge_index[1]
            ).sum().item()


            if self_loops != 0:

                print(
                    "ERROR: Self-loops found:",
                    self_loops
                )

                split_valid = False


            # ------------------------------------------------
            # Numerical checks
            # ------------------------------------------------

            if not torch.isfinite(x).all():

                print(
                    "ERROR: Non-finite "
                    "node features."
                )

                split_valid = False


            if not torch.isfinite(
                edge_attr
            ).all():

                print(
                    "ERROR: Non-finite "
                    "edge features."
                )

                split_valid = False


            # ------------------------------------------------
            # Label range
            # ------------------------------------------------

            if y.numel() > 0:

                minimum_label = int(
                    y.min().item()
                )

                maximum_label = int(
                    y.max().item()
                )


                if (
                    minimum_label < 0
                    or
                    maximum_label > 14
                ):

                    print(
                        "ERROR: Label outside "
                        "0-14 range."
                    )

                    split_valid = False


            # ------------------------------------------------
            # Chronological ordering
            # ------------------------------------------------

            current_window = np.datetime64(
                snapshot["window_start"]
            )


            if last_window is not None:

                if current_window < last_window:

                    print(
                        "ERROR: Temporal windows "
                        "are not chronological."
                    )

                    split_valid = False


            last_window = current_window


            # ------------------------------------------------
            # Counters
            # ------------------------------------------------

            total_edges += (
                edge_index.shape[1]
            )

            total_nodes += (
                x.shape[0]
            )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    expected_split = EXPECTED[
        split_name
    ]


    print("\n" + "-" * 60)

    print(
        f"{split_name.upper()} SUMMARY"
    )

    print("-" * 60)


    print(
        "Snapshots:",
        total_snapshots
    )


    print(
        "Expected:",
        expected_split["windows"]
    )


    print(
        "Edges:",
        f"{total_edges:,}"
    )


    print(
        "Expected:",
        f'{expected_split["edges"]:,}'
    )


    print(
        "Snapshot-node occurrences:",
        f"{total_nodes:,}"
    )


    # ========================================================
    # FIRST SNAPSHOT HISTORY CHECK
    # ========================================================

    if first_snapshot is not None:

        first_x = first_snapshot["x"]


        print(
            "\nFirst snapshot:",
            first_snapshot[
                "window_start"
            ]
        )


        print(
            "First snapshot node shape:",
            tuple(first_x.shape)
        )


        # ----------------------------------------------------
        # TRAINING FIRST SNAPSHOT
        # ----------------------------------------------------

        if split_name == "train":

            # The RAW historical vector is all zeros.
            #
            # But saved x has already gone through:
            #
            # log1p -> StandardScaler
            #
            # Therefore zero history becomes:
            #
            # (0 - scaler.mean_) / scaler.scale_

            raw_zero_features = np.zeros(
                (1, 11),
                dtype=np.float64
            )


            expected_scaled_zero = (
                node_scaler.transform(
                    raw_zero_features
                )
            )


            expected_scaled_zero = torch.tensor(
                expected_scaled_zero,
                dtype=torch.float32
            )


            expected_scaled_zero = (
                expected_scaled_zero
                .repeat(
                    first_x.shape[0],
                    1
                )
            )


            first_snapshot_matches_expected = (
                torch.allclose(
                    first_x,
                    expected_scaled_zero,
                    atol=1e-5,
                    rtol=1e-5
                )
            )


            print(
                "Training first snapshot "
                "matches scaled-zero history:",
                first_snapshot_matches_expected
            )


            if not first_snapshot_matches_expected:

                split_valid = False


            print(
                "Raw conceptual history for "
                "first snapshot: all zeros"
            )


        # ----------------------------------------------------
        # VALIDATION / TEST
        # ----------------------------------------------------

        else:

            history_nonzero = torch.any(
                first_x != 0
            ).item()


            print(
                f"{split_name.title()} first snapshot "
                "contains prior history:",
                history_nonzero
            )


            if not history_nonzero:

                split_valid = False


    # ========================================================
    # FINAL SPLIT CHECKS
    # ========================================================

    if (
        total_snapshots
        !=
        expected_split[
            "windows"
        ]
    ):

        split_valid = False


    if (
        total_edges
        !=
        expected_split[
            "edges"
        ]
    ):

        split_valid = False


    print(
        "\nSplit verification:",
        split_valid
    )


    all_checks = (
        all_checks
        and
        split_valid
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)


if all_checks:

    print(
        "ALL FULL TEMPORAL GRAPH CHECKS PASSED"
    )

else:

    print(
        "SOME TEMPORAL GRAPH CHECKS FAILED"
    )


print("=" * 60)