import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler


# ============================================================
# GRAPHSHIELD AI
# FULL TEMPORAL GRAPH DATASET BUILDER
# ============================================================

SPLIT_FILES = [
    "data/processed/train.csv",
    "data/processed/validation.csv",
    "data/processed/test.csv",
]

PREPROCESSOR_FILE = (
    "data/processed/graphshield_preprocessor.joblib"
)

OUTPUT_ROOT = "results/temporal_graphs_10min"

WINDOW_MINUTES = 10

# Number of snapshots stored in one .pt file
SNAPSHOTS_PER_BATCH = 100

TRAIN_DATES = {
    "2017-07-03",
    "2017-07-04",
    "2017-07-05",
}

VALIDATION_DATES = {
    "2017-07-06",
}

TEST_DATES = {
    "2017-07-07",
}


print("=" * 60)
print("GRAPHSHIELD AI - FULL TEMPORAL GRAPH BUILDER")
print("=" * 60)


# ============================================================
# 1. LOAD TRAINING-ONLY PREPROCESSOR
# ============================================================

print("\nLoading training-only preprocessor...")

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)

feature_columns = preprocessor["feature_columns"]
imputation_values = preprocessor["imputation_values"]
edge_scaler = preprocessor["scaler"]

imputation_series = pd.Series(
    imputation_values,
    dtype=np.float64
)

print(
    f"Model features: {len(feature_columns)}"
)

if len(feature_columns) != 66:
    raise ValueError(
        f"Expected 66 model features, "
        f"found {len(feature_columns)}"
    )

print("66 model features confirmed.")


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

for split_name in [
    "train",
    "validation",
    "test"
]:
    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            split_name
        ),
        exist_ok=True
    )


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Label",
]

use_columns = (
    required_columns
    +
    feature_columns
)


# ============================================================
# 4. DATA QUALITY POLICY
# ============================================================

non_negative_features = [
    "Flow Duration",

    "Total Fwd Packets",
    "Total Backward Packets",

    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",

    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",

    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",

    "Flow Bytes/s",
    "Flow Packets/s",

    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",

    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",

    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",

    "Fwd Header Length",
    "Bwd Header Length",

    "Fwd Packets/s",
    "Bwd Packets/s",

    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",

    "Average Packet Size",

    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",

    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",

    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",

    "act_data_pkt_fwd",
    "min_seg_size_forward",

    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",

    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

sentinel_minus_one_features = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


print(
    f"\nNon-negative feature rules: "
    f"{len(non_negative_features)}"
)

print(
    "Sentinel -1 features:",
    sentinel_minus_one_features
)


# ============================================================
# 5. CLEAN FEATURES
# ============================================================

def clean_features(chunk):
    """
    Apply the established preprocessing policy.

    Important:
    - No labels are used.
    - No future data is used.
    - Imputation values come from the training-only
      preprocessor.
    """

    chunk = chunk.copy()

    # Convert all 66 model features to numeric
    for column in feature_columns:
        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )

    # Replace +/- infinity
    chunk[feature_columns] = (
        chunk[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Replace sentinel -1 values
    for column in sentinel_minus_one_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] == -1,
                column
            ] = np.nan

    # Replace invalid negative values
    for column in non_negative_features:

        if column in chunk.columns:

            chunk.loc[
                chunk[column] < 0,
                column
            ] = np.nan

    # Training-only median imputation
    chunk[feature_columns] = (
        chunk[feature_columns]
        .fillna(imputation_series)
    )

    return chunk


# ============================================================
# 6. LOAD SPECIFIC TEMPORAL PERIOD
# ============================================================

def load_temporal_rows(
    dates,
    split_name
):
    """
    Load rows from all three random-split CSVs,
    then select only the requested calendar dates.

    This reconstructs the intended temporal experiment.
    """

    print(
        f"\nLoading {split_name.upper()} data..."
    )

    print(
        "Required dates:",
        sorted(dates)
    )

    parts = []

    total_rows_loaded = 0


    for split_file in SPLIT_FILES:

        print(
            f"  Reading: {split_file}"
        )

        chunk = pd.read_csv(
            split_file,
            usecols=use_columns,
            low_memory=False
        )

        total_rows_loaded += len(chunk)

        # Parse timestamps
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

        # Keep requested dates
        date_mask = (
            chunk["Timestamp"]
            .dt.strftime("%Y-%m-%d")
            .isin(dates)
        )

        chunk = chunk[
            date_mask
        ].copy()

        if len(chunk) == 0:
            continue

        chunk = clean_features(
            chunk
        )

        parts.append(
            chunk
        )


    if not parts:
        raise RuntimeError(
            f"No rows found for {split_name}."
        )


    data = pd.concat(
        parts,
        ignore_index=True
    )


    # Chronological ordering
    data = data.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )


    print(
        f"  Total source rows read: "
        f"{total_rows_loaded:,}"
    )

    print(
        f"  {split_name.upper()} rows: "
        f"{len(data):,}"
    )

    print(
        f"  Earliest: "
        f"{data['Timestamp'].iloc[0]}"
    )

    print(
        f"  Latest: "
        f"{data['Timestamp'].iloc[-1]}"
    )


    return data


# ============================================================
# 7. NODE HISTORY STATE
# ============================================================

node_state = {}


def empty_state():
    """
    State maintained for each network entity.
    """

    return {
        "out_flow_count": 0,
        "in_flow_count": 0,

        "unique_destinations": set(),
        "unique_sources": set(),

        "out_bytes": 0.0,
        "in_bytes": 0.0,

        "out_packets": 0.0,
        "in_packets": 0.0,

        "out_unique_ports": set(),
        "in_unique_ports": set(),

        "protocols": set(),
    }


def get_state(node):

    if node not in node_state:

        node_state[node] = empty_state()

    return node_state[node]


def node_feature_vector(node):
    """
    Extract historical node information.

    CRITICAL:
    This is called BEFORE the current window updates
    node_state.
    """

    state = get_state(node)

    return [
        state["out_flow_count"],
        state["in_flow_count"],

        len(
            state["unique_destinations"]
        ),

        len(
            state["unique_sources"]
        ),

        state["out_bytes"],
        state["in_bytes"],

        state["out_packets"],
        state["in_packets"],

        len(
            state["out_unique_ports"]
        ),

        len(
            state["in_unique_ports"]
        ),

        len(
            state["protocols"]
        ),
    ]


# ============================================================
# 8. NODE SCALER
# ============================================================

node_scaler = StandardScaler()


def update_node_state_from_row(row):
    """
    Update node history AFTER the current flow has been
    represented.

    Labels are never used here.
    """

    source = row["Source IP"]
    destination = row["Destination IP"]

    source_state = get_state(source)
    destination_state = get_state(destination)


    # --------------------------------------------------------
    # Flow values
    # --------------------------------------------------------

    fwd_bytes = pd.to_numeric(
        row["Total Length of Fwd Packets"],
        errors="coerce"
    )

    bwd_bytes = pd.to_numeric(
        row["Total Length of Bwd Packets"],
        errors="coerce"
    )

    fwd_packets = pd.to_numeric(
        row["Total Fwd Packets"],
        errors="coerce"
    )

    bwd_packets = pd.to_numeric(
        row["Total Backward Packets"],
        errors="coerce"
    )

    source_port = row["Source Port"]
    destination_port = row["Destination Port"]

    protocol = row["Protocol"]


    # ========================================================
    # SOURCE NODE
    # ========================================================

    source_state[
        "out_flow_count"
    ] += 1

    source_state[
        "unique_destinations"
    ].add(
        destination
    )

    if pd.notna(fwd_bytes):

        source_state[
            "out_bytes"
        ] += float(
            fwd_bytes
        )

    if pd.notna(fwd_packets):

        source_state[
            "out_packets"
        ] += float(
            fwd_packets
        )

    if pd.notna(destination_port):

        source_state[
            "out_unique_ports"
        ].add(
            destination_port
        )

    if pd.notna(protocol):

        source_state[
            "protocols"
        ].add(
            protocol
        )


    # ========================================================
    # DESTINATION NODE
    # ========================================================

    destination_state[
        "in_flow_count"
    ] += 1

    destination_state[
        "unique_sources"
    ].add(
        source
    )

    if pd.notna(bwd_bytes):

        destination_state[
            "in_bytes"
        ] += float(
            bwd_bytes
        )

    if pd.notna(bwd_packets):

        destination_state[
            "in_packets"
        ] += float(
            bwd_packets
        )

    if pd.notna(source_port):

        destination_state[
            "in_unique_ports"
        ].add(
            source_port
        )

    if pd.notna(protocol):

        destination_state[
            "protocols"
        ].add(
            protocol
        )


# ============================================================
# 9. FIT NODE SCALER ON TRAINING PERIOD ONLY
# ============================================================

print("\n" + "-" * 60)
print("PASS 1 - FIT NODE SCALER ON TRAINING PERIOD")
print("-" * 60)


node_state.clear()

training_data = load_temporal_rows(
    TRAIN_DATES,
    "training"
)


training_data["window_start"] = (
    training_data["Timestamp"]
    .dt.floor(
        f"{WINDOW_MINUTES}min"
    )
)


training_windows = (
    training_data["window_start"]
    .nunique()
)


print(
    f"\nTraining windows: "
    f"{training_windows:,}"
)


for window_index, (
    window_start,
    group
) in enumerate(
    training_data.groupby(
        "window_start",
        sort=True
    )
):

    group = group.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )


    # Current-window nodes
    nodes = pd.unique(
        pd.concat(
            [
                group["Source IP"],
                group["Destination IP"],
            ],
            ignore_index=True
        )
    )


    # Historical node features BEFORE
    # current-window updates
    raw_node_features = np.array(
        [
            node_feature_vector(node)
            for node in nodes
        ],
        dtype=np.float64
    )


    # Log transform reduces heavy skew
    transformed = np.log1p(
        np.maximum(
            raw_node_features,
            0
        )
    )


    node_scaler.partial_fit(
        transformed
    )


    # Now update state
    for _, row in group.iterrows():

        update_node_state_from_row(
            row
        )


    if (
        (window_index + 1) % 100 == 0
        or
        window_index + 1 == training_windows
    ):

        print(
            f"Processed "
            f"{window_index + 1:,} / "
            f"{training_windows:,} training windows"
        )


print(
    "\nNode scaler fitting complete."
)


# ============================================================
# 10. SAVE NODE SCALER
# ============================================================

node_scaler_file = (
    "data/processed/"
    "graphshield_node_scaler.joblib"
)


joblib.dump(
    {
        "scaler": node_scaler,

        "feature_names": [
            "out_flow_count",
            "in_flow_count",
            "unique_destinations",
            "unique_sources",
            "out_bytes",
            "in_bytes",
            "out_packets",
            "in_packets",
            "out_unique_ports",
            "in_unique_ports",
            "unique_protocols",
        ],

        "transform": "log1p_then_standardize",

        "fit_dates": sorted(
            TRAIN_DATES
        ),
    },
    node_scaler_file
)


print(
    "\nSaved node scaler:"
)

print(
    node_scaler_file
)


# ============================================================
# 11. NODE TRANSFORMATION
# ============================================================

def transform_node_features(
    raw_node_features
):

    values = np.asarray(
        raw_node_features,
        dtype=np.float64
    )


    values = np.log1p(
        np.maximum(
            values,
            0
        )
    )


    values = node_scaler.transform(
        values
    )


    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return values.astype(
        np.float32
    )


# ============================================================
# 12. RESET NODE STATE BEFORE FINAL BUILD
# ============================================================

node_state.clear()


# ============================================================
# 13. METADATA
# ============================================================

metadata = {

    "window_minutes": WINDOW_MINUTES,

    "node_features": 11,

    "edge_features": 66,

    "num_classes": 15,

    "train_dates": sorted(
        TRAIN_DATES
    ),

    "validation_dates": sorted(
        VALIDATION_DATES
    ),

    "test_dates": sorted(
        TEST_DATES
    ),

    "splits": {},
}


# ============================================================
# 14. BUILD ONE TEMPORAL SPLIT
# ============================================================

def process_split(
    dates,
    split_name
):

    print("\n" + "=" * 60)

    print(
        f"BUILDING {split_name.upper()} GRAPHS"
    )

    print("=" * 60)


    data = load_temporal_rows(
        dates,
        split_name
    )


    data["window_start"] = (
        data["Timestamp"]
        .dt.floor(
            f"{WINDOW_MINUTES}min"
        )
    )


    unique_windows = (
        data["window_start"]
        .nunique()
    )


    print(
        f"\nTemporal windows: "
        f"{unique_windows:,}"
    )


    output_dir = os.path.join(
        OUTPUT_ROOT,
        split_name
    )


    snapshots = []

    batch_index = 0

    total_edges = 0

    total_snapshot_nodes = 0

    label_counter = {}


    # ========================================================
    # PROCESS WINDOWS
    # ========================================================

    for window_index, (
        window_start,
        group
    ) in enumerate(
        data.groupby(
            "window_start",
            sort=True
        )
    ):

        group = group.sort_values(
            "Timestamp"
        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # Current window nodes
        # ----------------------------------------------------

        nodes = pd.unique(
            pd.concat(
                [
                    group["Source IP"],
                    group["Destination IP"],
                ],
                ignore_index=True
            )
        )


        node_to_id = {
            node: index
            for index, node in enumerate(nodes)
        }


        # ----------------------------------------------------
        # Historical node features
        # ----------------------------------------------------

        raw_node_features = np.array(
            [
                node_feature_vector(node)
                for node in nodes
            ],
            dtype=np.float64
        )


        node_features = transform_node_features(
            raw_node_features
        )


        # ----------------------------------------------------
        # Directed edges
        #
        # Source IP -> Destination IP
        # ----------------------------------------------------

        source_ids = [
            node_to_id[source]
            for source in group["Source IP"]
        ]


        destination_ids = [
            node_to_id[destination]
            for destination
            in group["Destination IP"]
        ]


        edge_index = torch.tensor(
            [
                source_ids,
                destination_ids,
            ],
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Edge features
        # ----------------------------------------------------

        raw_edge_features = (
            group[feature_columns]
            .to_numpy(
                dtype=np.float64
            )
        )


        scaled_edge_features = (
            edge_scaler.transform(
                raw_edge_features
            )
        )


        scaled_edge_features = np.nan_to_num(
            scaled_edge_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        edge_attr = torch.tensor(
            scaled_edge_features,
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # Edge labels
        # ----------------------------------------------------

        labels = torch.tensor(
            group["Label"].to_numpy(),
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Create snapshot
        # ----------------------------------------------------

        snapshot = {

            "window_index": window_index,

            "window_start": str(
                window_start
            ),

            "num_nodes": len(nodes),

            "node_ids": list(nodes),

            "x": torch.tensor(
                node_features,
                dtype=torch.float32
            ),

            "edge_index": edge_index,

            "edge_attr": edge_attr,

            "y": labels,

            "timestamps": (
                group["Timestamp"]
                .astype(str)
                .tolist()
            ),
        }


        snapshots.append(
            snapshot
        )


        total_edges += len(group)

        total_snapshot_nodes += len(nodes)


        # ----------------------------------------------------
        # Label counts
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            group["Label"].to_numpy(),
            return_counts=True
        )


        for label, count in zip(
            unique_labels,
            counts
        ):

            label = int(label)

            label_counter[label] = (
                label_counter.get(
                    label,
                    0
                )
                +
                int(count)
            )


        # ----------------------------------------------------
        # UPDATE NODE HISTORY AFTER CURRENT WINDOW
        # ----------------------------------------------------

        for _, row in group.iterrows():

            update_node_state_from_row(
                row
            )


        # ----------------------------------------------------
        # Save full batch
        # ----------------------------------------------------

        if len(snapshots) >= SNAPSHOTS_PER_BATCH:

            batch_file = os.path.join(
                output_dir,
                f"batch_{batch_index:04d}.pt"
            )


            torch.save(
                snapshots,
                batch_file
            )


            print(
                f"Saved: {batch_file} "
                f"({len(snapshots)} snapshots)"
            )


            snapshots = []

            batch_index += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (window_index + 1) % 100 == 0
            or
            window_index + 1 == unique_windows
        ):

            print(
                f"Processed "
                f"{window_index + 1:,} / "
                f"{unique_windows:,} windows | "
                f"Edges: {total_edges:,}"
            )


    # ========================================================
    # SAVE FINAL PARTIAL BATCH
    # ========================================================

    final_partial_batch = False


    if snapshots:

        batch_file = os.path.join(
            output_dir,
            f"batch_{batch_index:04d}.pt"
        )


        torch.save(
            snapshots,
            batch_file
        )


        print(
            f"Saved: {batch_file} "
            f"({len(snapshots)} snapshots)"
        )


        final_partial_batch = True

        batch_index += 1

        snapshots = []


    # ========================================================
    # LABEL COUNTS
    # ========================================================

    split_metadata = {

        "windows": int(
            unique_windows
        ),

        "edges": int(
            total_edges
        ),

        "snapshot_node_occurrences": int(
            total_snapshot_nodes
        ),

        "batches": int(
            batch_index
        ),

        "label_counts": {
            str(label): int(count)
            for label, count
            in sorted(
                label_counter.items()
            )
        },
    }


    metadata["splits"][
        split_name
    ] = split_metadata


    print(
        f"\n{split_name.upper()} COMPLETE"
    )

    print(
        f"Windows: {unique_windows:,}"
    )

    print(
        f"Edges: {total_edges:,}"
    )

    print(
        f"Snapshot-node occurrences: "
        f"{total_snapshot_nodes:,}"
    )

    print(
        f"Batches: {batch_index}"
    )


# ============================================================
# 15. BUILD TRAINING GRAPHS
# ============================================================

process_split(
    TRAIN_DATES,
    "train"
)


# ============================================================
# 16. BUILD VALIDATION GRAPHS
# ============================================================

process_split(
    VALIDATION_DATES,
    "validation"
)


# ============================================================
# 17. BUILD TEST GRAPHS
# ============================================================

process_split(
    TEST_DATES,
    "test"
)


# ============================================================
# 18. SAVE METADATA
# ============================================================

metadata_file = os.path.join(
    OUTPUT_ROOT,
    "metadata.json"
)


with open(
    metadata_file,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "\nSaved metadata:"
)

print(
    metadata_file
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FULL TEMPORAL GRAPH DATASET COMPLETE")
print("=" * 60)


print("\nGraph representation:")

print(
    "  Nodes = network entities / IP addresses"
)

print(
    "  Edges = network flows"
)

print(
    "  Node features = 11 historical features"
)

print(
    "  Edge features = 66 scaled flow features"
)

print(
    "  Target = edge-level attack class"
)

print(
    f"  Temporal window = {WINDOW_MINUTES} minutes"
)


print("\nTemporal protocol:")

print(
    "  TRAIN      = July 3-5"
)

print(
    "  VALIDATION = July 6"
)

print(
    "  TEST       = July 7"
)


print("\nOutput directory:")

print(
    OUTPUT_ROOT
)


print("\n" + "=" * 60)