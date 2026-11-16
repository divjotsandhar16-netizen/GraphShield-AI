import os
import numpy as np
import pandas as pd
import torch


# ============================================================
# GRAPHSHIELD AI - TEMPORAL GRAPH PROTOTYPE
# ============================================================

INPUT_FILE = "data/processed/train.csv"
OUTPUT_FILE = "results/temporal_graph_prototype.pt"

# Prototype size
N_ROWS = 50_000

# Temporal window size
WINDOW_MINUTES = 5

# Intended temporal training period
TRAIN_DATES = {
    "2017-07-03",
    "2017-07-04",
    "2017-07-05"
}


print("=" * 60)
print("GRAPHSHIELD AI - TEMPORAL GRAPH PROTOTYPE")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print(f"\nLoading {N_ROWS:,} rows from training data...")

df = pd.read_csv(
    INPUT_FILE,
    nrows=N_ROWS,
    low_memory=False
)

print(f"Loaded shape: {df.shape}")


# ============================================================
# 2. PARSE TIMESTAMP
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)


# Remove rows where essential graph information is missing
df = df.dropna(
    subset=[
        "Timestamp",
        "Source IP",
        "Destination IP",
        "Label"
    ]
).copy()


# ============================================================
# 3. CREATE TEMPORAL DATE HELPER
# ============================================================

df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")


# ============================================================
# 4. APPLY TEMPORAL TRAINING PERIOD
# ============================================================

df = df[
    df["Date"].isin(TRAIN_DATES)
].copy()

print(
    f"Rows after temporal training-date filter: {len(df):,}"
)


# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

print("\nChronological ordering applied.")

if len(df) > 0:
    print(
        "Earliest flow:",
        df["Timestamp"].iloc[0]
    )

    print(
        "Latest flow:",
        df["Timestamp"].iloc[-1]
    )


# ============================================================
# 6. CREATE TEMPORAL WINDOWS
# ============================================================

df["window_start"] = df[
    "Timestamp"
].dt.floor(
    f"{WINDOW_MINUTES}min"
)

unique_windows = df[
    "window_start"
].nunique()

print(
    f"\nWindow size: {WINDOW_MINUTES} minutes"
)

print(
    f"Number of temporal windows: {unique_windows}"
)


# ============================================================
# 7. IDENTIFY THE 66 EDGE FEATURES
# ============================================================

# These are identifiers / targets / temporal helper columns.
# They must NOT become predictive edge features.

excluded_columns = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "Date",
    "window_start"
}


feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


print(
    f"Edge feature count: {len(feature_columns)}"
)


# ============================================================
# 8. VERIFY FEATURE COUNT
# ============================================================

if len(feature_columns) != 66:

    print("\nERROR: Expected exactly 66 model features.")

    print("\nDetected feature columns:")

    for i, column in enumerate(feature_columns, start=1):
        print(f"{i:02d}. {column}")

    raise ValueError(
        f"Expected 66 edge features, "
        f"found {len(feature_columns)}"
    )


print("\n66 edge features confirmed.")


# ============================================================
# 9. NODE HISTORY STATE
# ============================================================

node_state = {}


def empty_state():
    """
    Creates an empty historical state for one network node.
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

        "protocols": set()
    }


def get_state(node):
    """
    Return the historical state of a node.
    Create it if the node has never appeared before.
    """

    if node not in node_state:
        node_state[node] = empty_state()

    return node_state[node]


def node_feature_vector(node):
    """
    Returns historical information available BEFORE
    the current temporal window.

    11 historical node features:

    1. outgoing flow count
    2. incoming flow count
    3. unique destinations
    4. unique sources
    5. outgoing bytes
    6. incoming bytes
    7. outgoing packets
    8. incoming packets
    9. unique outgoing ports
    10. unique incoming ports
    11. unique protocols
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
        )
    ]


# ============================================================
# 10. TEMPORAL GRAPH SNAPSHOTS
# ============================================================

snapshots = []

total_edges = 0
total_nodes = 0


print(
    "\nConstructing temporal graph snapshots..."
)


# ============================================================
# 11. PROCESS EACH TEMPORAL WINDOW
# ============================================================

for window_index, (
    window_start,
    group
) in enumerate(
    df.groupby(
        "window_start",
        sort=True
    )
):

    # --------------------------------------------------------
    # Sort flows within the window
    # --------------------------------------------------------

    group = group.sort_values(
        "Timestamp"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Identify nodes in current window
    # --------------------------------------------------------

    nodes = pd.unique(
        pd.concat(
            [
                group["Source IP"],
                group["Destination IP"]
            ],
            ignore_index=True
        )
    )


    # Map IP address → local integer node ID
    node_to_id = {
        node: idx
        for idx, node in enumerate(nodes)
    }


    # --------------------------------------------------------
    # IMPORTANT:
    # Node features are created BEFORE processing
    # the current window.
    # --------------------------------------------------------

    node_features = np.array(
        [
            node_feature_vector(node)
            for node in nodes
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Build directed edges
    #
    # Source IP → Destination IP
    # --------------------------------------------------------

    source_ids = [
        node_to_id[source]
        for source in group["Source IP"]
    ]

    destination_ids = [
        node_to_id[destination]
        for destination in group["Destination IP"]
    ]


    edge_index = torch.tensor(
        [
            source_ids,
            destination_ids
        ],
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Build edge features
    # --------------------------------------------------------

    edge_values = (
        group[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )


    # Prototype-only numerical safety.
    # Final GNN preprocessing will use the
    # established training-only preprocessing.

    edge_values = np.nan_to_num(
        edge_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    edge_attr = torch.tensor(
        edge_values,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Edge labels
    # --------------------------------------------------------

    labels = torch.tensor(
        group["Label"].to_numpy(),
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Store timestamps
    # --------------------------------------------------------

    timestamps = (
        group["Timestamp"]
        .astype(str)
        .tolist()
    )


    # --------------------------------------------------------
    # Construct snapshot
    # --------------------------------------------------------

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

        "timestamps": timestamps
    }


    snapshots.append(snapshot)


    total_edges += len(group)
    total_nodes += len(nodes)


    # --------------------------------------------------------
    # UPDATE NODE HISTORY
    #
    # This happens AFTER the snapshot has been created.
    #
    # Therefore current-window information cannot contaminate
    # its node-history features.
    # --------------------------------------------------------

    for _, row in group.iterrows():

        source = row["Source IP"]
        destination = row["Destination IP"]


        source_state = get_state(source)
        destination_state = get_state(destination)


        # ----------------------------------------------------
        # Safely read flow values
        # ----------------------------------------------------

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


        # ====================================================
        # SOURCE NODE UPDATE
        # ====================================================

        source_state[
            "out_flow_count"
        ] += 1


        source_state[
            "unique_destinations"
        ].add(destination)


        if pd.notna(fwd_bytes):

            source_state[
                "out_bytes"
            ] += float(fwd_bytes)


        if pd.notna(fwd_packets):

            source_state[
                "out_packets"
            ] += float(fwd_packets)


        if pd.notna(destination_port):

            source_state[
                "out_unique_ports"
            ].add(destination_port)


        if pd.notna(protocol):

            source_state[
                "protocols"
            ].add(protocol)


        # ====================================================
        # DESTINATION NODE UPDATE
        # ====================================================

        destination_state[
            "in_flow_count"
        ] += 1


        destination_state[
            "unique_sources"
        ].add(source)


        if pd.notna(bwd_bytes):

            destination_state[
                "in_bytes"
            ] += float(bwd_bytes)


        if pd.notna(bwd_packets):

            destination_state[
                "in_packets"
            ] += float(bwd_packets)


        if pd.notna(source_port):

            destination_state[
                "in_unique_ports"
            ].add(source_port)


        if pd.notna(protocol):

            destination_state[
                "protocols"
            ].add(protocol)


    # --------------------------------------------------------
    # Progress display
    # --------------------------------------------------------

    if (
        (window_index + 1) % 50 == 0
        or
        window_index == unique_windows - 1
    ):

        print(
            f"Processed "
            f"{window_index + 1} / "
            f"{unique_windows} windows"
        )


# ============================================================
# 12. VERIFICATION
# ============================================================

print("\n" + "-" * 60)
print("TEMPORAL GRAPH VERIFICATION")
print("-" * 60)


print(
    f"Total snapshots: {len(snapshots)}"
)

print(
    f"Total edges: {total_edges:,}"
)

print(
    f"Total snapshot-node occurrences: "
    f"{total_nodes:,}"
)


if len(snapshots) == 0:

    raise RuntimeError(
        "No graph snapshots were created."
    )


# ============================================================
# 13. FIRST SNAPSHOT CHECK
# ============================================================

first_snapshot = snapshots[0]


print("\nFirst snapshot:")

print(
    "Window:",
    first_snapshot["window_start"]
)

print(
    "Nodes:",
    first_snapshot["num_nodes"]
)

print(
    "Edges:",
    first_snapshot["edge_index"].shape[1]
)

print(
    "Node feature shape:",
    tuple(
        first_snapshot["x"].shape
    )
)

print(
    "Edge feature shape:",
    tuple(
        first_snapshot["edge_attr"].shape
    )
)

print(
    "Labels shape:",
    tuple(
        first_snapshot["y"].shape
    )
)


# ============================================================
# 14. NODE HISTORY LEAKAGE CHECK
# ============================================================

first_history_zero = torch.all(
    first_snapshot["x"] == 0
).item()


print(
    "First snapshot node history all zero:",
    first_history_zero
)


# ============================================================
# 15. LATER HISTORY CHECK
# ============================================================

later_history_nonzero = False

for snapshot in snapshots[1:]:

    if torch.any(
        snapshot["x"] != 0
    ):

        later_history_nonzero = True
        break


print(
    "Later snapshot contains historical information:",
    later_history_nonzero
)


# ============================================================
# 16. SELF-LOOP CHECK
# ============================================================

total_self_loops = 0


for snapshot in snapshots:

    edge_index = snapshot["edge_index"]

    total_self_loops += int(
        (
            edge_index[0]
            ==
            edge_index[1]
        ).sum().item()
    )


print(
    "Total self-loops:",
    total_self_loops
)


# ============================================================
# 17. EDGE-LABEL ALIGNMENT CHECK
# ============================================================

edge_label_alignment = True


for snapshot in snapshots:

    number_of_edges = (
        snapshot[
            "edge_index"
        ].shape[1]
    )

    number_of_labels = (
        snapshot[
            "y"
        ].shape[0]
    )


    if number_of_edges != number_of_labels:

        edge_label_alignment = False
        break


print(
    "Edge-label alignment:",
    edge_label_alignment
)


# ============================================================
# 18. EDGE FEATURE ALIGNMENT CHECK
# ============================================================

edge_feature_alignment = True


for snapshot in snapshots:

    number_of_edges = (
        snapshot[
            "edge_index"
        ].shape[1]
    )

    number_of_edge_features = (
        snapshot[
            "edge_attr"
        ].shape[0]
    )


    if (
        number_of_edges
        !=
        number_of_edge_features
    ):

        edge_feature_alignment = False
        break


print(
    "Edge-feature alignment:",
    edge_feature_alignment
)


# ============================================================
# 19. GRAPH STRUCTURE CHECK
# ============================================================

graph_structure_valid = True


for snapshot in snapshots:

    edge_index = snapshot[
        "edge_index"
    ]


    if edge_index.shape[0] != 2:

        graph_structure_valid = False
        break


print(
    "Directed edge_index shape valid:",
    graph_structure_valid
)


# ============================================================
# 20. LABEL CHECK
# ============================================================

all_labels = torch.cat(
    [
        snapshot["y"]
        for snapshot in snapshots
    ]
)


unique_labels = torch.unique(
    all_labels
).tolist()


print(
    "Unique labels observed:",
    unique_labels
)


# ============================================================
# 21. SAVE TEMPORAL GRAPH
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


torch.save(
    snapshots,
    OUTPUT_FILE
)


print(
    "\nSaved temporal graph prototype to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 22. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL GRAPH PROTOTYPE COMPLETE")
print("=" * 60)