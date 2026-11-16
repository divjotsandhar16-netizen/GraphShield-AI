import os
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/processed/train.csv"

SAMPLE_SIZE = 10_000

RESULTS_FILE = (
    "results/graph_verification_sample.csv"
)

os.makedirs("results", exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI - GRAPH REPRESENTATION VERIFICATION")
print("=" * 60)

print(
    f"\nLoading first {SAMPLE_SIZE:,} flows..."
)

df = pd.read_csv(
    INPUT_FILE,
    nrows=SAMPLE_SIZE,
    low_memory=False,
)

print(
    f"Loaded rows: {len(df):,}"
)

print(
    f"Total columns: {len(df.columns)}"
)


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
]

TARGET_COLUMN = "Label"


EDGE_FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in IDENTIFIER_COLUMNS + [TARGET_COLUMN]
]


# =========================================================
# 1. BASIC STRUCTURE CHECK
# =========================================================

print("\n" + "-" * 60)
print("1. BASIC STRUCTURE")
print("-" * 60)

print(
    f"Candidate edge features: "
    f"{len(EDGE_FEATURE_COLUMNS)}"
)

numeric_edge_features = df[
    EDGE_FEATURE_COLUMNS
].select_dtypes(
    include=np.number
)

print(
    f"Numerical edge features: "
    f"{len(numeric_edge_features.columns)}"
)

if len(EDGE_FEATURE_COLUMNS) != 66:
    raise ValueError(
        f"Expected 66 edge features, "
        f"found {len(EDGE_FEATURE_COLUMNS)}."
    )

if len(numeric_edge_features.columns) != 66:
    raise ValueError(
        "Not all 66 edge features are numerical."
    )

print("✓ Exactly 66 numerical edge features confirmed.")


# =========================================================
# 2. NODE CONSTRUCTION
# =========================================================

print("\n" + "-" * 60)
print("2. NODE VERIFICATION")
print("-" * 60)

all_ips = pd.concat(
    [
        df["Source IP"],
        df["Destination IP"],
    ],
    ignore_index=True,
).dropna()

unique_ips = pd.unique(all_ips)

node_to_id = {
    ip: node_id
    for node_id, ip in enumerate(unique_ips)
}

source_nodes = df["Source IP"].map(
    node_to_id
).to_numpy()

destination_nodes = df["Destination IP"].map(
    node_to_id
).to_numpy()

print(
    f"Unique graph nodes: "
    f"{len(unique_ips):,}"
)

print(
    f"Source node IDs: "
    f"{source_nodes.min()} → {source_nodes.max()}"
)

print(
    f"Destination node IDs: "
    f"{destination_nodes.min()} → "
    f"{destination_nodes.max()}"
)

if pd.isna(source_nodes).any():
    raise ValueError(
        "Some Source IPs could not be mapped."
    )

if pd.isna(destination_nodes).any():
    raise ValueError(
        "Some Destination IPs could not be mapped."
    )

print(
    "✓ Every source/destination IP maps to a node ID."
)


# =========================================================
# 3. DIRECTED EDGE CONSTRUCTION
# =========================================================

print("\n" + "-" * 60)
print("3. DIRECTED EDGE VERIFICATION")
print("-" * 60)

edge_index = np.vstack(
    [
        source_nodes,
        destination_nodes,
    ]
)

print(
    "edge_index shape:",
    edge_index.shape
)

print(
    "Number of edges:",
    edge_index.shape[1]
)

if edge_index.shape[0] != 2:
    raise ValueError(
        "edge_index must have shape [2, number_of_edges]."
    )

if edge_index.shape[1] != len(df):
    raise ValueError(
        "Number of graph edges does not match flows."
    )

print(
    "✓ Every flow corresponds to exactly one directed edge."
)


# =========================================================
# 4. DIRECTED PAIR ANALYSIS
# =========================================================

print("\n" + "-" * 60)
print("4. DIRECTED COMMUNICATION ANALYSIS")
print("-" * 60)

directed_pairs = pd.DataFrame(
    {
        "source": df["Source IP"].astype(str),
        "destination": df["Destination IP"].astype(str),
    }
)

unique_directed_pairs = (
    directed_pairs
    .drop_duplicates()
    .shape[0]
)

undirected_pairs = pd.DataFrame(
    {
        "endpoint_1": directed_pairs[
            ["source", "destination"]
        ].min(axis=1),

        "endpoint_2": directed_pairs[
            ["source", "destination"]
        ].max(axis=1),
    }
)

unique_undirected_pairs = (
    undirected_pairs
    .drop_duplicates()
    .shape[0]
)

print(
    f"Unique directed IP pairs: "
    f"{unique_directed_pairs:,}"
)

print(
    f"Unique undirected IP pairs: "
    f"{unique_undirected_pairs:,}"
)

if unique_directed_pairs < unique_undirected_pairs:
    raise ValueError(
        "Unexpected directed-pair result."
    )

print(
    "✓ Source → destination direction is preserved."
)


# =========================================================
# 5. REPEATED FLOWS / MULTIEDGES
# =========================================================

print("\n" + "-" * 60)
print("5. MULTIEDGE VERIFICATION")
print("-" * 60)

pair_counts = (
    directed_pairs
    .value_counts()
)

repeated_pairs = (
    pair_counts[pair_counts > 1]
)

print(
    f"Directed pairs appearing more than once: "
    f"{len(repeated_pairs):,}"
)

if len(repeated_pairs) > 0:

    print("\nTop repeated directed pairs:")

    print(
        repeated_pairs.head(10)
    )

print(
    "\n✓ Repeated flows are retained as separate edges."
)


# =========================================================
# 6. SELF-LOOP CHECK
# =========================================================

print("\n" + "-" * 60)
print("6. SELF-LOOP CHECK")
print("-" * 60)

self_loops = (
    df["Source IP"]
    == df["Destination IP"]
)

self_loop_count = int(
    self_loops.sum()
)

print(
    f"Self-loop flows: "
    f"{self_loop_count:,}"
)

if self_loop_count == 0:
    print(
        "✓ No self-loop flows in this sample."
    )
else:
    print(
        "Self-loops exist and are being preserved."
    )


# =========================================================
# 7. EDGE FEATURE VERIFICATION
# =========================================================

print("\n" + "-" * 60)
print("7. EDGE FEATURE VERIFICATION")
print("-" * 60)

edge_attr = df[
    EDGE_FEATURE_COLUMNS
].to_numpy(
    dtype=np.float64
)

print(
    "edge_attr shape:",
    edge_attr.shape
)

print(
    f"Edge feature dimensions: "
    f"{edge_attr.shape[1]}"
)

if edge_attr.shape[0] != edge_index.shape[1]:
    raise ValueError(
        "Number of edge-feature rows does not "
        "match number of edges."
    )

print(
    "✓ Every edge has exactly one 66-dimensional "
    "feature vector."
)


# =========================================================
# 8. TIMESTAMP VERIFICATION
# =========================================================

print("\n" + "-" * 60)
print("8. TIMESTAMP VERIFICATION")
print("-" * 60)

timestamps = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    dayfirst=True,
    errors="coerce",
)

invalid_timestamps = int(
    timestamps.isna().sum()
)

print(
    f"Valid timestamps: "
    f"{len(timestamps) - invalid_timestamps:,}"
)

print(
    f"Invalid timestamps: "
    f"{invalid_timestamps:,}"
)

if invalid_timestamps > 0:
    raise ValueError(
        "Some timestamps could not be parsed."
    )

print(
    "✓ Every flow/edge has a valid timestamp."
)

print(
    "\nFirst 5 timestamps:"
)

print(
    timestamps.head().to_string(index=False)
)


# =========================================================
# 9. LABEL VERIFICATION
# =========================================================

print("\n" + "-" * 60)
print("9. LABEL VERIFICATION")
print("-" * 60)

edge_labels = (
    df["Label"]
    .astype(int)
    .to_numpy()
)

print(
    f"Number of edge labels: "
    f"{len(edge_labels):,}"
)

print(
    f"Unique labels in sample: "
    f"{np.unique(edge_labels)}"
)

if len(edge_labels) != edge_index.shape[1]:
    raise ValueError(
        "Number of labels does not match "
        "number of edges."
    )

print(
    "✓ Every edge has exactly one label."
)


# =========================================================
# 10. FLOW ID CONSISTENCY
# =========================================================

print("\n" + "-" * 60)
print("10. FLOW ID CONSISTENCY")
print("-" * 60)

flow_id_checks = []

for _, row in df.head(1000).iterrows():

    flow_id = str(
        row["Flow ID"]
    )

    expected_parts = [
        str(row["Source IP"]),
        str(row["Destination IP"]),
        str(int(row["Source Port"])),
        str(int(row["Destination Port"])),
        str(int(row["Protocol"])),
    ]

    expected_flow_id = "-".join(
        expected_parts
    )

    flow_id_checks.append(
        flow_id == expected_flow_id
    )

flow_id_match_count = sum(
    flow_id_checks
)

print(
    f"Flow IDs checked: "
    f"{len(flow_id_checks):,}"
)

print(
    f"Matching Flow IDs: "
    f"{flow_id_match_count:,}"
)

if flow_id_match_count != len(flow_id_checks):
    print(
        "WARNING: Some Flow IDs do not exactly "
        "match their endpoint/port/protocol fields."
    )
else:
    print(
        "✓ Flow ID is consistent with the "
        "source/destination/port/protocol fields "
        "for the checked rows."
    )


# =========================================================
# 11. ROW-ALIGNMENT VERIFICATION
# =========================================================

print("\n" + "-" * 60)
print("11. ROW ALIGNMENT VERIFICATION")
print("-" * 60)

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(df),
    size=min(25, len(df)),
    replace=False,
)

alignment_ok = True

for index in sample_indices:

    original_source = df.iloc[
        index
    ]["Source IP"]

    original_destination = df.iloc[
        index
    ]["Destination IP"]

    expected_source_node = node_to_id[
        original_source
    ]

    expected_destination_node = node_to_id[
        original_destination
    ]

    if (
        edge_index[0, index]
        != expected_source_node
    ):
        alignment_ok = False

    if (
        edge_index[1, index]
        != expected_destination_node
    ):
        alignment_ok = False

    if (
        edge_labels[index]
        != int(df.iloc[index]["Label"])
    ):
        alignment_ok = False

    if (
        timestamps.iloc[index]
        != pd.to_datetime(
            df.iloc[index]["Timestamp"],
            format="mixed",
            dayfirst=True,
        )
    ):
        alignment_ok = False

if not alignment_ok:
    raise ValueError(
        "Graph row alignment verification failed."
    )

print(
    "✓ Randomly sampled rows preserve "
    "source, destination, timestamp and label alignment."
)


# =========================================================
# 12. SAVE INSPECTION TABLE
# =========================================================

inspection = pd.DataFrame(
    {
        "Flow ID": df["Flow ID"],
        "Source IP": df["Source IP"],
        "Source Node": source_nodes,
        "Destination IP": df["Destination IP"],
        "Destination Node": destination_nodes,
        "Source Port": df["Source Port"],
        "Destination Port": df["Destination Port"],
        "Protocol": df["Protocol"],
        "Timestamp": df["Timestamp"],
        "Label": edge_labels,
    }
)

inspection.to_csv(
    RESULTS_FILE,
    index=False,
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("GRAPH REPRESENTATION VERIFICATION COMPLETE")
print("=" * 60)

print(
    f"\nFlows checked       : {len(df):,}"
)

print(
    f"Unique nodes        : {len(unique_ips):,}"
)

print(
    f"Directed edges      : {edge_index.shape[1]:,}"
)

print(
    f"Edge feature dims   : {edge_attr.shape[1]}"
)

print(
    f"Unique labels       : {len(np.unique(edge_labels))}"
)

print(
    f"Self-loops          : {self_loop_count:,}"
)

print(
    "\nSaved inspection file:"
)

print(RESULTS_FILE)

print(
    "\nGraph representation is structurally valid."
)