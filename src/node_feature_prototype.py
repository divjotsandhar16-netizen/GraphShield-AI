import os
from collections import defaultdict

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/processed/train.csv"

SAMPLE_SIZE = 10_000

OUTPUT_FILE = (
    "results/node_feature_prototype.csv"
)

os.makedirs(
    "results",
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("GRAPHSHIELD AI - NODE FEATURE PROTOTYPE")
print("=" * 60)

print(
    f"\nLoading {SAMPLE_SIZE:,} flows..."
)

df = pd.read_csv(
    INPUT_FILE,
    nrows=SAMPLE_SIZE,
    low_memory=False,
)


# =========================================================
# PARSE TIMESTAMPS
# =========================================================

df["Parsed Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    dayfirst=True,
    errors="coerce",
)

invalid_timestamps = int(
    df["Parsed Timestamp"].isna().sum()
)

if invalid_timestamps > 0:

    raise ValueError(
        f"Found {invalid_timestamps} invalid timestamps."
    )


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================

df = df.sort_values(
    "Parsed Timestamp"
).reset_index(
    drop=True
)

print(
    "\nChronological ordering applied."
)

print(
    "Earliest flow:",
    df["Parsed Timestamp"].iloc[0]
)

print(
    "Latest flow:",
    df["Parsed Timestamp"].iloc[-1]
)


# =========================================================
# NODE STATE
# =========================================================

# Each IP gets a state describing its historical
# communication behavior.

node_state = defaultdict(
    lambda: {
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
)


# =========================================================
# STORE RESULTS
# =========================================================

records = []


# =========================================================
# PROCESS FLOWS CHRONOLOGICALLY
# =========================================================

print(
    "\nConstructing history-based node features..."
)

for index, row in df.iterrows():

    source_ip = str(
        row["Source IP"]
    )

    destination_ip = str(
        row["Destination IP"]
    )

    source_port = int(
        row["Source Port"]
    )

    destination_port = int(
        row["Destination Port"]
    )

    protocol = int(
        row["Protocol"]
    )

    fwd_bytes = float(
        row["Total Length of Fwd Packets"]
    )

    bwd_bytes = float(
        row["Total Length of Bwd Packets"]
    )

    fwd_packets = float(
        row["Total Fwd Packets"]
    )

    bwd_packets = float(
        row["Total Backward Packets"]
    )


    # -----------------------------------------------------
    # READ NODE HISTORY BEFORE CURRENT FLOW
    # -----------------------------------------------------

    source_state = node_state[
        source_ip
    ]

    destination_state = node_state[
        destination_ip
    ]


    # -----------------------------------------------------
    # SOURCE NODE FEATURES
    # -----------------------------------------------------

    source_features = {
        "source_out_flow_count_before":
            source_state["out_flow_count"],

        "source_in_flow_count_before":
            source_state["in_flow_count"],

        "source_unique_destinations_before":
            len(
                source_state[
                    "unique_destinations"
                ]
            ),

        "source_unique_sources_before":
            len(
                source_state[
                    "unique_sources"
                ]
            ),

        "source_out_bytes_before":
            source_state["out_bytes"],

        "source_in_bytes_before":
            source_state["in_bytes"],

        "source_out_packets_before":
            source_state["out_packets"],

        "source_in_packets_before":
            source_state["in_packets"],

        "source_out_unique_ports_before":
            len(
                source_state[
                    "out_unique_ports"
                ]
            ),

        "source_in_unique_ports_before":
            len(
                source_state[
                    "in_unique_ports"
                ]
            ),

        "source_unique_protocols_before":
            len(
                source_state[
                    "protocols"
                ]
            ),
    }


    # -----------------------------------------------------
    # DESTINATION NODE FEATURES
    # -----------------------------------------------------

    destination_features = {
        "destination_out_flow_count_before":
            destination_state["out_flow_count"],

        "destination_in_flow_count_before":
            destination_state["in_flow_count"],

        "destination_unique_destinations_before":
            len(
                destination_state[
                    "unique_destinations"
                ]
            ),

        "destination_unique_sources_before":
            len(
                destination_state[
                    "unique_sources"
                ]
            ),

        "destination_out_bytes_before":
            destination_state["out_bytes"],

        "destination_in_bytes_before":
            destination_state["in_bytes"],

        "destination_out_packets_before":
            destination_state["out_packets"],

        "destination_in_packets_before":
            destination_state["in_packets"],

        "destination_out_unique_ports_before":
            len(
                destination_state[
                    "out_unique_ports"
                ]
            ),

        "destination_in_unique_ports_before":
            len(
                destination_state[
                    "in_unique_ports"
                ]
            ),

        "destination_unique_protocols_before":
            len(
                destination_state[
                    "protocols"
                ]
            ),
    }


    # -----------------------------------------------------
    # SAVE CURRENT EDGE + HISTORICAL NODE STATE
    # -----------------------------------------------------

    record = {
        "Flow ID": row["Flow ID"],
        "Timestamp": row["Timestamp"],

        "Source IP": source_ip,
        "Destination IP": destination_ip,

        "Source Port": source_port,
        "Destination Port": destination_port,

        "Protocol": protocol,

        "Label": int(row["Label"]),
    }

    record.update(
        source_features
    )

    record.update(
        destination_features
    )

    records.append(
        record
    )


    # =====================================================
    # UPDATE NODE STATE AFTER CURRENT FLOW
    # =====================================================

    source_state["out_flow_count"] += 1

    source_state["out_bytes"] += fwd_bytes

    source_state["out_packets"] += fwd_packets

    source_state[
        "unique_destinations"
    ].add(
        destination_ip
    )

    source_state[
        "out_unique_ports"
    ].add(
        destination_port
    )

    source_state[
        "protocols"
    ].add(
        protocol
    )


    # -----------------------------------------------------
    # Destination receives backward traffic
    # -----------------------------------------------------

    destination_state[
        "in_flow_count"
    ] += 1

    destination_state[
        "in_bytes"
    ] += bwd_bytes

    destination_state[
        "in_packets"
    ] += bwd_packets

    destination_state[
        "unique_sources"
    ].add(
        source_ip
    )

    destination_state[
        "in_unique_ports"
    ].add(
        source_port
    )

    destination_state[
        "protocols"
    ].add(
        protocol
    )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (
        (index + 1) % 2000 == 0
    ):

        print(
            f"Processed "
            f"{index + 1:,} / "
            f"{len(df):,}"
        )


# =========================================================
# CREATE RESULT DATAFRAME
# =========================================================

result = pd.DataFrame(
    records
)


# =========================================================
# NODE FEATURE COLUMNS
# =========================================================

node_feature_columns = [

    "source_out_flow_count_before",
    "source_in_flow_count_before",
    "source_unique_destinations_before",
    "source_unique_sources_before",
    "source_out_bytes_before",
    "source_in_bytes_before",
    "source_out_packets_before",
    "source_in_packets_before",
    "source_out_unique_ports_before",
    "source_in_unique_ports_before",
    "source_unique_protocols_before",

    "destination_out_flow_count_before",
    "destination_in_flow_count_before",
    "destination_unique_destinations_before",
    "destination_unique_sources_before",
    "destination_out_bytes_before",
    "destination_in_bytes_before",
    "destination_out_packets_before",
    "destination_in_packets_before",
    "destination_out_unique_ports_before",
    "destination_in_unique_ports_before",
    "destination_unique_protocols_before",
]


# =========================================================
# VERIFICATION
# =========================================================

print(
    "\n" + "-" * 60
)

print(
    "NODE FEATURE VERIFICATION"
)

print(
    "-" * 60
)

print(
    "\nNumber of edge records:",
    len(result)
)

print(
    "Node-history features per endpoint:",
    11
)

print(
    "Total source + destination "
    "history features per flow:",
    len(node_feature_columns)
)


# ---------------------------------------------------------
# First edge should have zero history
# ---------------------------------------------------------

first_row = result.iloc[0]

first_features = first_row[
    node_feature_columns
].to_numpy(
    dtype=float
)

print(
    "\nFirst chronological flow:"
)

print(
    "Timestamp:",
    first_row["Timestamp"]
)

print(
    "All historical node features zero:",
    bool(
        np.all(
            first_features == 0
        )
    )
)


# ---------------------------------------------------------
# Node feature statistics
# ---------------------------------------------------------

print(
    "\nHistorical node feature ranges:"
)

stats = result[
    node_feature_columns
].describe().T[
    ["min", "max", "mean"]
]

print(
    stats.to_string()
)


# =========================================================
# SAVE
# =========================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    "\nSaved prototype to:"
)

print(OUTPUT_FILE)


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "NODE FEATURE PROTOTYPE COMPLETE"
)

print(
    "=" * 60
)