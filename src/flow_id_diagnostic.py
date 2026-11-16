import pandas as pd


FILE = "data/processed/train.csv"

CHECK_ROWS = 10000


print("=" * 60)
print("GRAPHSHIELD AI - FLOW ID DIAGNOSTIC")
print("=" * 60)


df = pd.read_csv(
    FILE,
    nrows=CHECK_ROWS,
    usecols=[
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
    ],
    low_memory=False,
)


forward_match = 0
reverse_match = 0
invalid_match = 0


for _, row in df.iterrows():

    flow_id = str(row["Flow ID"])

    parts = flow_id.split("-")

    if len(parts) != 5:
        invalid_match += 1
        continue

    flow_ip_1 = parts[0]
    flow_ip_2 = parts[1]
    flow_port_1 = parts[2]
    flow_port_2 = parts[3]
    flow_protocol = parts[4]

    source_ip = str(row["Source IP"])
    destination_ip = str(row["Destination IP"])

    source_port = str(int(row["Source Port"]))
    destination_port = str(int(row["Destination Port"]))

    protocol = str(int(row["Protocol"]))


    # -----------------------------------------------------
    # Forward orientation
    # -----------------------------------------------------

    if (
        flow_ip_1 == source_ip
        and flow_ip_2 == destination_ip
        and flow_port_1 == source_port
        and flow_port_2 == destination_port
        and flow_protocol == protocol
    ):

        forward_match += 1
        continue


    # -----------------------------------------------------
    # Reverse orientation
    # -----------------------------------------------------

    if (
        flow_ip_1 == destination_ip
        and flow_ip_2 == source_ip
        and flow_port_1 == destination_port
        and flow_port_2 == source_port
        and flow_protocol == protocol
    ):

        reverse_match += 1
        continue


    invalid_match += 1


# =========================================================
# RESULTS
# =========================================================

print(
    f"\nRows checked: {len(df):,}"
)

print(
    f"Forward Flow ID matches : {forward_match:,}"
)

print(
    f"Reverse Flow ID matches : {reverse_match:,}"
)

print(
    f"No valid orientation    : {invalid_match:,}"
)


print("\nPercentages:")

total = len(df)

print(
    f"Forward : {forward_match / total * 100:.2f}%"
)

print(
    f"Reverse : {reverse_match / total * 100:.2f}%"
)

print(
    f"Invalid : {invalid_match / total * 100:.2f}%"
)


print("\n" + "=" * 60)
print("FLOW ID DIAGNOSTIC COMPLETE")
print("=" * 60)