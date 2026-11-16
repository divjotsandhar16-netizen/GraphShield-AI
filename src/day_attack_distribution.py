import pandas as pd


FILES = {
    "TRAIN": "data/processed/train.csv",
    "VALIDATION": "data/processed/validation.csv",
    "TEST": "data/processed/test.csv",
}

CHUNK_SIZE = 200_000


print("=" * 60)
print("GRAPHSHIELD AI - DAY / ATTACK DISTRIBUTION")
print("=" * 60)


# =========================================================
# STORE COUNTS
# =========================================================

all_counts = []


# =========================================================
# READ EACH SPLIT
# =========================================================

for split_name, file_path in FILES.items():

    print(f"\nProcessing {split_name}...")

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        # CIC-IDS2017 timestamps use day/month/year.
        timestamp = pd.to_datetime(
            chunk["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        temp = pd.DataFrame({
            "Date": timestamp.dt.strftime("%Y-%m-%d"),
            "Label": chunk["Label"].astype(int),
        })

        temp["Split"] = split_name

        all_counts.append(temp)


# =========================================================
# COMBINE
# =========================================================

data = pd.concat(
    all_counts,
    ignore_index=True,
)


# =========================================================
# LABEL DISTRIBUTION BY DATE
# =========================================================

print("\n" + "-" * 60)
print("ATTACK DISTRIBUTION BY DAY")
print("-" * 60)

date_label_counts = pd.crosstab(
    data["Date"],
    data["Label"],
)

print(date_label_counts)


# =========================================================
# HUMAN-READABLE LABEL NAMES
# =========================================================

label_names = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS",
}

date_label_counts = date_label_counts.rename(
    columns=label_names
)

print("\nReadable labels:")

print(date_label_counts)


# =========================================================
# ATTACK / BENIGN SUMMARY
# =========================================================

data["Class Type"] = data["Label"].apply(
    lambda x: "BENIGN" if x == 0 else "ATTACK"
)

daily_summary = pd.crosstab(
    data["Date"],
    data["Class Type"],
)

daily_summary["Total"] = daily_summary.sum(axis=1)

daily_summary["Attack %"] = (
    daily_summary["ATTACK"]
    / daily_summary["Total"]
    * 100
).round(3)


print("\n" + "-" * 60)
print("DAILY BENIGN / ATTACK SUMMARY")
print("-" * 60)

print(daily_summary)


# =========================================================
# WHICH ATTACKS APPEAR ON EACH DAY?
# =========================================================

print("\n" + "-" * 60)
print("ATTACK FAMILIES PRESENT ON EACH DAY")
print("-" * 60)

for date in sorted(data["Date"].dropna().unique()):

    labels = (
        data.loc[
            (data["Date"] == date)
            & (data["Label"] != 0),
            "Label",
        ]
        .unique()
    )

    readable_labels = [
        label_names[int(label)]
        for label in sorted(labels)
    ]

    print(f"\n{date}")

    if readable_labels:
        for label in readable_labels:
            print("  -", label)
    else:
        print("  - BENIGN ONLY")


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("DAY / ATTACK DISTRIBUTION COMPLETE")
print("=" * 60)