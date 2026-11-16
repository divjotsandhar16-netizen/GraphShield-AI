import pandas as pd

FILES = {
    "TRAIN": "data/processed/train.csv",
    "VALIDATION": "data/processed/validation.csv",
    "TEST": "data/processed/test.csv"
}

print("=" * 60)
print("GRAPHSHIELD AI - ALL SPLITS DATA QUALITY CHECK")
print("=" * 60)

for split, file in FILES.items():

    print("\n" + "-" * 60)
    print(split)
    print("-" * 60)

    bad_header_segment = 0
    bad_rate_duration = 0
    total_rows = 0

    bad_header_labels = {}
    bad_rate_labels = {}

    usecols = [
        "Fwd Header Length",
        "Bwd Header Length",
        "min_seg_size_forward",
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Label"
    ]

    for chunk in pd.read_csv(
        file,
        usecols=usecols,
        chunksize=200_000,
        low_memory=False
    ):

        total_rows += len(chunk)

        # Impossible physical measurements
        header_mask = (
            (chunk["Fwd Header Length"] < 0)
            | (chunk["Bwd Header Length"] < 0)
            | (chunk["min_seg_size_forward"] < 0)
        )

        bad_header_segment += int(header_mask.sum())

        for label, count in chunk.loc[
            header_mask, "Label"
        ].value_counts().items():

            bad_header_labels[label] = (
                bad_header_labels.get(label, 0) + int(count)
            )

        # Invalid negative duration/rates
        rate_mask = (
            (chunk["Flow Duration"] < 0)
            | (chunk["Flow Bytes/s"] < 0)
            | (chunk["Flow Packets/s"] < 0)
        )

        bad_rate_duration += int(rate_mask.sum())

        for label, count in chunk.loc[
            rate_mask, "Label"
        ].value_counts().items():

            bad_rate_labels[label] = (
                bad_rate_labels.get(label, 0) + int(count)
            )

    print(f"Total rows: {total_rows:,}")

    print(
        f"Invalid header/segment rows: "
        f"{bad_header_segment:,}"
    )

    print(
        "Labels:",
        bad_header_labels
    )

    print(
        f"Negative duration/rate rows: "
        f"{bad_rate_duration:,}"
    )

    print(
        "Labels:",
        bad_rate_labels
    )

print("\n" + "=" * 60)
print("ALL SPLITS CHECK COMPLETE")
print("=" * 60)