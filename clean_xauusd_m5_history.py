import csv
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE = ROOT / "XAUUSD_M5_HistData_2022-2026.csv"
OUTPUT = ROOT / "XAUUSD_M5_HistData_2022-2026_clean.csv"
FIELDS = ("time_est", "open", "high", "low", "close")


def main():
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows.sort(key=lambda row: row["time_est"])

    unique_rows = []
    duplicates = 0
    for row in rows:
        if unique_rows and row["time_est"] == unique_rows[-1]["time_est"]:
            previous = unique_rows[-1]
            if any(row[field] != previous[field] for field in FIELDS[1:]):
                raise RuntimeError(
                    f"Conflicting OHLC values at {row['time_est']}"
                )
            duplicates += 1
            continue
        unique_rows.append(row)

    if any(
        previous["time_est"] >= current["time_est"]
        for previous, current in zip(unique_rows, unique_rows[1:])
    ):
        raise RuntimeError("Cleaned M5 timestamps are not strictly increasing")

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(
        f"Created {OUTPUT}: {len(unique_rows)} unique bars; "
        f"removed {duplicates} exact duplicate timestamp rows."
    )


if __name__ == "__main__":
    main()
