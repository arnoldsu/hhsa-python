#!/usr/bin/env python3
"""Download/convert NOAA PSL standard-format monthly Nino 3.4 anomalies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

import numpy as np

URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"


def parse_standard(text: str):
    lines = text.splitlines()
    first, last = map(int, lines[0].split()[:2])
    dates, values = [], []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 13:
            continue
        try:
            year = int(fields[0])
            monthly = [float(v) for v in fields[1:]]
        except ValueError:
            continue
        if not first <= year <= last:
            continue
        for month, value in enumerate(monthly, 1):
            if value > -99:
                dates.append(f"{year:04d}-{month:02d}-01")
                values.append(value)
    return np.asarray(dates), np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/nino34_noaa_raw.data"))
    parser.add_argument("--csv", type=Path, default=Path("data/nino34_monthly.csv"))
    parser.add_argument("--npz", type=Path, default=Path("data/nino34_monthly.npz"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download or not args.raw.exists():
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        args.raw.write_bytes(urlopen(URL).read())
    dates, values = parse_standard(args.raw.read_text())
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "nino34_anomaly_c"])
        writer.writerows(zip(dates, values))
    np.savez_compressed(args.npz, date=dates, nino34_anomaly_c=values,
                        sample_rate_per_year=np.array(12.0), source=np.array(URL))
    print(f"Converted {values.size} monthly values ({dates[0]} to {dates[-1]})")
    print(f"Saved {args.csv} and {args.npz}")


if __name__ == "__main__":
    main()
