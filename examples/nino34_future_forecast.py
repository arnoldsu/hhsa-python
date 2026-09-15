#!/usr/bin/env python3
"""Forecast Niño3.4 from the last observation through December 2027."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hhsa.envelope_ml import forecast_from_latest_origin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/nino34_hhsa_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/ml_envelope_multiply"))
    parser.add_argument("--max-lead", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        forecast_from_latest_origin(args.features, lead, seed=args.seed)
        for lead in range(1, args.max_lead + 1)
    ]
    frame = pd.DataFrame(rows)
    csv_path = args.output_dir / "future_forecast_to_2027-12.csv"
    frame.to_csv(csv_path, index=False)

    metadata = {
        "status": "exploratory_offline_hhsa_forecast",
        "origin_date": rows[0]["origin_date"],
        "first_target": rows[0]["target_date"],
        "last_target": rows[-1]["target_date"],
        "method": (
            "Direct-horizon XGBoost forecasts of HHSA envelope and normalized "
            "FM carrier, combined by forced multiplication"
        ),
        "warning": (
            "The HHSA decomposition uses the full observed record and its last "
            "point is subject to EMD envelope end effects. This is not an "
            "operational causal forecast."
        ),
    }
    metadata_path = args.output_dir / "future_forecast_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    dates = pd.to_datetime(frame["target_date"])
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(dates, frame["hhsa_no_lf_am"], "o-", lw=1.2,
            label="Envelope history only")
    ax.plot(dates, frame["hhsa_real_lf_am"], "o-", lw=1.5,
            label="Real 10–25 year LF-AM")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(
        title="Exploratory Niño3.4 forecast from July 2026",
        xlabel="Target month", ylabel="Niño3.4 anomaly (°C)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    figure_path = args.output_dir / "future_forecast_to_2027-12.png"
    fig.savefig(figure_path, dpi=180)

    print(frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {csv_path}")
    print(f"Saved {metadata_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()

