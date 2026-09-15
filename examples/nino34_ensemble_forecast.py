#!/usr/bin/env python3
"""Ten-member HHSA envelope × FM-carrier ensemble through December 2027."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hhsa.envelope_ml import forecast_from_latest_origin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/nino34_hhsa_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/ml_envelope_multiply"))
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--max-lead", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.members < 2:
        raise ValueError("members must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for member in range(args.members):
        member_seed = args.seed + member
        for lead in range(1, args.max_lead + 1):
            row = forecast_from_latest_origin(
                args.features, lead, seed=member_seed
            )
            row["member"] = member + 1
            row["seed"] = member_seed
            rows.append(row)
    members = pd.DataFrame(rows).sort_values(["lead_months", "member"])
    member_path = args.output_dir / "ensemble_members_to_2027-12.csv"
    members.to_csv(member_path, index=False)

    value_columns = ["hhsa_no_lf_am", "hhsa_real_lf_am"]
    summary_rows = []
    for lead, group in members.groupby("lead_months", sort=True):
        row = {
            "lead_months": lead,
            "origin_date": group["origin_date"].iloc[0],
            "target_date": group["target_date"].iloc[0],
        }
        for column in value_columns:
            values = group[column].to_numpy()
            row[f"{column}_mean"] = np.mean(values)
            row[f"{column}_median"] = np.median(values)
            row[f"{column}_p10"] = np.quantile(values, 0.10)
            row[f"{column}_p90"] = np.quantile(values, 0.90)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "ensemble_summary_to_2027-12.csv"
    summary.to_csv(summary_path, index=False)

    dates = pd.to_datetime(summary["target_date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    for axis, column, title in zip(
        axes, value_columns, ("Envelope history only", "Real 10–25 year LF-AM")
    ):
        for member in range(1, args.members + 1):
            group = members[members["member"] == member]
            axis.plot(pd.to_datetime(group["target_date"]), group[column],
                      color="tab:blue", alpha=0.25, lw=1)
        axis.fill_between(
            dates, summary[f"{column}_p10"], summary[f"{column}_p90"],
            color="tab:orange", alpha=0.25, label="10–90% ensemble interval",
        )
        axis.plot(dates, summary[f"{column}_mean"], color="tab:red", lw=2,
                  label="Ensemble mean")
        axis.plot(dates, summary[f"{column}_median"], color="black", ls="--",
                  lw=1.2, label="Ensemble median")
        axis.axhline(0, color="0.6", lw=0.8)
        axis.set(title=title, ylabel="Niño3.4 anomaly (°C)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Target month")
    figure_path = args.output_dir / "ensemble_forecast_to_2027-12.png"
    fig.savefig(figure_path, dpi=180)

    print(f"Generated {args.members} members with seeds "
          f"{args.seed}–{args.seed + args.members - 1}.")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {member_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()

