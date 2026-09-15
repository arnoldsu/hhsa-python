#!/usr/bin/env python3
"""Forecast AM envelopes and normalized FM carriers, then multiply them."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hhsa.envelope_ml import run_envelope_forecast
from hhsa.ml import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/nino34_hhsa_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/ml_envelope_multiply"))
    parser.add_argument("--leads", type=int, nargs="+", default=(3, 6, 12))
    parser.add_argument("--train-end", default="2013-12-01")
    parser.add_argument("--validation-end", default="2020-12-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, frames = [], []
    for lead in args.leads:
        dataset, masks, predictions = run_envelope_forecast(
            args.features, lead, args.train_end, args.validation_end,
            seed=args.seed,
        )
        for split in ("validation", "test"):
            mask = masks[split]
            for model, prediction in predictions.items():
                score = regression_metrics(
                    dataset.target[mask], prediction[mask],
                    dataset.persistence[mask],
                )
                metrics.append({
                    "lead_months": lead, "split": split, "model": model,
                    "n": int(mask.sum()), **score,
                })
        test = masks["test"]
        frames.append(pd.DataFrame({
            "lead_months": lead,
            "origin_date": dataset.origin_date[test].astype(str),
            "target_date": dataset.target_date[test].astype(str),
            "observed": dataset.target[test],
            **{name: values[test] for name, values in predictions.items()},
        }))

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(frames, ignore_index=True)
    metric_path = args.output_dir / "metrics.csv"
    prediction_path = args.output_dir / "predictions.csv"
    metric_frame.to_csv(metric_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)

    fig, axes = plt.subplots(
        len(args.leads), 1, figsize=(13, 3.6 * len(args.leads)),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    lines = {
        "persistence": "Persistence",
        "hhsa_no_lf_am": "Envelope history only",
        "hhsa_shuffled_lf_am": "Shuffled LF-AM",
        "hhsa_real_lf_am": "Real LF-AM",
        "oracle_envelope": "Oracle envelope",
        "oracle_carrier": "Oracle carrier",
    }
    for ax, lead in zip(axes[:, 0], args.leads):
        shown = prediction_frame[prediction_frame["lead_months"] == lead]
        dates = pd.to_datetime(shown["target_date"])
        ax.plot(dates, shown["observed"], color="black", lw=1.5, label="Observed")
        for column, label in lines.items():
            style = "--" if column.startswith("oracle") else "-"
            ax.plot(dates, shown[column], style, lw=1, label=label)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set(title=f"Envelope × FM carrier: lead {lead} months", ylabel="°C")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Target date")
    figure_path = args.output_dir / "test_predictions.png"
    fig.savefig(figure_path, dpi=180)

    print("EXPLORATORY ONLY: HHSA decomposition uses the full record.")
    print("Oracle rows are diagnostics, not forecasts.")
    print(metric_frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {metric_path}")
    print(f"Saved {prediction_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()

