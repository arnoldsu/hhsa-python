#!/usr/bin/env python3
"""Test XGBoost inside an explicit HHSA amplitude/phase reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hhsa.ml import regression_metrics
from hhsa.structured_ml import run_structured_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/nino34_hhsa_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/ml_structured_hhsa"))
    parser.add_argument("--leads", type=int, nargs="+", default=(3, 6, 12))
    parser.add_argument("--train-end", default="2013-12-01")
    parser.add_argument("--validation-end", default="2020-12-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, prediction_frames = [], []
    for lead in args.leads:
        result = run_structured_forecast(
            args.features, lead, args.train_end, args.validation_end,
            seed=args.seed,
        )
        for split in ("validation", "test"):
            mask = result.masks[split]
            for model, prediction in result.predictions.items():
                score = regression_metrics(
                    result.dataset.target[mask], prediction[mask],
                    result.dataset.persistence[mask],
                )
                metrics.append({
                    "lead_months": lead, "split": split, "model": model,
                    "n": int(mask.sum()), **score,
                })
        test = result.masks["test"]
        prediction_frames.append(pd.DataFrame({
            "lead_months": lead,
            "origin_date": result.dataset.origin_date[test].astype(str),
            "target_date": result.dataset.target_date[test].astype(str),
            "observed": result.dataset.target[test],
            **{
                name: prediction[test]
                for name, prediction in result.predictions.items()
            },
        }))

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    metric_path = args.output_dir / "metrics.csv"
    prediction_path = args.output_dir / "predictions.csv"
    metric_frame.to_csv(metric_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)

    fig, axes = plt.subplots(
        len(args.leads), 1, figsize=(13, 3.6 * len(args.leads)),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    labels = {
        "persistence": "Persistence",
        "structured_no_gate": "HHSA gate=1",
        "structured_shuffled_am": "HHSA shuffled AM",
        "structured_real_am": "HHSA real AM",
    }
    for ax, lead in zip(axes[:, 0], args.leads):
        shown = prediction_frame[prediction_frame["lead_months"] == lead]
        dates = pd.to_datetime(shown["target_date"])
        ax.plot(dates, shown["observed"], color="black", lw=1.5, label="Observed")
        for column, label in labels.items():
            ax.plot(dates, shown[column], lw=1, label=label)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set(title=f"Structured HHSA-XGBoost: lead {lead} months", ylabel="°C")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Target date")
    figure_path = args.output_dir / "test_predictions.png"
    fig.savefig(figure_path, dpi=180)

    print("EXPLORATORY ONLY: latent HHSA states use the full record.")
    print(metric_frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {metric_path}")
    print(f"Saved {prediction_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()

