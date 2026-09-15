#!/usr/bin/env python3
"""Exploratory monthly Niño3.4 forecasts using saved offline HHSA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hhsa.ml import (
    chronological_masks,
    load_forecast_dataset,
    model_factories,
    regression_metrics,
    top_feature_correlations,
)


def run_lead(feature_path: Path, lead: int, train_end: str,
             validation_end: str, seed: int):
    dataset = load_forecast_dataset(feature_path, lead)
    masks = chronological_masks(dataset, train_end, validation_end)
    models = model_factories(seed)
    predictions: dict[str, np.ndarray] = {
        "persistence": dataset.persistence.copy()
    }
    metrics: list[dict[str, float | int | str]] = []

    for evaluation, fit_mask in (
        ("validation", masks["train"]),
        ("test", masks["train"] | masks["validation"]),
    ):
        evaluate_mask = masks[evaluation]
        baseline = regression_metrics(
            dataset.target[evaluate_mask],
            predictions["persistence"][evaluate_mask],
            dataset.persistence[evaluate_mask],
        )
        metrics.append({
            "lead_months": lead, "split": evaluation, "model": "persistence",
            "n": int(evaluate_mask.sum()), **baseline,
        })
        for model_name, (group, factory) in models.items():
            model = factory()
            model.fit(
                dataset.features[group][fit_mask],
                dataset.target[fit_mask],
            )
            predicted = model.predict(dataset.features[group][evaluate_mask])
            if evaluation == "test":
                predictions[model_name] = np.full(dataset.target.shape, np.nan)
                predictions[model_name][evaluate_mask] = predicted
            score = regression_metrics(
                dataset.target[evaluate_mask], predicted,
                dataset.persistence[evaluate_mask],
            )
            metrics.append({
                "lead_months": lead, "split": evaluation, "model": model_name,
                "n": int(evaluate_mask.sum()), **score,
            })

    test = masks["test"]
    frame = pd.DataFrame({
        "origin_date": dataset.origin_date[test].astype(str),
        "target_date": dataset.target_date[test].astype(str),
        "observed": dataset.target[test],
        **{name: values[test] for name, values in predictions.items()},
    })
    correlations = top_feature_correlations(dataset, masks["train"])
    return metrics, frame, correlations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/nino34_hhsa_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/ml_selected_am"))
    parser.add_argument("--leads", type=int, nargs="+", default=(3, 6, 12))
    parser.add_argument("--train-end", default="2013-12-01")
    parser.add_argument("--validation-end", default="2020-12-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, all_predictions, all_correlations = [], [], {}
    for lead in args.leads:
        metrics, predictions, correlations = run_lead(
            args.features, lead, args.train_end, args.validation_end, args.seed
        )
        all_metrics.extend(metrics)
        predictions.insert(0, "lead_months", lead)
        all_predictions.append(predictions)
        all_correlations[str(lead)] = correlations

    metric_frame = pd.DataFrame(all_metrics)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    metric_path = args.output_dir / "metrics.csv"
    prediction_path = args.output_dir / "predictions.csv"
    correlation_path = args.output_dir / "feature_correlations.json"
    metric_frame.to_csv(metric_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)
    correlation_path.write_text(json.dumps(all_correlations, indent=2) + "\n")

    fig, axes = plt.subplots(
        len(args.leads), 1, figsize=(13, 3.6 * len(args.leads)),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    for ax, lead in zip(axes[:, 0], args.leads):
        shown = prediction_frame[prediction_frame["lead_months"] == lead].copy()
        dates = pd.to_datetime(shown["target_date"])
        ax.plot(dates, shown["observed"], color="black", lw=1.5, label="Observed")
        ax.plot(dates, shown["persistence"], color="0.6", lw=1, label="Persistence")
        ax.plot(dates, shown["ridge_raw"], lw=1, label="Ridge raw")
        ax.plot(dates, shown["xgb_raw"], lw=1, label="XGB raw")
        ax.plot(dates, shown["xgb_enso_carrier"], lw=1,
                label="XGB ENSO carrier")
        ax.plot(dates, shown["xgb_enso_envelope"], lw=1,
                label="XGB carrier + envelope")
        ax.plot(dates, shown["xgb_enso_selected_am"], lw=1.3,
                label="XGB + selected AM")
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set(title=f"Niño3.4 test forecasts: lead {lead} months", ylabel="°C")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Target date")
    figure_path = args.output_dir / "test_predictions.png"
    fig.savefig(figure_path, dpi=180)

    print("EXPLORATORY ONLY: HHSA predictors were decomposed using the full record.")
    print(metric_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved {metric_path}")
    print(f"Saved {prediction_path}")
    print(f"Saved {correlation_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
