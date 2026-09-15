"""Forecast HHSA envelopes and normalized FM carriers, then multiply them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


@dataclass(frozen=True)
class EnvelopeDataset:
    origin_date: np.ndarray
    target_date: np.ndarray
    target: np.ndarray
    persistence: np.ndarray
    carrier_target: np.ndarray
    envelope_target: np.ndarray
    residual_target: np.ndarray
    carrier_features: list[np.ndarray]
    envelope_base_features: list[np.ndarray]
    envelope_lf_features: list[np.ndarray]
    residual_features: np.ndarray


def _xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=8.0,
        n_jobs=1,
        random_state=seed,
    )


def _lag_columns(values: np.ndarray, rows: np.ndarray,
                 lags: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return np.concatenate([values[rows - lag] for lag in lags], axis=1)


def _safe_envelope(envelope: np.ndarray) -> np.ndarray:
    """Repair tiny cubic-spline undershoots while preserving envelope scale."""
    positive = envelope[envelope > 0]
    floor = max(np.quantile(positive, 0.001) * 0.1, 1e-6)
    return np.maximum(envelope, floor)


def build_envelope_dataset(
    path: str | Path,
    lead: int,
    *,
    carrier_modes: tuple[int, ...] = (3, 4),
    modulation_modes: tuple[tuple[int, ...], ...] = ((1, 2), (0, 1)),
    history: int = 60,
    include_latest_origin: bool = False,
) -> EnvelopeDataset:
    """Create separate targets for the AM envelope and normalized FM carrier."""
    if lead <= 0:
        raise ValueError("lead must be positive")
    if len(carrier_modes) != len(modulation_modes):
        raise ValueError("each carrier must have selected modulation modes")
    with np.load(path) as archive:
        dates = archive["date"].astype("datetime64[D]")
        signal = archive["nino34_anomaly_c"].astype(float)
        imf = archive["IMF"].astype(float)
        fm = archive["fm"].astype(float)
        envelope_raw = archive["am"].astype(float)
        imf2 = archive["IMF2"].astype(float)
        modulation_frequency = archive["FM"].astype(float)
        modulation_amplitude = archive["AM"].astype(float)

    if include_latest_origin:
        original_size = signal.size
        future_months = (
            dates[-1].astype("datetime64[M]")
            + np.arange(1, lead + 1)
        ).astype("datetime64[D]")
        dates = np.concatenate((dates, future_months))
        signal = np.pad(signal, (0, lead), constant_values=np.nan)
        imf = np.pad(imf, ((0, lead), (0, 0)), constant_values=np.nan)
        fm = np.pad(fm, ((0, lead), (0, 0)), constant_values=np.nan)
        envelope_raw = np.pad(
            envelope_raw, ((0, lead), (0, 0)), constant_values=np.nan
        )
        imf2 = np.pad(
            imf2, ((0, lead), (0, 0), (0, 0)), constant_values=np.nan
        )
        modulation_frequency = np.pad(
            modulation_frequency, ((0, lead), (0, 0), (0, 0)),
            constant_values=np.nan,
        )
        modulation_amplitude = np.pad(
            modulation_amplitude, ((0, lead), (0, 0), (0, 0)),
            constant_values=np.nan,
        )
    else:
        original_size = signal.size

    n = signal.size
    if n <= history + lead:
        raise ValueError("record is too short for history and lead")
    rows = np.arange(history, n - lead)
    targets = rows + lead
    envelope = np.column_stack([
        np.r_[
            _safe_envelope(envelope_raw[:original_size, mode]),
            envelope_raw[original_size:, mode],
        ]
        for mode in carrier_modes
    ])
    carrier = imf[:, carrier_modes] / envelope
    selected_imf = imf[:, carrier_modes]
    residual = signal - selected_imf.sum(axis=1)

    carrier_lags = tuple(range(25))
    envelope_lags = tuple(range(history + 1))
    lf_lags = (0, 1, 3, 6, 12, 24, 36, 48, 60)
    raw_lags = tuple(range(13))
    month = dates[rows].astype("datetime64[M]").astype(int) % 12
    seasonal = np.column_stack((
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ))

    carrier_features, envelope_base_features, envelope_lf_features = [], [], []
    for local_index, (carrier_mode, selected_modulations) in enumerate(
        zip(carrier_modes, modulation_modes)
    ):
        carrier_features.append(np.column_stack((
            _lag_columns(carrier[:, local_index], rows, carrier_lags),
            _lag_columns(fm[:, carrier_mode], rows, raw_lags),
            seasonal,
        )))
        base_envelope = _lag_columns(
            np.log(envelope[:, local_index]), rows, envelope_lags
        )
        delta = np.diff(np.log(envelope[:, local_index]), prepend=np.nan)
        delta[0] = 0
        envelope_base_features.append(np.column_stack((
            base_envelope,
            _lag_columns(delta, rows, lf_lags),
        )))
        lf_state = np.column_stack([
            source[:, modulation, carrier_mode]
            for source in (imf2, modulation_frequency, modulation_amplitude)
            for modulation in selected_modulations
        ])
        envelope_lf_features.append(
            _lag_columns(lf_state, rows, lf_lags)
        )

    return EnvelopeDataset(
        origin_date=dates[rows],
        target_date=dates[targets],
        target=signal[targets],
        persistence=signal[rows],
        carrier_target=carrier[targets],
        envelope_target=envelope[targets],
        residual_target=residual[targets],
        carrier_features=carrier_features,
        envelope_base_features=envelope_base_features,
        envelope_lf_features=envelope_lf_features,
        residual_features=np.column_stack((
            _lag_columns(signal, rows, raw_lags),
            _lag_columns(residual, rows, raw_lags),
            seasonal,
        )),
    )


def forecast_from_latest_origin(
    path: str | Path, lead: int, *, seed: int = 42,
) -> dict[str, float | str | int]:
    """Fit all available labelled origins and forecast from the last observation."""
    dataset = build_envelope_dataset(path, lead, include_latest_origin=True)
    fit = np.isfinite(dataset.target)
    evaluate = np.zeros(dataset.target.shape, dtype=bool)
    evaluate[-1] = True
    if not fit.any() or fit[-1]:
        raise ValueError("future forecast row was not constructed correctly")

    residual_model = _xgb(seed)
    residual_model.fit(
        dataset.residual_features[fit], dataset.residual_target[fit]
    )
    residual = float(residual_model.predict(
        dataset.residual_features[evaluate]
    )[0])
    no_lf_prediction = residual
    real_lf_prediction = residual
    envelope_predictions = []
    carrier_predictions = []

    for carrier_index in range(dataset.carrier_target.shape[1]):
        carrier_model = _xgb(seed + 100 + carrier_index)
        carrier_model.fit(
            dataset.carrier_features[carrier_index][fit],
            dataset.carrier_target[fit, carrier_index],
        )
        carrier = float(np.clip(carrier_model.predict(
            dataset.carrier_features[carrier_index][evaluate]
        )[0], -1.25, 1.25))

        target = np.log(dataset.envelope_target[fit, carrier_index])
        base_fit = dataset.envelope_base_features[carrier_index][fit]
        base_evaluate = dataset.envelope_base_features[carrier_index][evaluate]
        lf_fit = dataset.envelope_lf_features[carrier_index][fit]
        lf_evaluate = dataset.envelope_lf_features[carrier_index][evaluate]

        no_lf_model = _xgb(seed + 200 + carrier_index)
        no_lf_model.fit(base_fit, target)
        no_lf_envelope = float(np.exp(
            no_lf_model.predict(base_evaluate)[0]
        ))
        real_model = _xgb(seed + 300 + carrier_index)
        real_model.fit(np.column_stack((base_fit, lf_fit)), target)
        real_envelope = float(np.exp(real_model.predict(
            np.column_stack((base_evaluate, lf_evaluate))
        )[0]))

        no_lf_prediction += no_lf_envelope * carrier
        real_lf_prediction += real_envelope * carrier
        carrier_predictions.append(carrier)
        envelope_predictions.append(real_envelope)

    return {
        "lead_months": lead,
        "origin_date": str(dataset.origin_date[-1]),
        "target_date": str(dataset.target_date[-1]),
        "hhsa_no_lf_am": no_lf_prediction,
        "hhsa_real_lf_am": real_lf_prediction,
        "envelope_imf3": envelope_predictions[0],
        "envelope_imf4": envelope_predictions[1],
        "normalized_carrier_imf3": carrier_predictions[0],
        "normalized_carrier_imf4": carrier_predictions[1],
        "predicted_residual": residual,
    }


def chronological_masks(dataset: EnvelopeDataset, train_end: str,
                        validation_end: str) -> dict[str, np.ndarray]:
    train_boundary = np.datetime64(train_end)
    validation_boundary = np.datetime64(validation_end)
    target = dataset.target_date
    masks = {
        "train": target <= train_boundary,
        "validation": (target > train_boundary) & (target <= validation_boundary),
        "test": target > validation_boundary,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("each chronological split must contain observations")
    return masks


def run_envelope_forecast(
    path: str | Path, lead: int, train_end: str, validation_end: str,
    *, seed: int = 42,
) -> tuple[EnvelopeDataset, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit real/shuffled/no-LF envelope forecasts and multiply by FM carrier."""
    dataset = build_envelope_dataset(path, lead)
    masks = chronological_masks(dataset, train_end, validation_end)
    names = (
        "hhsa_no_lf_am", "hhsa_shuffled_lf_am", "hhsa_real_lf_am",
        "oracle_envelope", "oracle_carrier",
    )
    predictions = {name: np.full(dataset.target.shape, np.nan) for name in names}
    predictions["persistence"] = dataset.persistence.copy()

    for evaluation_index, (evaluation, fit_mask) in enumerate((
        ("validation", masks["train"]),
        ("test", masks["train"] | masks["validation"]),
    )):
        evaluate_mask = masks[evaluation]
        residual_model = _xgb(seed + 10 * evaluation_index)
        residual_model.fit(
            dataset.residual_features[fit_mask],
            dataset.residual_target[fit_mask],
        )
        residual_prediction = residual_model.predict(
            dataset.residual_features[evaluate_mask]
        )
        reconstructed = {
            name: residual_prediction.copy() for name in names
        }

        for carrier_index in range(dataset.carrier_target.shape[1]):
            carrier_model = _xgb(seed + 100 + carrier_index)
            carrier_model.fit(
                dataset.carrier_features[carrier_index][fit_mask],
                dataset.carrier_target[fit_mask, carrier_index],
            )
            carrier_prediction = np.clip(
                carrier_model.predict(
                    dataset.carrier_features[carrier_index][evaluate_mask]
                ),
                -1.25, 1.25,
            )
            target_log_envelope = np.log(
                dataset.envelope_target[fit_mask, carrier_index]
            )
            base_fit = dataset.envelope_base_features[carrier_index][fit_mask]
            base_evaluate = dataset.envelope_base_features[carrier_index][
                evaluate_mask
            ]
            lf_fit = dataset.envelope_lf_features[carrier_index][fit_mask]
            lf_evaluate = dataset.envelope_lf_features[carrier_index][
                evaluate_mask
            ]

            no_lf_model = _xgb(seed + 200 + carrier_index)
            no_lf_model.fit(base_fit, target_log_envelope)
            no_lf_envelope = np.exp(no_lf_model.predict(base_evaluate))

            real_model = _xgb(seed + 300 + carrier_index)
            real_model.fit(
                np.column_stack((base_fit, lf_fit)), target_log_envelope
            )
            real_envelope = np.exp(real_model.predict(
                np.column_stack((base_evaluate, lf_evaluate))
            ))

            rng = np.random.default_rng(seed + 400 + carrier_index)
            shuffled_lf = lf_fit[rng.permutation(lf_fit.shape[0])]
            shuffled_model = _xgb(seed + 500 + carrier_index)
            shuffled_model.fit(
                np.column_stack((base_fit, shuffled_lf)), target_log_envelope
            )
            shuffled_envelope = np.exp(shuffled_model.predict(
                np.column_stack((base_evaluate, lf_evaluate))
            ))

            true_envelope = dataset.envelope_target[
                evaluate_mask, carrier_index
            ]
            true_carrier = dataset.carrier_target[evaluate_mask, carrier_index]
            reconstructed["hhsa_no_lf_am"] += (
                no_lf_envelope * carrier_prediction
            )
            reconstructed["hhsa_shuffled_lf_am"] += (
                shuffled_envelope * carrier_prediction
            )
            reconstructed["hhsa_real_lf_am"] += (
                real_envelope * carrier_prediction
            )
            reconstructed["oracle_envelope"] += (
                true_envelope * carrier_prediction
            )
            reconstructed["oracle_carrier"] += (
                real_envelope * true_carrier
            )

        for name, values in reconstructed.items():
            predictions[name][evaluate_mask] = values
    return dataset, masks, predictions
