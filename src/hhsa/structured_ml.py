"""HHSA-structured XGBoost forecasting with an explicit multiplicative AM gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .instantaneous import direct_quadrature
from .ml import ForecastDataset, chronological_masks, load_forecast_dataset


@dataclass(frozen=True)
class StructuredResult:
    dataset: ForecastDataset
    masks: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]


def _xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=8.0,
        n_jobs=1,
        random_state=seed,
    )


def _oof_prediction(x: np.ndarray, y: np.ndarray, seed: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-time out-of-fold predictions for honest gate residuals."""
    prediction = np.full(y.shape, np.nan)
    splitter = TimeSeriesSplit(n_splits=5)
    for fold, (fit, evaluate) in enumerate(splitter.split(x)):
        model = _xgb(seed + fold)
        model.fit(x[fit], y[fit])
        prediction[evaluate] = model.predict(x[evaluate])
    valid = np.isfinite(prediction)
    return prediction, valid


def _fit_predict(fit_x: np.ndarray, fit_y: np.ndarray,
                 evaluate_x: np.ndarray, seed: int) -> np.ndarray:
    model = _xgb(seed)
    model.fit(fit_x, fit_y)
    return model.predict(evaluate_x)


def run_structured_forecast(
    feature_path: str | Path,
    lead: int,
    train_end: str,
    validation_end: str,
    *,
    seed: int = 42,
    carrier_modes: tuple[int, ...] = (3, 4),
) -> StructuredResult:
    """Predict latent amplitude/phase, apply an AM gate, then reconstruct."""
    dataset = load_forecast_dataset(feature_path, lead)
    masks = chronological_masks(dataset, train_end, validation_end)
    with np.load(feature_path) as archive:
        signal = archive["nino34_anomaly_c"].astype(float)
        imf = archive["IMF"].astype(float)
        sample_rate = float(archive["sample_rate_per_year"])

    _, quadrature_amplitude, phase = direct_quadrature(
        imf[:, carrier_modes], sample_rate
    )
    max_lag = 12
    origins = np.arange(max_lag, signal.size - lead)
    targets = origins + lead
    if origins.size != dataset.target.size:
        raise ValueError("structured target alignment differs from ML dataset")

    base_x = dataset.features["enso_envelope"]
    selected_x = dataset.features["enso_selected_am"]
    am_x = selected_x[:, base_x.shape[1]:]
    epsilon = 1e-6
    log_amplitude_target = np.log(
        np.maximum(quadrature_amplitude[targets], epsilon)
    )
    target_phase = phase[targets]
    true_carrier = np.sum(
        quadrature_amplitude[targets] * np.cos(target_phase), axis=1
    )
    residual_target = dataset.target - true_carrier
    output = {
        name: np.full(dataset.target.shape, np.nan)
        for name in ("structured_no_gate", "structured_shuffled_am",
                     "structured_real_am")
    }
    output["persistence"] = dataset.persistence.copy()

    for evaluation_index, (evaluation, fit_mask) in enumerate((
        ("validation", masks["train"]),
        ("test", masks["train"] | masks["validation"]),
    )):
        evaluate_mask = masks[evaluation]
        x_fit, x_evaluate = base_x[fit_mask], base_x[evaluate_mask]
        am_fit, am_evaluate = am_x[fit_mask], am_x[evaluate_mask]
        reconstructed = {
            "structured_no_gate": np.zeros(evaluate_mask.sum()),
            "structured_shuffled_am": np.zeros(evaluate_mask.sum()),
            "structured_real_am": np.zeros(evaluate_mask.sum()),
        }

        for carrier_index in range(len(carrier_modes)):
            target_log_amp = log_amplitude_target[fit_mask, carrier_index]
            oof_base, valid = _oof_prediction(
                x_fit, target_log_amp, seed + 100 * evaluation_index + carrier_index
            )
            base_model = _xgb(seed + 200 + carrier_index)
            base_model.fit(x_fit, target_log_amp)
            base_log_evaluate = base_model.predict(x_evaluate)

            gate_target = target_log_amp[valid] - oof_base[valid]
            gate_fit_x = np.column_stack((am_fit[valid], oof_base[valid]))
            gate_evaluate_x = np.column_stack((am_evaluate, base_log_evaluate))
            real_gate = _fit_predict(
                gate_fit_x, gate_target, gate_evaluate_x,
                seed + 300 + carrier_index,
            )
            rng = np.random.default_rng(seed + 400 + carrier_index)
            shuffled_gate_x = gate_fit_x.copy()
            shuffled_gate_x[:, :-1] = shuffled_gate_x[
                rng.permutation(shuffled_gate_x.shape[0]), :-1
            ]
            shuffled_gate = _fit_predict(
                shuffled_gate_x, gate_target, gate_evaluate_x,
                seed + 500 + carrier_index,
            )

            sin_future = _fit_predict(
                x_fit, np.sin(target_phase[fit_mask, carrier_index]),
                x_evaluate, seed + 600 + carrier_index,
            )
            cos_future = _fit_predict(
                x_fit, np.cos(target_phase[fit_mask, carrier_index]),
                x_evaluate, seed + 700 + carrier_index,
            )
            norm = np.maximum(np.hypot(sin_future, cos_future), epsilon)
            cos_future /= norm

            amplitudes = {
                "structured_no_gate": np.exp(base_log_evaluate),
                "structured_shuffled_am": np.exp(
                    base_log_evaluate + shuffled_gate
                ),
                "structured_real_am": np.exp(base_log_evaluate + real_gate),
            }
            for name, amplitude in amplitudes.items():
                reconstructed[name] += amplitude * cos_future

        residual = _fit_predict(
            x_fit, residual_target[fit_mask], x_evaluate,
            seed + 800 + evaluation_index,
        )
        for name in reconstructed:
            output[name][evaluate_mask] = reconstructed[name] + residual

    return StructuredResult(dataset, masks, output)

