"""Time-ordered machine-learning utilities for monthly HHSA features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ForecastDataset:
    """Aligned predictors and a future Niño3.4 target."""

    origin_date: np.ndarray
    target_date: np.ndarray
    target: np.ndarray
    persistence: np.ndarray
    features: dict[str, np.ndarray]
    feature_names: dict[str, tuple[str, ...]]


def _lagged(values: np.ndarray, names: list[str], rows: np.ndarray,
            lags: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    matrices = [values[rows - lag] for lag in lags]
    columns = [f"{name}_lag{lag}" for lag in lags for name in names]
    return np.concatenate(matrices, axis=1), columns


def load_forecast_dataset(path: str | Path, lead: int, *,
                          raw_lags: tuple[int, ...] = tuple(range(13)),
                          hhsa_lags: tuple[int, ...] = (0, 1, 3, 6),
                          carrier_modes: tuple[int, ...] = (3, 4),
                          am_components: tuple[tuple[int, int], ...] = (
                              (1, 3), (2, 3), (0, 4), (1, 4)
                          ),
                          ) -> ForecastDataset:
    """Build physically selected ENSO carrier and AM groups without shuffling."""
    if lead <= 0:
        raise ValueError("lead must be a positive number of months")
    with np.load(path) as archive:
        dates = archive["date"].astype("datetime64[D]")
        signal = archive["nino34_anomaly_c"].astype(float)
        imf, fm, am = (archive[key].astype(float) for key in ("IMF", "fm", "am"))
        imf2, FM, AM = (
            archive[key].astype(float) for key in ("IMF2", "FM", "AM")
        )

    n = signal.size
    arrays = (imf, fm, am, imf2, FM, AM)
    if any(array.shape[0] != n for array in arrays):
        raise ValueError("all HHSA arrays must share the signal time axis")
    if any(mode < 0 or mode >= imf.shape[1] for mode in carrier_modes):
        raise ValueError("carrier mode index is outside the saved decomposition")
    if any(
        modulation < 0 or modulation >= imf2.shape[1]
        or carrier < 0 or carrier >= imf2.shape[2]
        for modulation, carrier in am_components
    ):
        raise ValueError("AM component index is outside the saved decomposition")
    max_lag = max((*raw_lags, *hhsa_lags))
    if n <= max_lag + lead:
        raise ValueError("not enough observations for requested lags and lead")
    rows = np.arange(max_lag, n - lead)

    raw, raw_names = _lagged(
        signal[:, None], ["nino34"], rows, raw_lags
    )
    month = dates[rows].astype("datetime64[M]").astype(int) % 12
    seasonal = np.column_stack((
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ))
    raw = np.column_stack((raw, seasonal))
    raw_names += ["month_sin", "month_cos"]

    carrier = imf[:, carrier_modes]
    carrier_names = [f"IMF_{mode}" for mode in carrier_modes]
    carrier_lagged, carrier_lagged_names = _lagged(
        carrier, carrier_names, rows, hhsa_lags
    )

    envelope = np.concatenate((fm[:, carrier_modes], am[:, carrier_modes]), axis=1)
    envelope_names = [
        f"{quantity}_{mode}"
        for quantity in ("fm", "am")
        for mode in carrier_modes
    ]
    envelope_lagged, envelope_lagged_names = _lagged(
        envelope, envelope_names, rows, hhsa_lags
    )

    selected_am = np.column_stack([
        source[:, modulation, carrier]
        for source in (imf2, FM, AM)
        for modulation, carrier in am_components
    ])
    selected_am_names = [
        f"{quantity}_m{modulation}_c{carrier}"
        for quantity in ("IMF2", "FM", "AM")
        for modulation, carrier in am_components
    ]
    selected_am_lagged, selected_am_lagged_names = _lagged(
        selected_am, selected_am_names, rows, hhsa_lags
    )

    # Retain the broad groups for diagnostic/backward compatibility, but the
    # selected-model experiment below uses only the physically focused groups.
    first = np.concatenate((imf, fm, am), axis=1)
    first_names = [
        f"{quantity}_{mode}"
        for quantity in ("IMF", "fm", "am")
        for mode in range(imf.shape[1])
    ]
    first_lagged, first_lagged_names = _lagged(
        first, first_names, rows, hhsa_lags
    )

    second = np.concatenate((
        imf2.reshape(n, -1),
        FM.reshape(n, -1),
        AM.reshape(n, -1),
    ), axis=1)
    second_names = [
        f"{quantity}_m{modulation}_c{carrier}"
        for quantity in ("IMF2", "FM", "AM")
        for modulation in range(imf2.shape[1])
        for carrier in range(imf2.shape[2])
    ]
    second_lagged, second_lagged_names = _lagged(
        second, second_names, rows, hhsa_lags
    )

    features = {
        "raw": raw,
        "enso_carrier": np.column_stack((raw, carrier_lagged)),
        "enso_envelope": np.column_stack((
            raw, carrier_lagged, envelope_lagged
        )),
        "enso_selected_am": np.column_stack((
            raw, carrier_lagged, envelope_lagged, selected_am_lagged
        )),
        "hhsa_no_am": np.column_stack((raw, first_lagged)),
        "hhsa_am": np.column_stack((raw, first_lagged, second_lagged)),
    }
    feature_names = {
        "raw": tuple(raw_names),
        "enso_carrier": tuple(raw_names + carrier_lagged_names),
        "enso_envelope": tuple(
            raw_names + carrier_lagged_names + envelope_lagged_names
        ),
        "enso_selected_am": tuple(
            raw_names + carrier_lagged_names + envelope_lagged_names
            + selected_am_lagged_names
        ),
        "hhsa_no_am": tuple(raw_names + first_lagged_names),
        "hhsa_am": tuple(raw_names + first_lagged_names + second_lagged_names),
    }
    return ForecastDataset(
        origin_date=dates[rows],
        target_date=dates[rows + lead],
        target=signal[rows + lead],
        persistence=signal[rows],
        features=features,
        feature_names=feature_names,
    )


def chronological_masks(dataset: ForecastDataset, train_end: str,
                        validation_end: str) -> dict[str, np.ndarray]:
    """Split by target date so training labels never cross the cutoff."""
    train_boundary = np.datetime64(train_end)
    validation_boundary = np.datetime64(validation_end)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must precede validation_end")
    target = dataset.target_date
    masks = {
        "train": target <= train_boundary,
        "validation": (target > train_boundary) & (target <= validation_boundary),
        "test": target > validation_boundary,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("each chronological split must contain observations")
    return masks


def regression_metrics(observed: np.ndarray, predicted: np.ndarray,
                       persistence: np.ndarray) -> dict[str, float]:
    """Return RMSE, anomaly correlation, and MSE skill over persistence."""
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    if np.std(observed) == 0 or np.std(predicted) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    baseline_mse = mean_squared_error(observed, persistence)
    mse = mean_squared_error(observed, predicted)
    skill = float(1 - mse / baseline_mse) if baseline_mse > 0 else float("nan")
    return {"rmse": rmse, "acc": correlation, "skill_vs_persistence": skill}


def model_factories(random_state: int = 42
                    ) -> dict[str, tuple[str, Callable[[], RegressorMixin]]]:
    """Models paired with the feature group they consume."""
    def ridge() -> RegressorMixin:
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    def xgb() -> RegressorMixin:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=5.0,
            n_jobs=1,
            random_state=random_state,
        )

    return {
        "ridge_raw": ("raw", ridge),
        "xgb_raw": ("raw", xgb),
        "xgb_enso_carrier": ("enso_carrier", xgb),
        "xgb_enso_envelope": ("enso_envelope", xgb),
        "xgb_enso_selected_am": ("enso_selected_am", xgb),
    }


def top_feature_correlations(dataset: ForecastDataset, train_mask: np.ndarray,
                             group: str = "enso_selected_am", top_n: int = 20
                             ) -> list[dict[str, float | str]]:
    """Rank univariate feature correlations using training data only."""
    x = dataset.features[group][train_mask]
    y = dataset.target[train_mask]
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean()
    denominator = np.sqrt((x_centered**2).sum(axis=0) * (y_centered**2).sum())
    correlation = np.divide(
        x_centered.T @ y_centered,
        denominator,
        out=np.zeros(x.shape[1], dtype=float),
        where=denominator > 0,
    )
    order = np.argsort(np.abs(correlation))[::-1][:top_n]
    names = dataset.feature_names[group]
    return [
        {"feature": names[index], "correlation": float(correlation[index])}
        for index in order
    ]
