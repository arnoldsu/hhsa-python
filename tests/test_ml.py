import numpy as np

from hhsa.ml import chronological_masks, load_forecast_dataset, regression_metrics


def test_forecast_dataset_alignment(tmp_path):
    n, carriers, modulations = 40, 2, 3
    dates = np.arange(
        np.datetime64("2000-01"), np.datetime64("2000-01") + n
    ).astype("datetime64[D]")
    signal = np.arange(n, dtype=float)
    first = np.ones((n, carriers))
    second = np.ones((n, modulations, carriers))
    path = tmp_path / "features.npz"
    np.savez(
        path, date=dates.astype(str), nino34_anomaly_c=signal,
        IMF=first, fm=first, am=first, IMF2=second, FM=second, AM=second,
    )
    dataset = load_forecast_dataset(
        path, lead=3, raw_lags=(0, 1, 2), hhsa_lags=(0, 2),
        carrier_modes=(0, 1), am_components=((1, 0), (2, 1)),
    )
    assert dataset.origin_date.size == n - 2 - 3
    np.testing.assert_array_equal(dataset.persistence, signal[2:-3])
    np.testing.assert_array_equal(dataset.target, signal[5:])
    assert dataset.features["raw"].shape[1] == 5
    assert dataset.features["hhsa_am"].shape[1] > dataset.features["hhsa_no_am"].shape[1]
    assert (
        dataset.features["enso_selected_am"].shape[1]
        > dataset.features["enso_envelope"].shape[1]
        > dataset.features["enso_carrier"].shape[1]
        > dataset.features["raw"].shape[1]
    )


def test_masks_and_perfect_metrics(tmp_path):
    observed = np.array([1.0, 2.0, 3.0])
    persistence = np.array([0.0, 1.0, 2.0])
    score = regression_metrics(observed, observed, persistence)
    assert score["rmse"] == 0
    assert np.isclose(score["acc"], 1)
    assert np.isclose(score["skill_vs_persistence"], 1)
