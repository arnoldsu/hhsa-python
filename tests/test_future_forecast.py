import numpy as np

from hhsa.envelope_ml import build_envelope_dataset


def test_latest_origin_padding(tmp_path):
    n, carrier_count, modulation_count = 100, 6, 4
    dates = (
        np.datetime64("2000-01") + np.arange(n)
    ).astype("datetime64[D]").astype(str)
    signal = np.sin(np.arange(n) / 10)
    imf = np.tile(signal[:, None], (1, carrier_count))
    envelope = np.ones_like(imf)
    second = np.ones((n, modulation_count, carrier_count))
    path = tmp_path / "features.npz"
    np.savez(
        path, date=dates, nino34_anomaly_c=signal,
        sample_rate_per_year=np.array(12.0),
        IMF=imf, fm=envelope, am=envelope,
        IMF2=second, FM=second, AM=second,
    )
    dataset = build_envelope_dataset(
        path, lead=3, carrier_modes=(3, 4),
        modulation_modes=((1, 2), (0, 1)),
        include_latest_origin=True,
    )
    assert dataset.origin_date[-1] == np.datetime64(dates[-1])
    assert dataset.target_date[-1] == (
        np.datetime64(dates[-1]).astype("datetime64[M]") + 3
    ).astype("datetime64[D]")
    assert np.isnan(dataset.target[-1])
    assert np.isfinite(dataset.carrier_features[0][-1]).all()

