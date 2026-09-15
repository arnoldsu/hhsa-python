import numpy as np

from hhsa.structured_ml import _oof_prediction


def test_oof_prediction_is_strictly_forward():
    x = np.arange(120, dtype=float)[:, None]
    y = np.sin(x[:, 0] / 10)
    prediction, valid = _oof_prediction(x, y, seed=1)
    assert valid.sum() > 0
    assert not valid[:20].any()
    assert np.isfinite(prediction[valid]).all()

