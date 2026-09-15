import numpy as np

from hhsa.envelope_ml import _safe_envelope


def test_envelope_multiplication_reconstructs_imf():
    imf = np.array([-2.0, -0.5, 0.0, 0.75, 3.0])
    envelope = _safe_envelope(np.array([2.0, 1.0, -1e-5, 1.5, 3.0]))
    carrier = imf / envelope
    np.testing.assert_allclose(envelope * carrier, imf)
    assert np.all(envelope > 0)

