import numpy as np

from hhsa import decompose, project
from hhsa.emd import extrema, masking_emd, matlab_round, rcada_emd
from hhsa.instantaneous import direct_quadrature


def test_matlab_rounding_and_extrema():
    np.testing.assert_array_equal(matlab_round(np.array([-1.5, -.5, .5, 1.5])), [-2, -1, 1, 2])
    imin, imax, izero = extrema(np.array([-1., 0., 1., 1., 1., 0., -1.]))
    np.testing.assert_array_equal(imax, [3])
    np.testing.assert_array_equal(izero, [1, 5])


def test_emd_reconstruction():
    t = np.arange(256) / 128
    x = np.sin(2*np.pi*20*t) + .3*np.sin(2*np.pi*3*t)
    modes = rcada_emd(x, 5, siftings=4)
    np.testing.assert_allclose(modes.sum(1), x, atol=1e-10)


def test_pipeline_shapes():
    t = np.arange(384) / 128
    x = (1 + .3*np.sin(2*np.pi*2*t))*np.sin(2*np.pi*20*t)
    f, a, phase = direct_quadrature(x, 128)
    assert f.shape == a.shape == phase.shape == (x.size, 1)
    result = decompose(x, 128, max_imfs=4, max_modulation_imfs=3, upsample_level=0)
    spectrum = project(result, start=0, stop=x.size, time_bins=32)
    assert spectrum.power.shape[:3] == (57, 58, 32)
    assert np.all(spectrum.power >= 0)
