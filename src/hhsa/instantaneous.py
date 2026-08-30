"""Direct-quadrature frequency and amplitude from the original IFS code."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .emd import extrema, matlab_round, resample_spline


def _local_max_abs(x: np.ndarray) -> np.ndarray:
    # EMAX condition: left < centre and centre >= right.
    return np.flatnonzero((x[:-2] < x[1:-1]) & (x[1:-1] >= x[2:])) + 1


def pchip_normalize(data: np.ndarray, upsample: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``pchipnormalize.m`` including its interval fallback."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    work = resample_spline(values, upsample) if upsample > 1 else values
    normalized = np.zeros_like(work)
    amplitude = np.ones_like(work)
    grid = np.arange(work.shape[0])
    for column in range(work.shape[1]):
        peaks = _local_max_abs(np.abs(work[:, column]))
        if peaks.size <= 1:
            continue
        controls = np.r_[0, peaks, work.shape[0] - 1]
        heights = np.r_[np.abs(work[peaks[0], column]), np.abs(work[peaks, column]),
                        np.abs(work[peaks[-1], column])]
        env = PchipInterpolator(controls, heights)(grid)
        linear = np.interp(grid, controls, heights)
        for left, right in zip(controls[:-1], controls[1:]):
            sl = slice(left, right + 1)
            if np.any(env[sl] < work[sl, column]):
                env[sl] = linear[sl]
        amplitude[:, column] = env
        normalized[:, column] = np.divide(work[:, column], env, out=np.zeros_like(env), where=env != 0)
    if upsample > 1:
        normalized, amplitude = normalized[::upsample], amplitude[::upsample]
    return normalized[: values.shape[0]], amplitude[: values.shape[0]]


def direct_quadrature(data: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``fa(...,'quad')`` and ``FAquadrature.m``.

    Three PCHIP normalizations match the two passes in ``fa.m`` plus the
    additional pass inside ``FAquadrature.m``.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    total_amplitude = np.ones_like(values)
    normalized = values
    for _ in range(3):
        normalized, amplitude = pchip_normalize(normalized, 4)
        total_amplitude *= amplitude
    upsampled = resample_spline(normalized, 2)
    delta = np.diff(upsampled, axis=0)
    slope = np.zeros_like(normalized)
    slope[0] = 2 * delta[0]
    slope[-1] = 2 * delta[-1]
    slope[1:-1] = delta[1:-1:2] + delta[2::2]
    mask = np.where(slope > 0, -1.0, 1.0)
    quadrature = normalized + 1j * np.sqrt(np.maximum(0, 1 - normalized**2)) * mask
    phase = np.unwrap(np.angle(quadrature), axis=0)
    # The MATLAB code only repairs phase when it decreases. Preserve all
    # already-monotonic columns; enforce the same physical constraint on the rest.
    for column in range(phase.shape[1]):
        if np.any(np.diff(phase[:, column])[int(.05 * phase.shape[0]):-int(.05 * phase.shape[0])] < 0):
            phase[:, column] = np.maximum.accumulate(phase[:, column])
    frequency = np.empty_like(phase)
    dphase = np.diff(phase, axis=0)
    frequency[0], frequency[-1] = dphase[0], dphase[-1]
    frequency[1:-1] = 0.5 * (dphase[:-1] + dphase[1:])
    frequency *= sample_rate / (2 * np.pi)
    return frequency, total_amplitude, phase
