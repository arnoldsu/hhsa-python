"""EMD and masking-EMD routines ported from the supplied MATLAB code."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``extr`` including flat extrema and zero plateaus."""
    x = np.asarray(x, dtype=float).ravel()
    d = np.diff(x)
    d1, d2 = d[:-1], d[1:]
    minima = list(np.flatnonzero((d1 * d2 < 0) & (d1 < 0)) + 1)
    maxima = list(np.flatnonzero((d1 * d2 < 0) & (d1 > 0)) + 1)
    zero = list(np.flatnonzero(x[:-1] * x[1:] < 0))
    zidx = np.flatnonzero(x == 0)
    if zidx.size:
        cuts = np.flatnonzero(np.diff(zidx) > 1) + 1
        for group in np.split(zidx, cuts):
            zero.append(matlab_round((group[0] + group[-1]) / 2))
    flat = d == 0
    edges = np.diff(np.r_[False, flat, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        if start == 0 or end == d.size:
            continue
        middle = matlab_round((start + end) / 2)
        if d[start - 1] > 0 and d[end] < 0:
            maxima.append(middle)
        elif d[start - 1] < 0 and d[end] > 0:
            minima.append(middle)
    return np.array(sorted(set(minima)), int), np.array(sorted(set(maxima)), int), np.array(sorted(set(zero)), int)


def matlab_round(x):
    """MATLAB round: halves away from zero (unlike NumPy bankers rounding)."""
    x = np.asarray(x)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(out) if out.ndim == 0 else out.astype(int)


def _control_points(x: np.ndarray, indices: np.ndarray, upper: bool) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint treatment used by ``emdx/extrema_x``."""
    n = x.size
    points = np.r_[0, indices, n - 1]
    values = x[points].copy()
    if points.size >= 4:
        left = values[1] + (points[0] - points[1]) * (values[1] - values[2]) / (points[1] - points[2])
        right = values[-2] + (points[-1] - points[-2]) * (values[-2] - values[-3]) / (points[-2] - points[-3])
        if upper:
            values[0], values[-1] = max(values[0], left), max(values[-1], right)
        else:
            values[0], values[-1] = min(values[0], left), min(values[-1], right)
    return points, values


def rcada_emd(signal: np.ndarray, max_imfs: int, siftings: int = 10) -> np.ndarray:
    """Portable source-level equivalent for the unavailable ``rcada_emd`` MEX.

    Implements the supplied ``emdx.m`` ten-sift algorithm, including its
    extrema and modified linear endpoint rules. Components sum exactly to the
    input; the last component is the residual.
    """
    y = np.asarray(signal, dtype=float).ravel()
    if y.size < 4 or not np.all(np.isfinite(y)):
        raise ValueError("signal must contain at least four finite samples")
    max_allowed = max(1, int(np.floor(np.log2(y.size))))
    count = max_allowed if max_imfs <= 0 or max_imfs > max_allowed else int(max_imfs)
    scale = np.std(y, ddof=1)
    if scale == 0:
        return y[:, None]
    residual = y / scale
    modes: list[np.ndarray] = []
    grid = np.arange(y.size)
    for _ in range(count - 1):
        candidate = residual.copy()
        imin, imax, _ = extrema(candidate)
        if imin.size == 0 or imax.size == 0:
            break
        for _ in range(siftings):
            imin, imax, _ = extrema(candidate)
            if imin.size == 0 or imax.size == 0:
                break
            px, py = _control_points(candidate, imax, True)
            nx, ny = _control_points(candidate, imin, False)
            upper = CubicSpline(px, py, bc_type="not-a-knot")(grid)
            lower = CubicSpline(nx, ny, bc_type="not-a-knot")(grid)
            candidate -= (upper + lower) / 2
        modes.append(candidate * scale)
        residual -= candidate
    modes.append(residual * scale)
    return np.column_stack(modes)


def resample_spline(x: np.ndarray, factor: int) -> np.ndarray:
    """Port of ``spmmhh_resamplev``."""
    x = np.asarray(x, dtype=float)
    old = np.arange(x.shape[0])
    new = np.linspace(0, x.shape[0] - 1, (x.shape[0] - 1) * factor + 1)
    return CubicSpline(old, x, axis=0, bc_type="not-a-knot")(new)


def masking_emd(
    signal: np.ndarray,
    max_imfs: int = -1,
    mask_order: int = 0,
    upsample_level: int = 0,
    amplitude_ratio: float = 2.0,
    frequency_ratio: float = 1.0,
    siftings: int = 10,
) -> np.ndarray:
    """Port of ``cmask_emd3GU.m`` using four-phase masking by default."""
    original = np.asarray(signal, dtype=float).ravel()
    factor = 2**int(upsample_level)
    work = resample_spline(original, factor) if factor > 1 else original.copy()
    requested = int(np.floor(np.log2(original.size))) if max_imfs <= 0 else int(max_imfs)
    total_requested = requested + upsample_level
    first = rcada_emd(work, 2, siftings)[:, 0]
    crossings = int(np.count_nonzero(np.r_[np.diff(first), 0] * np.r_[0, np.diff(first)] < 0))
    if crossings <= 2:
        return original[:, None]
    omega = factor * frequency_ratio * np.pi * crossings / (work.size - 1)
    n_mask_modes = min(int(np.floor(np.log2(crossings))) + upsample_level, total_requested - 1)
    modes = np.zeros((work.size, total_requested))
    time = np.arange(1, work.size + 1)
    residual = work.copy()
    phases = 2**mask_order
    for mode in range(n_mask_modes):
        base = rcada_emd(residual, 2, siftings)[:, 0]
        amplitude = np.full(work.size, amplitude_ratio * np.std(base, ddof=1))
        accum = np.zeros(work.size)
        for phase in range(phases):
            angle = phase * np.pi / (phases * 2)
            cosine = amplitude * np.cos(omega * time + angle)
            sine = amplitude * np.sin(omega * time + angle)
            for mask in (cosine, -cosine, sine, -sine):
                accum += rcada_emd(residual + mask, 2, siftings)[:, 0]
        modes[:, mode] = accum / (4 * phases)
        residual -= modes[:, mode]
        omega /= 2
    modes[:, n_mask_modes] = residual
    modes = modes[:, : n_mask_modes + 1]
    if factor > 1:
        modes = modes[::factor, upsample_level:]
        if modes.shape[0] < original.size:
            modes = np.vstack([modes, np.repeat(modes[-1:, :], original.size - modes.shape[0], axis=0)])
    return modes
