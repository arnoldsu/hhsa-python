"""Faithful Python translation of ``multi_EMD_DCM_SV.m``."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline

from .emd import extrema, masking_emd
from .instantaneous import direct_quadrature


@dataclass(frozen=True)
class HHSAResult:
    fm: np.ndarray
    am: np.ndarray
    FM: np.ndarray
    AM: np.ndarray
    IMF: np.ndarray
    IMF2: np.ndarray


def _mirror_envelope(data: np.ndarray) -> np.ndarray:
    """Combined absolute extrema envelope used in multi_EMD_DCM_SV."""
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, None]
    result = np.empty_like(values)
    grid = np.arange(values.shape[0])
    for col in range(values.shape[1]):
        x = values[:, col]
        imin, imax, _ = extrema(x)
        ids = np.unique(np.r_[0, imin, imax, x.size - 1])
        if imin.size + imax.size < 3:
            result[:, col] = np.abs(x)
        else:
            result[:, col] = CubicSpline(ids, np.abs(x[ids]), bc_type="not-a-knot")(grid)
    return result


def _valid_modes(modes: np.ndarray, scale: float) -> int:
    last = -1
    for idx in range(modes.shape[1]):
        imin, imax, izero = extrema(modes[:, idx])
        if imin.size and imax.size and modes[imax, idx].sum() > 1e-10 * scale:
            if imin.size + imax.size + izero.size >= 5:
                last = idx
    return last + 1


def decompose(signal: np.ndarray, sample_rate: float, *, max_imfs: int = -1,
              max_modulation_imfs: int = -1, mask_order: int = 0,
              mask_order2: int = 0, amplitude_ratio: float = 2,
              amplitude_ratio2: float = 2, upsample_level: int = 1) -> HHSAResult:
    x = np.asarray(signal, dtype=float).ravel()
    n1 = int(np.floor(np.log2(x.size))) if max_imfs <= 0 else max_imfs
    n2 = int(np.floor(np.log2(x.size))) if max_modulation_imfs <= 0 else max_modulation_imfs
    first_raw = masking_emd(x, -1 if max_imfs <= 0 else n1, mask_order,
                            upsample_level, amplitude_ratio, 1.0)
    count = _valid_modes(first_raw, np.std(x, ddof=0))
    first = np.column_stack([first_raw[:, :count], first_raw[:, count:].sum(axis=1)]) if count < first_raw.shape[1] else first_raw
    fm = np.zeros_like(first)
    am = np.zeros_like(first)
    if count:
        fm[:, :count], _, _ = direct_quadrature(first[:, :count], sample_rate)
        am[:, :count] = _mirror_envelope(first[:, :count])
    if count < first.shape[1]:
        am[:, count] = np.abs(first[:, count])
    second_parts, valid_counts = [], []
    for carrier in range(count):
        part = masking_emd(am[:, carrier], -1 if max_modulation_imfs <= 0 else n2,
                           mask_order2, 0, amplitude_ratio2, -1.0)
        second_parts.append(part)
        valid_counts.append(_valid_modes(part, np.std(x, ddof=0)))
    width = max((min(p.shape[1], c + 1) for p, c in zip(second_parts, valid_counts)), default=0)
    IMF2 = np.zeros((x.size, width, first.shape[1]))
    FM, AM = np.zeros_like(IMF2), np.zeros_like(IMF2)
    for carrier, (part, valid) in enumerate(zip(second_parts, valid_counts)):
        keep = min(part.shape[1], valid + 1)
        IMF2[:, :valid, carrier] = part[:, :valid]
        if keep > valid:
            IMF2[:, valid, carrier] = part[:, valid:].sum(axis=1)
        if valid:
            FM[:, :valid, carrier], _, _ = direct_quadrature(IMF2[:, :valid, carrier], sample_rate)
            AM[:, :valid, carrier] = _mirror_envelope(IMF2[:, :valid, carrier])
        if keep > valid:
            AM[:, valid, carrier] = np.abs(IMF2[:, valid, carrier])
    return HHSAResult(fm, am, FM, AM, first, IMF2)
