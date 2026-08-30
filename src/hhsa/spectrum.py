"""Port of the collapse branch of ``nspplotf3d_tres3x.m``."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import HHSAResult
from .emd import matlab_round


@dataclass(frozen=True)
class Spectrum:
    power: np.ndarray
    carrier_scale: np.ndarray
    modulation_scale: np.ndarray


def project(result: HHSAResult, start: int = 500, stop: int = 2500,
            time_bins: int = 500, bins_per_octave: int = 8,
            carrier_range=(0.0, 7.0), modulation_range=(-1.0, 6.0)) -> Spectrum:
    """Project using Python half-open indices equivalent to MATLAB 501:2500."""
    cf_n = int((carrier_range[1] - carrier_range[0]) * bins_per_octave + 1)
    mf_n = int((modulation_range[1] - modulation_range[0]) * bins_per_octave + 1)
    out = np.zeros((cf_n, mf_n + 1, time_bins, result.FM.shape[2]))
    n = result.fm.shape[0]
    matlab_t = np.ceil(np.arange(1, n + 1) * time_bins / n).astype(int) - 1
    samples = np.arange(start, min(stop, n))
    for carrier_imf in range(result.FM.shape[2]):
        f = result.fm[:, carrier_imf]
        with np.errstate(divide="ignore", invalid="ignore"):
            pf = matlab_round((cf_n - 1) * (np.log2(f) - carrier_range[0]) /
                              (carrier_range[1] - carrier_range[0]))
        for mod_imf in range(result.FM.shape[1]):
            F, A = result.FM[:, mod_imf, carrier_imf], result.AM[:, mod_imf, carrier_imf]
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = (mf_n - 1) * (np.log2(F) - modulation_range[0]) / (modulation_range[1] - modulation_range[0])
            finite = np.isfinite(raw)
            pF = np.zeros_like(pf)
            pF[finite] = matlab_round(raw[finite])
            base = (pf >= 0) & (pf < cf_n) & (f > F)
            regular = base & finite & (pF >= 0) & (pF < mf_n)
            low = base & finite & (pF < 0)
            dc = base & ~finite & (F == 0)
            for mask, col, weight in ((regular, None, .5), (low, 1, .5), (dc, 0, 1.0)):
                use = mask & np.isin(np.arange(n), samples)
                ids = np.flatnonzero(use)
                if not ids.size:
                    continue
                cols = pF[ids] + 1 if col is None else np.full(ids.size, col)
                np.add.at(out[..., carrier_imf], (pf[ids], cols, matlab_t[ids]), weight * A[ids] ** 2)
    return Spectrum(out, np.linspace(*carrier_range, cf_n), np.linspace(*modulation_range, mf_n))
