#!/usr/bin/env python3
"""HHSA example using the NOAA monthly Nino 3.4 SST anomaly index."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from hhsa import decompose, project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/nino34_monthly.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nino34_hhsa.png"))
    parser.add_argument("--max-imfs", type=int, default=8)
    parser.add_argument("--max-modulation-imfs", type=int, default=6)
    args = parser.parse_args()

    dataset = np.load(args.input)
    dates = dataset["date"].astype("datetime64[D]")
    signal = dataset["nino34_anomaly_c"].astype(float)
    sample_rate = float(dataset["sample_rate_per_year"])

    result = decompose(signal, sample_rate, max_imfs=args.max_imfs,
                       max_modulation_imfs=args.max_modulation_imfs,
                       upsample_level=1)
    spectrum = project(result, start=0, stop=signal.size, time_bins=120,
                       bins_per_octave=8, carrier_range=(-4.0, 1.0),
                       modulation_range=(-6.0, 0.0))
    power = gaussian_filter(spectrum.power.sum(axis=(2, 3)).T, sigma=0.6)
    power_db = 10*np.log10(np.maximum(power/power.max(), 1e-6))
    power_db = np.ma.masked_less(power_db, -45)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(dates, signal, color="black", linewidth=0.8)
    axes[0].axhline(0, color="0.6", linewidth=0.7)
    axes[0].set(title="NOAA ERSST v6 Nino 3.4 monthly SST anomaly",
                ylabel="SST anomaly (deg C)")
    image = axes[1].imshow(power_db, origin="lower", aspect="auto",
                           extent=(-4, 1, -6, 0), cmap="turbo", vmin=-45, vmax=0,
                           interpolation="bilinear")
    axes[1].plot([-4, 0], [-4, 0], "w:", linewidth=1.3)
    xticks = np.arange(-4, 2)
    yticks = np.arange(-6, 1)
    axes[1].set_xticks(xticks, [f"{2.0**v:g}" for v in xticks])
    axes[1].set_yticks(yticks, [f"{2.0**v:g}" for v in yticks])
    axes[1].set(xlabel="Carrier frequency (cycles/year)",
                ylabel="AM frequency (cycles/year)")
    axes[1].set_facecolor("#eeeeee")
    fig.colorbar(image, ax=axes[1], label="Relative energy (dB)")
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
