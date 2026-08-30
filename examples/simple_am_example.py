#!/usr/bin/env python3
"""Validate HHSA with a known synthetic amplitude-modulated signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from hhsa import decompose, project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/simple_am_hhsa.png"))
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    sample_rate = 200.0
    carrier_hz = 20.0
    modulation_hz = 2.0
    modulation_depth = 0.7
    time = np.arange(int(args.duration * sample_rate)) / sample_rate
    signal = (1.0 + modulation_depth * np.cos(2*np.pi*modulation_hz*time)) * np.cos(
        2*np.pi*carrier_hz*time
    )

    result = decompose(signal, sample_rate, max_imfs=7,
                       max_modulation_imfs=5, upsample_level=1)
    spectrum = project(result, start=0, stop=signal.size, time_bins=80,
                       bins_per_octave=16, carrier_range=(0.0, 6.0),
                       modulation_range=(-2.0, 4.0))
    power = spectrum.power.sum(axis=(2, 3))
    # Column zero is DC amplitude; columns 1: map to modulation_scale.
    regular = power[:, 1:]
    peak = np.unravel_index(np.argmax(regular), regular.shape)
    detected_carrier = 2.0 ** spectrum.carrier_scale[peak[0]]
    detected_modulation = 2.0 ** spectrum.modulation_scale[peak[1]]
    carrier_error = abs(detected_carrier - carrier_hz)
    modulation_error = abs(detected_modulation - modulation_hz)

    print(f"Expected carrier:   {carrier_hz:.3f} Hz")
    print(f"Detected carrier:   {detected_carrier:.3f} Hz")
    print(f"Expected modulation:{modulation_hz:8.3f} Hz")
    print(f"Detected modulation:{detected_modulation:8.3f} Hz")
    print(f"Absolute errors: carrier={carrier_error:.3f} Hz, AM={modulation_error:.3f} Hz")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    shown = time < 2
    axes[0].plot(time[shown], signal[shown], color="black", linewidth=0.9)
    axes[0].plot(time[shown], 1 + modulation_depth*np.cos(2*np.pi*modulation_hz*time[shown]),
                 "r--", linewidth=1, label="known envelope")
    axes[0].plot(time[shown], -(1 + modulation_depth*np.cos(2*np.pi*modulation_hz*time[shown])),
                 "r--", linewidth=1)
    axes[0].set(title="Known AM signal: carrier 20 Hz, modulation 2 Hz",
                xlabel="Time (s)", ylabel="Amplitude")
    axes[0].legend()

    display = gaussian_filter(regular.T, sigma=1.0)
    display_db = 10*np.log10(np.maximum(display/display.max(), 1e-6))
    display_db = np.ma.masked_less(display_db, -45)
    image = axes[1].imshow(display_db, origin="lower", aspect="auto",
                           extent=(0, 6, -2, 4), cmap="turbo", vmin=-45, vmax=0,
                           interpolation="bilinear")
    axes[1].scatter(np.log2(carrier_hz), np.log2(modulation_hz),
                    marker="x", s=90, color="red", linewidth=2, label="expected 20/2 Hz")
    axes[1].scatter(np.log2(detected_carrier), np.log2(detected_modulation),
                    facecolors="none", edgecolors="white", s=90, label="detected peak")
    xticks = np.arange(0, 7)
    yticks = np.arange(-2, 5)
    axes[1].set_xticks(xticks, [f"{2.0**v:g}" for v in xticks])
    axes[1].set_yticks(yticks, [f"{2.0**v:g}" for v in yticks])
    axes[1].set(xlabel="Carrier frequency (Hz)", ylabel="AM frequency (Hz)")
    axes[1].legend()
    axes[1].set_facecolor("#eeeeee")
    fig.colorbar(image, ax=axes[1], label="Relative energy (dB)")
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")

    if carrier_error > 2.0 or modulation_error > 0.5:
        raise SystemExit("Validation failed: detected peak is outside tolerance")


if __name__ == "__main__":
    main()
