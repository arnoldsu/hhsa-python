#!/usr/bin/env python3
"""Extract time-resolved HHSA energy for selected Niño 3.4 carrier–AM peaks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from hhsa import decompose, project


# (short label, carrier period in years, AM period in years)
PEAKS = (
    ("c2.18_am9.51", 2.181, 9.514),
    ("c2.18_am13.45", 2.181, 13.454),
    ("c2.18_am19.03", 2.181, 19.027),
    ("c2.18_am29.34", 2.181, 29.344),
    ("c4.00_am9.51", 4.000, 9.514),
    ("c4.36_am11.31", 4.362, 11.314),
    ("c4.36_am17.45", 4.362, 17.448),
    ("c4.36_am24.68", 4.362, 24.675),
    ("c5.66_am17.45", 5.657, 17.448),
    ("c6.73_am14.67", 6.727, 14.672),
)


def bin_dates(dates: np.ndarray, sample_bins: np.ndarray, count: int) -> np.ndarray:
    """Return the middle observation date assigned to each HHSA time bin."""
    result = []
    for index in range(count):
        members = dates[sample_bins == index]
        result.append(members[len(members) // 2] if members.size else dates[-1])
    return np.asarray(result)


def extract_neighbourhood(power: np.ndarray, ci: int, mi: int, radius: int) -> np.ndarray:
    """Sum a local carrier × AM neighbourhood and all first-layer IMFs."""
    c0, c1 = max(0, ci - radius), min(power.shape[0], ci + radius + 1)
    # Spectrum column zero is DC; positive AM scale index mi starts at column mi+1.
    centre = mi + 1
    m0, m1 = max(1, centre - radius), min(power.shape[1], centre + radius + 1)
    return power[c0:c1, m0:m1, :, :].sum(axis=(0, 1, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/nino34_monthly.npz"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/nino34_peak_timeseries.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nino34_peak_timeseries.png"))
    parser.add_argument("--time-bins", type=int, default=120,
                        help="120 gives about 7.9 months per bin for this record")
    parser.add_argument("--radius", type=int, default=1,
                        help="frequency-bin radius summed around each peak")
    args = parser.parse_args()

    dataset = np.load(args.input)
    dates = dataset["date"].astype("datetime64[D]")
    signal = dataset["nino34_anomaly_c"].astype(float)
    result = decompose(signal, 12.0, max_imfs=8, max_modulation_imfs=6,
                       upsample_level=1)
    spectrum = project(result, start=0, stop=signal.size, time_bins=args.time_bins,
                       bins_per_octave=8, carrier_range=(-4.0, 1.0),
                       modulation_range=(-6.0, 0.0))
    carrier_frequency = 2.0 ** spectrum.carrier_scale
    modulation_frequency = 2.0 ** spectrum.modulation_scale
    sample_bins = np.ceil(
        np.arange(1, signal.size + 1) * args.time_bins / signal.size
    ).astype(int) - 1
    times = bin_dates(dates, sample_bins, args.time_bins)

    series: dict[str, np.ndarray] = {}
    metadata = []
    for label, carrier_period, am_period in PEAKS:
        ci = int(np.argmin(np.abs(carrier_frequency - 1.0 / carrier_period)))
        mi = int(np.argmin(np.abs(modulation_frequency - 1.0 / am_period)))
        energy = extract_neighbourhood(spectrum.power, ci, mi, args.radius)
        series[label] = energy
        metadata.append((label, 1/carrier_frequency[ci], 1/modulation_frequency[mi]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        header = ["date"]
        for label in series:
            header.extend((f"{label}_energy", f"{label}_relative"))
        writer.writerow(header)
        for row in range(args.time_bins):
            values = [str(times[row])]
            for energy in series.values():
                maximum = energy.max()
                values.extend((f"{energy[row]:.12g}",
                               f"{energy[row]/maximum if maximum else 0:.9g}"))
            writer.writerow(values)

    groups = (
        ("2.18-year carrier", tuple(k for k in series if k.startswith("c2.18"))),
        ("4.0–4.36-year carrier", tuple(k for k in series if k.startswith(("c4.00", "c4.36")))),
        ("5.66-year carrier", ("c5.66_am17.45",)),
        ("6.73-year carrier", ("c6.73_am14.67",)),
    )
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for ax, (title, labels) in zip(axes, groups):
        for label in labels:
            energy = series[label]
            relative = energy / energy.max() if energy.max() else energy
            smooth = gaussian_filter1d(relative, sigma=1.0)
            am_period = next(p[2] for p in PEAKS if p[0] == label)
            ax.plot(times, smooth, linewidth=1.3, label=f"AM {am_period:.2f} yr")
        ax.set(title=title, ylabel="Relative HHSA energy", ylim=(0, 1.05))
        ax.grid(alpha=0.25)
        ax.legend(ncol=min(4, len(labels)), fontsize=9, loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Niño 3.4 carrier–AM energy through time", fontsize=16)
    fig.savefig(args.output, dpi=180)

    print(f"Saved {args.csv}")
    print(f"Saved {args.output}")
    print("Resolved spectral-bin periods:")
    for label, carrier_period, am_period in metadata:
        print(f"  {label}: carrier={carrier_period:.3f} yr, AM={am_period:.3f} yr")


if __name__ == "__main__":
    main()
