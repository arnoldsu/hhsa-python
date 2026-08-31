# Niño 3.4 HHSA reproducibility check

This document explains how to reproduce and test the time-resolved
carrier–amplitude-modulation analysis.

## Files used

```text
data/nino34_noaa_raw.data
data/nino34_monthly.csv
data/nino34_monthly.npz
scripts/prepare_nino34.py
examples/nino34_example.py
examples/nino34_peak_timeseries.py
src/hhsa/
```

The detailed scientific results and limitations are in
[`nino34_details.md`](nino34_details.md).

## 1. Load the Gadi environment

```bash
module purge
module load pbs
module use /g/data/xp65/public/modules
module load conda/analysis3-26.01

cd /g/data/p66/ars599/HHSA_WK/hhsa-python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. Optionally refresh NOAA data

```bash
python scripts/prepare_nino34.py --download
```

Expected message (the final month will change when NOAA updates the file):

```text
Converted 943 monthly values (1948-01-01 to 2026-07-01)
Saved data/nino34_monthly.csv and data/nino34_monthly.npz
```

The converter removes NOAA missing values (`-99.99`) and stores:

- ISO date strings;
- monthly Niño 3.4 SST anomaly in °C;
- sample rate of 12 samples/year;
- source URL.

## 3. Run package tests

```bash
python -m pytest -q
```

Current expected result:

```text
3 passed
```

## 4. Reproduce the integrated Niño 3.4 spectrum

```bash
python examples/nino34_example.py \
    --max-imfs 8 \
    --max-modulation-imfs 6
```

Expected output:

```text
outputs/nino34_hhsa.png
```

The HHSA configuration is:

```python
result = decompose(
    signal,
    sample_rate=12.0,
    max_imfs=8,
    max_modulation_imfs=6,
    upsample_level=1,
)

spectrum = project(
    result,
    start=0,
    stop=signal.size,
    time_bins=120,
    bins_per_octave=8,
    carrier_range=(-4.0, 1.0),
    modulation_range=(-6.0, 0.0),
)
```

Because frequencies are stored on log2 axes, these ranges correspond to:

```text
carrier: 2^-4 to 2^1 = 0.0625 to 2 cycles/year
AM:      2^-6 to 2^0 = 0.015625 to 1 cycle/year
```

## 5. Reproduce carrier–AM peak time series

```bash
python examples/nino34_peak_timeseries.py
```

Expected files:

```text
outputs/nino34_peak_timeseries.csv
outputs/nino34_peak_timeseries.png
```

Expected resolved spectral-bin periods:

```text
c2.18_am9.51:   carrier=2.181 yr, AM=9.514 yr
c2.18_am13.45:  carrier=2.181 yr, AM=13.454 yr
c2.18_am19.03:  carrier=2.181 yr, AM=19.027 yr
c2.18_am29.34:  carrier=2.181 yr, AM=29.344 yr
c4.00_am9.51:   carrier=4.000 yr, AM=9.514 yr
c4.36_am11.31:  carrier=4.362 yr, AM=11.314 yr
c4.36_am17.45:  carrier=4.362 yr, AM=17.448 yr
c4.36_am24.68:  carrier=4.362 yr, AM=24.675 yr
c5.66_am17.45:  carrier=5.657 yr, AM=17.448 yr
c6.73_am14.67:  carrier=6.727 yr, AM=14.672 yr
```

## 6. Extraction method

The full projected spectrum has dimensions:

```text
All_nt[carrier_frequency, AM_frequency, time, first_layer_IMF]
```

For each selected carrier–AM peak, the script:

1. Converts target periods to frequencies using `frequency = 1 / period`.
2. Finds the nearest carrier and AM frequency bins.
3. Selects a ±1-bin neighbourhood in both frequency dimensions.
4. Excludes spectrum column zero, which represents DC modulation.
5. Sums energy over the local frequency neighbourhood and all first-layer IMFs.
6. Retains the 120-bin time dimension.

The core extraction is:

```python
def extract_neighbourhood(power, carrier_index, am_index, radius=1):
    c0 = max(0, carrier_index - radius)
    c1 = min(power.shape[0], carrier_index + radius + 1)

    # Column zero is DC. Positive AM bin j is stored in column j + 1.
    centre = am_index + 1
    m0 = max(1, centre - radius)
    m1 = min(power.shape[1], centre + radius + 1)

    return power[c0:c1, m0:m1, :, :].sum(axis=(0, 1, 3))
```

With 943 monthly values and 120 output bins, each time point represents
approximately:

```text
943 / 120 = 7.86 months
```

The date assigned to each output bin is the middle monthly observation mapped
to that bin.

## 7. CSV columns

The first column is:

```text
date
```

Every carrier–AM combination has two columns:

```text
<label>_energy    raw unsmoothed HHSA energy
<label>_relative  energy divided by that combination's maximum
```

Examples:

```text
c2.18_am9.51_energy
c2.18_am9.51_relative
c4.36_am17.45_energy
c4.36_am17.45_relative
```

Use `*_energy` to compare absolute strength between combinations. Use
`*_relative` to examine when one specific combination becomes stronger or
weaker.

## 8. Plot processing

For readability, the plotted relative time series receive only a one-bin
Gaussian smoothing:

```python
relative = energy / energy.max()
smooth = gaussian_filter1d(relative, sigma=1.0)
```

The CSV energy values are not smoothed. Each line is normalised independently,
so plot height compares timing within a line, not absolute energy between lines.

## 9. Basic output checks

Check that files exist:

```bash
ls -lh \
    outputs/nino34_peak_timeseries.csv \
    outputs/nino34_peak_timeseries.png
```

Check row and column counts:

```bash
python - <<'PY'
import pandas as pd

data = pd.read_csv("outputs/nino34_peak_timeseries.csv")
print(data.shape)
print(data.columns.tolist())
print(data.head())
print(data.tail())
PY
```

Expected shape:

```text
(120, 21)
```

This consists of one date column plus raw and relative columns for 10 peaks.

Check that all energies are finite and non-negative:

```bash
python - <<'PY'
import numpy as np
import pandas as pd

data = pd.read_csv("outputs/nino34_peak_timeseries.csv")
energy = data.filter(regex="_energy$").to_numpy()
relative = data.filter(regex="_relative$").to_numpy()

assert np.isfinite(energy).all()
assert np.isfinite(relative).all()
assert (energy >= 0).all()
assert (relative >= 0).all()
assert (relative <= 1 + 1e-9).all()
print("Niño 3.4 time-series checks passed")
PY
```

## 10. Change time or frequency resolution

Use more time bins for finer timing:

```bash
python examples/nino34_peak_timeseries.py --time-bins 240
```

This gives approximately 3.9 months per output bin, but individual bins will
contain less energy and may look noisier.

Change the local frequency neighbourhood:

```bash
python examples/nino34_peak_timeseries.py --radius 0  # exact bin only
python examples/nino34_peak_timeseries.py --radius 2  # broader frequency band
```

The default `--radius 1` is a compromise between a single quantised bin and an
overly broad band.

## 11. Interpretation limits

- These are time-resolved HHSA energy series, not reconstructed SST signals.
- A high value means that a carrier–AM combination is active at that time.
- The lines are not statistical significance probabilities.
- Significance should be evaluated using red-noise and/or phase-randomised
  surrogate series.
- AM periods near 29–32 years contain only about 2–3 cycles in this record and
  should be interpreted cautiously.
- Edge effects can affect the beginning and end of the record.
- The Python `emdx.m` port replaces the unavailable proprietary
  `rcada_emd` MEX source, so bit-for-bit MATLAB MEX equivalence is not claimed.
