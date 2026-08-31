# Niño 3.4 HHSA analysis details

## Purpose

This analysis asks which slower amplitude-modulation (AM) periods carry or
modulate the familiar 2–7 year Niño 3.4 oscillations.

In this document, a result such as **2.18 years × 9.51 years** means that a
Niño 3.4 carrier oscillation with an approximately 2.18-year period has an
amplitude envelope varying on an approximately 9.51-year period.

## Data

- Source: NOAA Physical Sciences Laboratory
- Dataset: monthly Niño 3.4 ERSST v6 SST anomaly index
- Region: 5°N–5°S, 170°W–120°W
- Units: °C
- Sampling rate: 12 samples/year
- Valid values used: 943 months
- Coverage: January 1948 through July 2026
- Official source: <https://psl.noaa.gov/data/timeseries/month/Nino34_CPC/>
- Local files:
  - `data/nino34_noaa_raw.data`
  - `data/nino34_monthly.csv`
  - `data/nino34_monthly.npz`

Refresh and convert the official data with:

```bash
python scripts/prepare_nino34.py --download
```

## HHSA configuration

The numerical results were obtained by running the Python HHSA implementation,
not by estimating locations from the PNG colours.

```python
result = decompose(
    nino34,
    sample_rate=12,
    max_imfs=8,
    max_modulation_imfs=6,
    upsample_level=1,
)

spectrum = project(
    result,
    start=0,
    stop=nino34.size,
    time_bins=120,
    bins_per_octave=8,
    carrier_range=(-4.0, 1.0),
    modulation_range=(-6.0, 0.0),
)
```

The displayed and analysed two-dimensional spectrum was produced by summing
HHSA energy over time and first-layer IMF number. The DC modulation column was
excluded from peak detection.

## Analysis region

The selected physical region was:

| Quantity | Period range | Frequency range |
|---|---:|---:|
| Carrier | 2–7 years | 0.1429–0.5 cycles/year |
| AM envelope | 8–32 years | 0.03125–0.125 cycles/year |

Frequency and period are related by:

```text
period in years = 1 / frequency in cycles per year
```

Local maxima were detected in the smoothed two-dimensional energy array. Peaks
closer than two spectral bins in both dimensions were treated as repeated
samples of the same broad ridge. Peaks were ranked relative to the strongest
peak within the selected region.

## Detected carrier–AM peaks

| Rank | Carrier frequency | Carrier period | AM frequency | AM period | Relative energy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.458502 /yr | 2.181 years | 0.105112 /yr | 9.514 years | 0.00 dB |
| 2 | 0.229251 /yr | 4.362 years | 0.057313 /yr | 17.448 years | −4.35 dB |
| 3 | 0.250000 /yr | 4.000 years | 0.105112 /yr | 9.514 years | −4.39 dB |
| 4 | 0.229251 /yr | 4.362 years | 0.040526 /yr | 24.675 years | −4.71 dB |
| 5 | 0.229251 /yr | 4.362 years | 0.088388 /yr | 11.314 years | −4.76 dB |
| 6 | 0.458502 /yr | 2.181 years | 0.052556 /yr | 19.027 years | −7.60 dB |
| 7 | 0.176777 /yr | 5.657 years | 0.057313 /yr | 17.448 years | −8.53 dB |
| 8 | 0.458502 /yr | 2.181 years | 0.074325 /yr | 13.454 years | −10.99 dB |
| 9 | 0.148651 /yr | 6.727 years | 0.068157 /yr | 14.672 years | −11.39 dB |
| 10 | 0.458502 /yr | 2.181 years | 0.034078 /yr | 29.344 years | −18.21 dB |

The selected 2–7 year carrier by 8–32 year AM region contains approximately
**58.83%** of the plotted oscillatory energy after excluding the DC modulation
column.

## Main structures

### Approximately 2.18-year carrier

The strongest modulation is approximately 9.51 years. Additional local AM
peaks occur near 13.45, 19.03, and 29.34 years.

### Approximately 4.0–4.36-year carrier

This is the richest modulation group. Its local AM peaks occur near 9.51,
11.31, 17.45, and 24.68 years.

### Approximately 5.66-year carrier

Its main detected AM peak is approximately 17.45 years.

### Approximately 6.73-year carrier

Its main detected AM peak is approximately 14.67 years.

The spectrum therefore suggests at least two broad ENSO structures:

1. A faster approximately 2.2-year mode, most strongly modulated near 9.5 years.
2. An approximately 4.0–4.4-year mode with several modulation scales spanning
   roughly 9–25 years.

## Plot interpretation

The figure is available at `outputs/nino34_hhsa.png`.

- The upper panel is the monthly Niño 3.4 SST anomaly time series.
- The lower panel is the time-integrated carrier-frequency × AM-frequency
  HHSA spectrum.
- The white diagonal is the boundary `AM frequency = carrier frequency`.
- The grey region above the diagonal is excluded because the projection
  requires carrier frequency to be greater than AM frequency.
- White patches within the plotted region are values masked below −45 dB; they
  are not missing Niño 3.4 observations.
- Colours show energy relative to the strongest displayed peak, from −45 dB
  to 0 dB.

## Important limitations

1. **Spectral-bin precision:** eight bins per octave quantise peak locations.
   For example, 4.362 years should be interpreted as the centre of an
   approximately 4.2–4.5-year band, not an exact permanent physical period.
2. **Time integration:** the current two-dimensional spectrum sums over the
   entire 1948–2026 record, so it does not show when a peak was active.
3. **Statistical significance:** local HHSA energy maxima are not automatically
   statistically significant climate oscillations. They should be tested
   against red-noise and/or phase-randomised surrogate series.
4. **Long-period resolution:** the record is about 78.6 years long. AM periods
   near 29–32 years contain only about 2–3 cycles and are therefore less robust.
5. **Boundary and mode-mixing effects:** results should be checked for EMD edge
   effects and sensitivity to IMF/masking settings.
6. **MEX equivalence:** the original MATLAB distribution provides
   `rcada_emd` only as compiled platform-specific MEX files. The Python package
   uses the supplied source-level `emdx.m` algorithm at that boundary and must
   not be described as bit-for-bit equivalent to the proprietary MEX routine.

## Recommended next analyses

1. Extract carrier–AM energy as a function of time for each peak listed above.
2. Determine which decades dominate the 2.18 × 9.51-year and
   4.36 × 17.45-year combinations.
3. Test whether the carrier–AM peaks exceed red-noise or phase-randomised
   surrogate thresholds at 90%, 95%, and 99% levels.
4. Repeat the calculation with different `bins_per_octave`, IMF limits, and
   masking settings to quantify sensitivity.
5. Compare epochs before and after the 1976/77 Pacific climate shift.

## Reproduce the example

Run only the Niño 3.4 example:

```bash
cd /g/data/p66/ars599/HHSA_WK/hhsa-python

module purge
module load pbs
module use /g/data/xp65/public/modules
module load conda/analysis3-26.01

PYTHONPATH=src python examples/nino34_example.py \
    --max-imfs 8 \
    --max-modulation-imfs 6
```

Run every verified project example:

```bash
./run_example.sh
```
