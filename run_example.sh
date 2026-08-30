#!/bin/bash
# Run every verified HHSA Python example on Gadi.
set -euo pipefail

PROJECT_DIR="/g/data/p66/ars599/HHSA_WK/hhsa-python"

module purge
module load pbs
module use /g/data/xp65/public/modules
module load conda/analysis3-26.01

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "1/4 Running automated tests"
python -m pytest -q

echo
echo "2/4 Running known synthetic AM validation"
echo "Signal: (1 + 0.7*cos(2*pi*2*t))*cos(2*pi*20*t)"
echo "Expected carrier: 20 Hz; expected AM frequency: 2 Hz; sample rate: 200 Hz"
python examples/simple_am_example.py

echo
echo "3/4 Running the bundled MATLAB-data example"
python -m hhsa.cli --max-imfs 8 --max-modulation-imfs 6

echo
echo "4/4 Running the NOAA Nino 3.4 monthly-index example"
python examples/nino34_example.py --max-imfs 8 --max-modulation-imfs 6

echo
echo "All examples completed successfully."
echo "Outputs:"
echo "  outputs/simple_am_hhsa.png"
echo "  outputs/HHS_ex1_python.png"
echo "  outputs/HHS_ex1_python.mat"
echo "  outputs/nino34_hhsa.png"
