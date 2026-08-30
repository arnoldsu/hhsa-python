#!/bin/bash

module purge
module load pbs
module use /g/data/xp65/public/modules
module load conda/analysis3-26.01

cd /g/data/p66/ars599/HHSA_WK/hhsa-python

PYTHONPATH=src python -m hhsa.cli \
    --max-imfs 8 \
    --max-modulation-imfs 6

