"""Hilbert–Huang spectral analysis."""

from .core import HHSAResult, decompose
from .spectrum import Spectrum, project

__all__ = ["HHSAResult", "Spectrum", "decompose", "project"]
__version__ = "0.2.0"
