"""SPECTRA - Signal Parameter Extraction, Classification, Transformation & Recovery Analyzer.

An AI-assisted analyst workbench for off-air signal recordings (.wav / .iq / SigMF).
This package holds the DSP + coding core; the Streamlit GUI (``app.py``) and CLI
(``spectra.cli``) are thin layers on top of it.

Pipeline: INGEST -> SURVEY -> FRONT-END ESTIMATION -> DEMODULATE -> BIT-STRUCTURE -> CORRELATE.
"""

from __future__ import annotations

from .core import Signal

__version__ = "0.1.0"
__all__ = ["Signal", "__version__"]
