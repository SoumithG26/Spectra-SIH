"""Synthetic signal factory - the labeled 'digital twin' data source.

Generates baseband IQ with full ground-truth labels (modulation, symbol rate,
FEC, interleaver, SNR, CFO), which is exactly the training/validation data no
public dataset provides for the FEC/interleaver parameters.
"""

from __future__ import annotations

from .generator import SignalSpec, default_suite, generate

__all__ = ["SignalSpec", "generate", "default_suite"]
