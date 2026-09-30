"""Core data model shared across the SPECTRA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Signal:
    """A normalized complex-baseband recording.

    All loaders converge on this representation so the rest of the pipeline is
    format-agnostic. Real recordings (mono WAV) are converted to an analytic
    signal on load, so ``samples`` is always complex.

    Attributes
    ----------
    samples:
        1-D ``complex64`` array of baseband IQ samples.
    sample_rate:
        Sampling rate in Hz. For WAV this is read from the header; for raw IQ it
        must be supplied by the analyst (it is *not* stored in the bytes).
    center_freq:
        RF center frequency in Hz if known (0.0 = baseband / unknown).
    source:
        Human-readable provenance string (file path, "synthetic", ...).
    meta:
        Free-form metadata (SigMF fields, ground-truth labels for synthetic
        signals, loader notes, ...).
    """

    samples: np.ndarray
    sample_rate: float
    center_freq: float = 0.0
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = np.ascontiguousarray(self.samples, dtype=np.complex64)
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got shape {self.samples.shape}")
        self.sample_rate = float(self.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self) / self.sample_rate

    @property
    def is_real_valued(self) -> bool:
        """True if the imaginary part is (numerically) zero everywhere."""
        return bool(np.allclose(self.samples.imag, 0.0, atol=1e-6))

    def normalized(self) -> "Signal":
        """Return a copy scaled to unit average power (does not mutate self)."""
        power = float(np.mean(np.abs(self.samples) ** 2))
        if power <= 0:
            return Signal(self.samples.copy(), self.sample_rate, self.center_freq,
                          self.source, dict(self.meta))
        scaled = self.samples / np.sqrt(power)
        return Signal(scaled, self.sample_rate, self.center_freq, self.source,
                      dict(self.meta))

    def slice_time(self, t0: float, t1: float) -> "Signal":
        """Return the sub-signal between ``t0`` and ``t1`` seconds."""
        i0 = max(0, int(round(t0 * self.sample_rate)))
        i1 = min(len(self), int(round(t1 * self.sample_rate)))
        return Signal(self.samples[i0:i1], self.sample_rate, self.center_freq,
                      f"{self.source}[{t0:.4f}:{t1:.4f}s]", dict(self.meta))

    def summary(self) -> dict[str, Any]:
        """Compact dict for logging / UI headers."""
        return {
            "source": self.source,
            "samples": len(self),
            "sample_rate_hz": self.sample_rate,
            "duration_s": round(self.duration, 6),
            "center_freq_hz": self.center_freq,
            "real_valued": self.is_real_valued,
        }
