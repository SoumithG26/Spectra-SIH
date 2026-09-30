"""Spectral survey: PSD (Welch) and waterfall (STFT spectrogram)."""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram as _spectrogram
from scipy.signal import welch

from ..core import Signal

_EPS = 1e-20


def welch_psd(signal: Signal, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided power spectral density in dB, centered at 0 Hz (baseband).

    Returns (freqs_hz, psd_db). ``freqs_hz`` are baseband offsets; add
    ``signal.center_freq`` for absolute RF.
    """
    nperseg = int(min(nperseg, len(signal)))
    if nperseg < 8:
        nperseg = min(8, len(signal))
    f, pxx = welch(
        signal.samples,
        fs=signal.sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        detrend=False,
    )
    order = np.argsort(f)
    f = f[order]
    pxx = pxx[order]
    return f, 10.0 * np.log10(pxx + _EPS)


def waterfall(
    signal: Signal, nperseg: int = 256, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-frequency spectrogram (the waterfall view).

    Returns (freqs_hz, times_s, power_db) with power_db shaped
    (n_freqs, n_times) and frequency ascending / centered at 0 Hz.
    """
    nperseg = int(min(nperseg, len(signal)))
    if nperseg < 8:
        nperseg = min(8, len(signal))
    noverlap = int(nperseg * overlap)
    f, t, sxx = _spectrogram(
        signal.samples,
        fs=signal.sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        mode="psd",
        detrend=False,
    )
    order = np.argsort(f)
    f = f[order]
    sxx = sxx[order, :]
    return f, t, 10.0 * np.log10(sxx + _EPS)
