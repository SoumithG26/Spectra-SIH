"""Front-end parameter estimation: bandwidth, SNR, CFO, symbol rate.

These operate on the signal itself (not on metadata) and each returns a value
plus a rough confidence, matching the workbench principle that every automated
estimate is scored, never asserted as ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..core import Signal

_EPS = 1e-20


def occupied_bandwidth(signal: Signal, fraction: float = 0.99) -> dict:
    """Estimate the band carrying ``fraction`` of the total power.

    Returns dict with ``bandwidth`` (Hz), ``center`` (Hz offset) and the low/high
    band edges.
    """
    nperseg = int(min(4096, len(signal)))
    f, pxx = welch(signal.samples, fs=signal.sample_rate, nperseg=nperseg,
                   return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    total = pxx.sum()
    if total <= 0:
        return {"bandwidth": 0.0, "center": 0.0, "low": 0.0, "high": 0.0}
    csum = np.cumsum(pxx)
    lo_thresh = (1 - fraction) / 2 * total
    hi_thresh = (1 + fraction) / 2 * total
    lo = int(np.searchsorted(csum, lo_thresh))
    hi = int(np.searchsorted(csum, hi_thresh))
    lo = min(lo, len(f) - 1)
    hi = min(hi, len(f) - 1)
    return {
        "bandwidth": float(f[hi] - f[lo]),
        "center": float((f[hi] + f[lo]) / 2),
        "low": float(f[lo]),
        "high": float(f[hi]),
    }


def estimate_snr_db(signal: Signal) -> dict:
    """Rough in-band SNR from the PSD (noise floor = lower quartile of bins)."""
    nperseg = int(min(2048, len(signal)))
    f, pxx = welch(signal.samples, fs=signal.sample_rate, nperseg=nperseg,
                   return_onesided=False, detrend=False)
    noise = float(np.percentile(pxx, 25))
    occupied = pxx > noise * 3.0
    if not occupied.any():
        return {"snr_db": 0.0, "noise_floor": noise, "confidence": 0.2}
    sig = float(np.mean(pxx[occupied]) - noise)
    snr = 10.0 * np.log10(max(sig, _EPS) / max(noise, _EPS))
    return {"snr_db": float(snr), "noise_floor": noise, "confidence": 0.6}


def estimate_cfo(signal: Signal, orders: tuple[int, ...] = (2, 4, 8)) -> dict:
    """Estimate carrier-frequency offset via the M-th-power spectral line.

    For an M-PSK signal, raising x(t) to the M-th power strips the data
    modulation and leaves a tone at ``M * cfo``. We try several orders and keep
    the one that produces the strongest tone.
    """
    x = signal.normalized().samples.astype(np.complex128)
    N = len(x)
    if N < 16:
        return {"cfo_hz": 0.0, "order": 0, "confidence": 0.0}
    fs = signal.sample_rate
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))
    best = {"cfo_hz": 0.0, "order": 0, "confidence": 0.0}
    for M in orders:
        spec = np.abs(np.fft.fftshift(np.fft.fft(x ** M)))
        idx = int(np.argmax(spec))
        strength = float(spec[idx] / (np.mean(spec) + _EPS))
        cfo = float(freqs[idx] / M)
        conf = float(np.clip((strength - 1.0) / 50.0, 0.0, 1.0))
        if conf > best["confidence"]:
            best = {"cfo_hz": cfo, "order": M, "confidence": conf,
                    "tone_strength": strength}
    return best


def estimate_symbol_rate(signal: Signal, min_sps: float = 2.0) -> dict:
    """Estimate the symbol rate from the spectral line in |x|^2.

    Linearly-modulated signals with excess bandwidth (RRC roll-off > 0) exhibit
    a cyclostationary line at the symbol rate in the spectrum of the squared
    envelope. We search for the dominant such peak below ``fs / min_sps``.
    """
    x = signal.normalized().samples.astype(np.complex128)
    fs = signal.sample_rate
    N = len(x)
    if N < 64:
        return {"symbol_rate": 0.0, "sps": 0.0, "confidence": 0.0}
    y = np.abs(x) ** 2
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    # search window: avoid DC leakage, cap at fs/min_sps
    lo = max(4, int(N * 0.001))
    hi_freq = fs / min_sps
    hi = int(np.searchsorted(freqs, hi_freq))
    hi = max(lo + 1, min(hi, len(freqs)))
    band = spec[lo:hi]
    if band.size == 0:
        return {"symbol_rate": 0.0, "sps": 0.0, "confidence": 0.0}
    k = int(np.argmax(band)) + lo
    rs = float(freqs[k])
    conf = float(np.clip((spec[k] / (np.median(band) + _EPS) - 1.0) / 20.0, 0.0, 1.0))
    return {
        "symbol_rate": rs,
        "sps": float(fs / rs) if rs > 0 else 0.0,
        "confidence": conf,
    }
