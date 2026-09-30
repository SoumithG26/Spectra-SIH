"""Modem primitives shared by the synthetic generator and the demodulator.

Keeping the constellations, Gray maps and pulse-shaping filter in one place
guarantees that what the generator transmits is exactly what the demodulator
expects, so round-trip tests are meaningful.

Supported linear modulations: BPSK, QPSK, 8PSK, 16QAM, 64QAM. FSK is frequency-
domain and handled in :mod:`spectra.dsp.demod` / the generator.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# name -> (family, M)
SUPPORTED_LINEAR: dict[str, tuple[str, int]] = {
    "BPSK": ("psk", 2),
    "QPSK": ("psk", 4),
    "8PSK": ("psk", 8),
    "16QAM": ("qam", 16),
    "64QAM": ("qam", 64),
}


def bits_per_symbol(mod: str) -> int:
    fam, M = SUPPORTED_LINEAR[mod.upper()]
    return int(np.log2(M))


def _gray(v: int) -> int:
    return v ^ (v >> 1)


def _inv_gray(g: int) -> int:
    b = g
    shift = 1
    while (g >> shift) > 0:
        b ^= g >> shift
        shift += 1
    return b


def _psk_point(v: int, M: int) -> complex:
    g = _gray(v)
    return complex(np.exp(1j * 2 * np.pi * g / M))


def _qam_point(v: int, M: int) -> complex:
    L = int(round(np.sqrt(M)))
    k = int(np.log2(L))
    i_val = v >> k
    q_val = v & (L - 1)
    pi_ = _inv_gray(i_val)
    pq_ = _inv_gray(q_val)
    i = 2 * pi_ - (L - 1)
    q = 2 * pq_ - (L - 1)
    return complex(i + 1j * q)


@lru_cache(maxsize=None)
def constellation(mod: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (points, bit_table, norm) for a linear modulation.

    ``points`` are normalized to unit average power. ``bit_table[i]`` is the
    MSB-first bit tuple carried by ``points[i]``. ``norm`` is the scale factor
    applied (useful for un-normalizing).
    """
    mod = mod.upper()
    fam, M = SUPPORTED_LINEAR[mod]
    bps = int(np.log2(M))
    raw = np.empty(M, dtype=np.complex128)
    bits = np.empty((M, bps), dtype=np.uint8)
    for v in range(M):
        raw[v] = _psk_point(v, M) if fam == "psk" else _qam_point(v, M)
        for i in range(bps):
            bits[v, i] = (v >> (bps - 1 - i)) & 1
    norm = float(np.sqrt(np.mean(np.abs(raw) ** 2)))
    points = (raw / norm).astype(np.complex64)
    return points, bits, norm


def map_symbols(bit_stream: np.ndarray, mod: str) -> np.ndarray:
    """Map a bit stream to complex symbols (zero-padded to a whole symbol)."""
    points, _, _ = constellation(mod)
    bps = bits_per_symbol(mod)
    b = np.asarray(bit_stream, dtype=np.uint8).ravel()
    pad = (-len(b)) % bps
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    groups = b.reshape(-1, bps)
    weights = (1 << np.arange(bps - 1, -1, -1)).astype(np.int64)
    idx = groups.astype(np.int64) @ weights
    return points[idx]


def demap_symbols(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Nearest-point (hard) demapping of complex symbols back to bits."""
    points, bit_table, _ = constellation(mod)
    syms = np.asarray(symbols, dtype=np.complex64).ravel()
    # nearest constellation point per symbol
    d = np.abs(syms[:, None] - points[None, :])
    idx = np.argmin(d, axis=1)
    return bit_table[idx].reshape(-1).astype(np.uint8)


def rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine pulse. ``span`` = filter length in symbols."""
    N = span * sps
    n = np.arange(-(N // 2), N // 2 + 1)
    t = n / sps  # time in symbol periods
    h = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float64)
