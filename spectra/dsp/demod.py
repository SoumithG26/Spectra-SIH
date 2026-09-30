"""Demodulation: FSK / PSK / QAM baseband -> symbols and bits.

The chain is CFO correction -> resample to an integer samples/symbol ->
matched filter (RRC) -> symbol-timing recovery -> carrier-phase recovery ->
slicing. Carrier recovery resolves the residual phase but leaves the usual
M-fold rotation ambiguity of blind PSK/QAM; we resolve it by picking the
rotation with the lowest EVM (an "aided" step the analyst can pin with a known
preamble in the full system).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..core import Signal
from .modem import SUPPORTED_LINEAR, bits_per_symbol, constellation, demap_symbols, rrc
from .params import estimate_cfo


def _target_sps(sps_est: float) -> int:
    if sps_est >= 6:
        return 8
    if sps_est >= 3:
        return 4
    return 2


def recover_symbols(
    signal: Signal,
    symbol_rate: float,
    *,
    beta: float = 0.35,
    span: int = 8,
    correct_cfo: bool = True,
) -> dict:
    """Timing-synchronize a linearly-modulated signal down to 1 sample/symbol.

    Returns dict with ``symbols`` (complex, unit-power), the chosen integer
    ``sps``, the timing ``phase`` and the CFO that was removed.
    """
    x = signal.normalized().samples.astype(np.complex128)
    fs = signal.sample_rate
    cfo = 0.0
    if correct_cfo:
        est = estimate_cfo(signal)
        cfo = est["cfo_hz"]
        n = np.arange(len(x))
        x = x * np.exp(-1j * 2 * np.pi * cfo * n / fs)

    sps_est = fs / symbol_rate if symbol_rate > 0 else 8.0
    tsps = _target_sps(sps_est)
    frac = Fraction(tsps / sps_est).limit_denominator(2000)
    up, down = max(1, frac.numerator), max(1, frac.denominator)
    xr = resample_poly(x, up, down)

    taps = rrc(beta, tsps, span)
    mf = np.convolve(xr, taps, mode="same")

    # Symbol timing: the phase whose decimated stream has the most energy sits
    # on the pulse peaks (RRC*RRC = raised cosine, peak at symbol centers).
    best_phase, best_metric = 0, -np.inf
    for p in range(tsps):
        s = mf[p::tsps]
        if s.size < 4:
            continue
        metric = float(np.mean(np.abs(s) ** 2))
        if metric > best_metric:
            best_metric, best_phase = metric, p
    syms = mf[best_phase::tsps]
    power = np.mean(np.abs(syms) ** 2)
    if power > 0:
        syms = syms / np.sqrt(power)
    return {
        "symbols": syms.astype(np.complex64),
        "sps": tsps,
        "phase": best_phase,
        "cfo_hz": cfo,
        "resample": (up, down),
    }


def _evm(symbols: np.ndarray, points: np.ndarray) -> float:
    d = np.abs(symbols[:, None] - points[None, :])
    nearest = points[np.argmin(d, axis=1)]
    err = np.mean(np.abs(symbols - nearest) ** 2)
    ref = np.mean(np.abs(points) ** 2)
    return float(np.sqrt(err / ref) * 100.0)


def _resolve_rotation(symbols: np.ndarray, mod: str) -> tuple[np.ndarray, float, float]:
    """Try the constellation's rotational symmetries; return best (syms, phase, evm)."""
    fam, M = SUPPORTED_LINEAR[mod.upper()]
    points, _, _ = constellation(mod)
    n_rot = M if fam == "psk" else 4
    best = (symbols, 0.0, np.inf)
    for k in range(n_rot):
        phi = 2 * np.pi * k / n_rot
        rot = symbols * np.exp(1j * phi)
        e = _evm(rot, points)
        if e < best[2]:
            best = (rot, phi, e)
    return best


def _demod_psk_qam(signal: Signal, mod: str, symbol_rate: float) -> dict:
    fam, M = SUPPORTED_LINEAR[mod.upper()]
    rec = recover_symbols(signal, symbol_rate)
    syms = rec["symbols"].astype(np.complex128)
    if syms.size == 0:
        return {"modulation": mod, "symbols": syms, "bits": np.zeros(0, np.uint8),
                "evm_percent": 100.0, **rec}

    if fam == "psk":
        # M-th power carrier phase estimate, then resolve the M-fold ambiguity.
        phi = np.angle(np.mean(syms ** M)) / M
        syms = syms * np.exp(-1j * phi)
    syms, _, evm = _resolve_rotation(syms, mod)
    bits = demap_symbols(syms, mod)
    return {
        "modulation": mod,
        "symbols": syms.astype(np.complex64),
        "bits": bits,
        "evm_percent": evm,
        "sps": rec["sps"],
        "cfo_hz": rec["cfo_hz"],
        "n_symbols": int(syms.size),
    }


def _demod_fsk(signal: Signal, mod: str, symbol_rate: float) -> dict:
    M = 2 if mod.upper() in ("2FSK", "FSK", "BFSK") else 4
    x = signal.normalized().samples.astype(np.complex128)
    fs = signal.sample_rate
    inst = np.diff(np.unwrap(np.angle(x))) * fs / (2 * np.pi)
    sps = fs / symbol_rate if symbol_rate > 0 else 8.0
    n_sym = int(len(inst) / sps)
    if n_sym < 1:
        return {"modulation": mod, "bits": np.zeros(0, np.uint8), "n_symbols": 0,
                "tone_freqs": []}
    means = np.array([
        np.mean(inst[int(i * sps):int((i + 1) * sps)]) for i in range(n_sym)
    ])
    # M cluster centers via quantiles; assign each symbol to nearest center.
    centers = np.quantile(means, np.linspace(0.1, 0.9, M))
    vals = np.argmin(np.abs(means[:, None] - centers[None, :]), axis=1)
    k = int(np.log2(M))
    bits = np.array([[(v >> (k - 1 - j)) & 1 for j in range(k)] for v in vals],
                    dtype=np.uint8).reshape(-1)
    return {
        "modulation": mod,
        "bits": bits,
        "n_symbols": n_sym,
        "tone_freqs": [float(c) for c in centers],
        "symbols": means.astype(np.float64),
    }


def demodulate(signal: Signal, modulation: str, symbol_rate: float) -> dict:
    """Demodulate ``signal`` given a modulation label and symbol rate.

    Dispatches to the FSK or linear (PSK/QAM) path. This is the Tier-1
    "demod with estimated params" capability; the label and symbol rate normally
    come from :func:`spectra.dsp.classify_modulation` and
    :func:`spectra.dsp.estimate_symbol_rate`.
    """
    mod = modulation.upper()
    if "FSK" in mod:
        return _demod_fsk(signal, mod, symbol_rate)
    if mod in SUPPORTED_LINEAR:
        return _demod_psk_qam(signal, mod, symbol_rate)
    raise ValueError(f"unsupported modulation {modulation!r}")
