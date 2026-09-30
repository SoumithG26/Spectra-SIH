"""Feature-based automatic modulation classification (AMC).

This is the explainable, training-free Tier-1 classifier: it separates the
constant-envelope (FSK) family from linearly-modulated signals using the
envelope coefficient of variation, then uses the recovered-symbol amplitude
spread to split PSK from QAM, an M-th-power tone test to find PSK order, and an
amplitude-ring measure to guess QAM order. Each decision yields a confidence.

A learned CNN/CLDNN classifier (trained on the synthetic factory + RadioML) is
the roadmap upgrade; it slots in behind the same :func:`classify_modulation`
interface and the workbench continues to show confidence + rationale.
"""

from __future__ import annotations

import numpy as np

from ..core import Signal
from .demod import recover_symbols
from .params import estimate_symbol_rate

_CANDIDATES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK"]


def _fsk_score(signal: Signal) -> tuple[float, float]:
    """Return (envelope_cv, freq_spread). Low envelope CV => constant modulus."""
    x = signal.normalized().samples
    amp = np.abs(x)
    env_cv = float(np.std(amp) / (np.mean(amp) + 1e-12))
    inst = np.diff(np.unwrap(np.angle(x.astype(np.complex128))))
    freq_spread = float(np.std(inst) / (np.pi + 1e-12))
    return env_cv, freq_spread


def _choose_fsk_order(signal: Signal, symbol_rate: float) -> tuple[str, float]:
    x = signal.normalized().samples.astype(np.complex128)
    fs = signal.sample_rate
    inst = np.diff(np.unwrap(np.angle(x))) * fs / (2 * np.pi)
    sps = fs / symbol_rate if symbol_rate > 0 else 8.0
    n_sym = max(1, int(len(inst) / sps))
    means = np.array([np.mean(inst[int(i * sps):int((i + 1) * sps)])
                      for i in range(n_sym)])
    best_M, best_ratio = 2, np.inf
    for M in (2, 4):
        centers = np.quantile(means, np.linspace(0.1, 0.9, M))
        assign = np.argmin(np.abs(means[:, None] - centers[None, :]), axis=1)
        within = np.mean([(means[assign == c] - centers[c]).var()
                          for c in range(M) if np.any(assign == c)])
        between = np.var(centers) + 1e-12
        ratio = within / between
        if ratio < best_ratio:
            best_ratio, best_M = ratio, M
    name = "2FSK" if best_M == 2 else "4FSK"
    conf = float(np.clip(1.0 - best_ratio, 0.3, 0.95))
    return name, conf


def _psk_order(symbols: np.ndarray) -> tuple[str, float]:
    s = symbols[np.abs(symbols) > 1e-6]
    if s.size == 0:
        return "QPSK", 0.2
    u = s / np.abs(s)
    concentration = {M: float(np.abs(np.mean(u ** M))) for M in (2, 4, 8)}
    for M, name in ((2, "BPSK"), (4, "QPSK"), (8, "8PSK")):
        if concentration[M] > 0.5:
            return name, float(np.clip(concentration[M], 0.5, 0.98))
    # fall back to the strongest tone
    M = max(concentration, key=concentration.get)
    return {2: "BPSK", 4: "QPSK", 8: "8PSK"}[M], float(concentration[M])


def _qam_order(symbols: np.ndarray) -> tuple[str, float]:
    amp = np.abs(symbols)
    cv = float(np.std(amp) / (np.mean(amp) + 1e-12))
    # 64QAM has a wider amplitude spread (more rings) than 16QAM.
    if cv < 0.47:
        return "16QAM", float(np.clip(0.47 - cv, 0.3, 0.8) + 0.2)
    return "64QAM", float(np.clip(cv - 0.47, 0.3, 0.8) + 0.2)


def classify_modulation(signal: Signal, symbol_rate: float | None = None) -> dict:
    """Classify the modulation and return a ranked, confidence-scored result."""
    if symbol_rate is None:
        symbol_rate = estimate_symbol_rate(signal)["symbol_rate"]

    env_cv, freq_spread = _fsk_score(signal)
    scores: dict[str, float] = {}

    # Constant-envelope => FSK family.
    if env_cv < 0.15:
        name, conf = _choose_fsk_order(signal, symbol_rate)
        scores[name] = conf
        other = "4FSK" if name == "2FSK" else "2FSK"
        scores[other] = max(0.05, 1.0 - conf - 0.1)
        family = "fsk"
    else:
        rec = recover_symbols(signal, symbol_rate or signal.sample_rate / 8.0)
        syms = rec["symbols"].astype(np.complex128)
        amp = np.abs(syms)
        amp_cv = float(np.std(amp) / (np.mean(amp) + 1e-12))
        if amp_cv < 0.35:  # tight rings => PSK
            name, conf = _psk_order(syms)
            scores[name] = conf
            family = "psk"
        else:
            name, conf = _qam_order(syms)
            scores[name] = conf
            family = "qam"

    for c in _CANDIDATES:
        scores.setdefault(c, 0.02)
    total = sum(scores.values())
    ranking = sorted(((k, v / total) for k, v in scores.items()),
                     key=lambda kv: kv[1], reverse=True)
    top, top_conf = ranking[0]
    return {
        "prediction": top,
        "confidence": round(float(top_conf), 3),
        "family": family,
        "ranking": [(k, round(float(v), 3)) for k, v in ranking],
        "features": {
            "envelope_cv": round(env_cv, 4),
            "freq_spread": round(freq_spread, 4),
            "symbol_rate_used": symbol_rate,
        },
    }
