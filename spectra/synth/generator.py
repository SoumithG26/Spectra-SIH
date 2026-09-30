"""Synthetic signal generation with ground-truth labels.

Chain:  message bits -> FEC encode -> interleave -> symbol map / pulse shape
        -> CFO + phase noise -> AWGN -> complex baseband IQ.

Every stage's parameters are recorded in the returned label so the analysis
pipeline can be scored against truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..coding.fec import ConvCode, hamming74_encode
from ..coding.interleave import block_interleave, conv_interleave
from ..core import Signal
from ..dsp.modem import SUPPORTED_LINEAR, bits_per_symbol, map_symbols, rrc


@dataclass
class SignalSpec:
    """Parameters for one synthetic recording."""

    modulation: str = "QPSK"          # BPSK/QPSK/8PSK/16QAM/64QAM/2FSK/4FSK
    symbol_rate: float = 2400.0       # baud
    sps: int = 8                      # samples per symbol (=> sample_rate)
    snr_db: float = 20.0
    cfo_hz: float = 0.0
    beta: float = 0.35                # RRC roll-off (linear mods)
    freq_sep: float | None = None     # FSK tone spacing (default = symbol_rate)
    phase_noise: float = 0.0          # rad RMS random-walk step
    n_bits: int = 2048
    fec: str = "none"                 # none / conv / hamming
    interleaver: tuple | None = None  # None / ("block",rows,cols) / ("conv",N,D)
    seed: int = 0

    @property
    def sample_rate(self) -> float:
        return self.symbol_rate * self.sps


def _apply_fec(bits: np.ndarray, fec: str) -> tuple[np.ndarray, dict]:
    if fec == "none":
        return bits, {"fec": "none"}
    if fec == "conv":
        code = ConvCode()
        return code.encode(bits), {"fec": "conv", "K": code.K, "gens": ["171", "133"]}
    if fec == "hamming":
        return hamming74_encode(bits), {"fec": "hamming", "n": 7, "k": 4}
    raise ValueError(f"unknown fec {fec!r}")


def _apply_interleave(bits: np.ndarray, spec_il: tuple | None) -> tuple[np.ndarray, dict]:
    if spec_il is None:
        return bits, {"interleaver": "none"}
    kind = spec_il[0]
    if kind == "block":
        _, rows, cols = spec_il
        return block_interleave(bits, rows, cols), {
            "interleaver": "block", "rows": rows, "cols": cols}
    if kind == "conv":
        _, branches, depth = spec_il
        return conv_interleave(bits, branches, depth), {
            "interleaver": "conv", "branches": branches, "depth": depth}
    raise ValueError(f"unknown interleaver {spec_il!r}")


def _modulate_linear(bits: np.ndarray, spec: SignalSpec) -> np.ndarray:
    symbols = map_symbols(bits, spec.modulation)
    up = np.zeros(len(symbols) * spec.sps, dtype=np.complex128)
    up[:: spec.sps] = symbols
    taps = rrc(spec.beta, spec.sps, span=8)
    bb = np.convolve(up, taps, mode="same")
    return bb


def _modulate_fsk(bits: np.ndarray, spec: SignalSpec) -> np.ndarray:
    M = 2 if spec.modulation.upper() in ("2FSK", "FSK", "BFSK") else 4
    k = int(np.log2(M))
    b = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(b)) % k
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    groups = b.reshape(-1, k)
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    vals = groups.astype(np.int64) @ weights
    spacing = spec.freq_sep if spec.freq_sep is not None else spec.symbol_rate
    tone = (vals - (M - 1) / 2.0) * spacing
    inst = np.repeat(tone, spec.sps)
    phase = 2 * np.pi * np.cumsum(inst) / spec.sample_rate
    return np.exp(1j * phase)


def generate(spec: SignalSpec) -> tuple[Signal, dict]:
    """Generate one labeled synthetic signal."""
    rng = np.random.default_rng(spec.seed)
    message = rng.integers(0, 2, size=spec.n_bits, dtype=np.uint8)

    coded, fec_meta = _apply_fec(message, spec.fec)
    interleaved, il_meta = _apply_interleave(coded, spec.interleaver)

    mod = spec.modulation.upper()
    if "FSK" in mod:
        bb = _modulate_fsk(interleaved, spec)
    elif mod in SUPPORTED_LINEAR:
        bb = _modulate_linear(interleaved, spec)
    else:
        raise ValueError(f"unsupported modulation {spec.modulation!r}")

    # normalize to unit power
    p = np.mean(np.abs(bb) ** 2)
    if p > 0:
        bb = bb / np.sqrt(p)

    # carrier-frequency offset
    n = np.arange(len(bb))
    if spec.cfo_hz:
        bb = bb * np.exp(1j * 2 * np.pi * spec.cfo_hz * n / spec.sample_rate)

    # phase noise (random walk)
    if spec.phase_noise > 0:
        pn = np.cumsum(rng.normal(0, spec.phase_noise, size=len(bb)))
        bb = bb * np.exp(1j * pn)

    # AWGN for the target SNR
    sig_pow = np.mean(np.abs(bb) ** 2)
    noise_pow = sig_pow / (10 ** (spec.snr_db / 10.0))
    noise = np.sqrt(noise_pow / 2) * (rng.standard_normal(len(bb))
                                      + 1j * rng.standard_normal(len(bb)))
    bb = bb + noise

    label: dict[str, Any] = {
        "modulation": mod,
        "symbol_rate": spec.symbol_rate,
        "sample_rate": spec.sample_rate,
        "sps": spec.sps,
        "snr_db": spec.snr_db,
        "cfo_hz": spec.cfo_hz,
        "beta": spec.beta,
        "n_message_bits": int(len(message)),
        "message_bits": message,
        "tx_symbol_bits": np.asarray(interleaved, dtype=np.uint8),
        **fec_meta,
        **il_meta,
    }
    sig = Signal(bb, sample_rate=spec.sample_rate, source=f"synthetic:{mod}",
                 meta={"synthetic": True, "label": label})
    return sig, label


def default_suite() -> dict[str, SignalSpec]:
    """A small labeled set spanning the modulation families (demos + tests)."""
    return {
        "bpsk_clean": SignalSpec("BPSK", 2400, 8, snr_db=25, seed=1),
        "qpsk_cfo": SignalSpec("QPSK", 4800, 8, snr_db=20, cfo_hz=300, seed=2),
        "8psk": SignalSpec("8PSK", 4800, 8, snr_db=25, seed=3),
        "16qam": SignalSpec("16QAM", 9600, 8, snr_db=28, seed=4),
        "64qam": SignalSpec("64QAM", 9600, 8, snr_db=32, seed=5),
        "2fsk": SignalSpec("2FSK", 1200, 10, snr_db=20, seed=6),
        "4fsk": SignalSpec("4FSK", 2400, 10, snr_db=22, seed=7),
        "qpsk_conv_block": SignalSpec(
            "QPSK", 4800, 8, snr_db=18, seed=8,
            fec="conv", interleaver=("block", 8, 16)),
    }
