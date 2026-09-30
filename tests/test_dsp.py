"""Tests for parameter estimation, modulation classification and demodulation."""

import numpy as np

from spectra.dsp import classify_modulation, demodulate, estimate_symbol_rate
from spectra.synth import SignalSpec, default_suite, generate

FAMILY = {
    "BPSK": "psk", "QPSK": "psk", "8PSK": "psk",
    "16QAM": "qam", "64QAM": "qam",
    "2FSK": "fsk", "4FSK": "fsk",
}


def test_symbol_rate_estimate_linear():
    for mod in ["BPSK", "QPSK", "16QAM"]:
        spec = SignalSpec(mod, symbol_rate=4800, sps=8, snr_db=30,
                          n_bits=4000, seed=1)
        sig, _ = generate(spec)
        est = estimate_symbol_rate(sig)["symbol_rate"]
        assert abs(est - 4800) / 4800 < 0.15, (mod, est)


def test_modulation_family_across_suite():
    for name, spec in default_suite().items():
        sig, label = generate(spec)
        res = classify_modulation(sig)
        assert res["family"] == FAMILY[label["modulation"]], (name, res)


def test_demod_evm_low_at_high_snr():
    for mod in ["BPSK", "QPSK"]:
        spec = SignalSpec(mod, symbol_rate=4800, sps=8, snr_db=30,
                          n_bits=4000, seed=2)
        sig, _ = generate(spec)
        dm = demodulate(sig, mod, 4800)
        assert dm["evm_percent"] < 20.0, (mod, dm["evm_percent"])
