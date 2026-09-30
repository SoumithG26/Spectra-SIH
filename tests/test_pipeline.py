"""Smoke test for the end-to-end analysis pipeline."""

from spectra.pipeline import analyze, format_report
from spectra.synth import SignalSpec, generate


def test_analyze_returns_expected_sections():
    sig, _ = generate(SignalSpec("QPSK", 4800, 8, snr_db=25, n_bits=2000, seed=6))
    res = analyze(sig)
    for key in ["signal", "survey", "parameters", "modulation", "demod"]:
        assert key in res
    assert res["parameters"]["symbol_rate_est_hz"] > 0
    report = format_report(res)
    assert "SPECTRA analysis report" in report


def test_analyze_with_known_params():
    sig, _ = generate(SignalSpec("BPSK", 2400, 8, snr_db=25, n_bits=1000, seed=7))
    res = analyze(sig, known_modulation="BPSK", known_symbol_rate=2400)
    assert res["modulation_used"] == "BPSK"
    assert res["demod"]["n_bits"] > 0
