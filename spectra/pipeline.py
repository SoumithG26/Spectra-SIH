"""End-to-end analysis orchestration: SURVEY -> ESTIMATE -> CLASSIFY -> DEMOD.

Ties the DSP modules together into a single :func:`analyze` call that the CLI
and GUI both use, so there is one code path and one place decisions are scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import Signal
from .dsp.demod import demodulate
from .dsp.modulation import classify_modulation
from .dsp.params import estimate_cfo, estimate_snr_db, estimate_symbol_rate, occupied_bandwidth


def analyze(
    signal: Signal,
    *,
    demod: bool = True,
    known_modulation: str | None = None,
    known_symbol_rate: float | None = None,
) -> dict[str, Any]:
    """Run the front-end analysis chain and return a structured result dict.

    ``known_*`` let the analyst override automatic estimates (the workbench's
    human-in-the-loop path); when omitted, everything is estimated.
    """
    results: dict[str, Any] = {"signal": signal.summary()}

    # --- survey ---
    bw = occupied_bandwidth(signal)
    snr = estimate_snr_db(signal)
    results["survey"] = {
        "occupied_bandwidth_hz": round(bw["bandwidth"], 2),
        "band_center_hz": round(bw["center"], 2),
        "snr_db": round(snr["snr_db"], 2),
        "noise_floor": snr["noise_floor"],
    }

    # --- front-end estimation ---
    cfo = estimate_cfo(signal)
    sr = estimate_symbol_rate(signal)
    symbol_rate = known_symbol_rate if known_symbol_rate else sr["symbol_rate"]
    results["parameters"] = {
        "cfo_hz": round(cfo["cfo_hz"], 3),
        "cfo_confidence": round(cfo["confidence"], 3),
        "symbol_rate_est_hz": round(sr["symbol_rate"], 2),
        "symbol_rate_confidence": round(sr["confidence"], 3),
        "sps_est": round(sr["sps"], 3),
        "symbol_rate_used_hz": round(float(symbol_rate), 2),
    }

    # --- modulation classification ---
    mod = classify_modulation(signal, symbol_rate=symbol_rate or None)
    results["modulation"] = mod
    chosen_mod = known_modulation or mod["prediction"]
    results["modulation_used"] = chosen_mod

    # --- demodulation ---
    if demod and symbol_rate and symbol_rate > 0:
        try:
            dm = demodulate(signal, chosen_mod, symbol_rate)
            bits = dm.get("bits", np.zeros(0, np.uint8))
            results["demod"] = {
                "modulation": dm.get("modulation", chosen_mod),
                "n_symbols": dm.get("n_symbols", 0),
                "n_bits": int(len(bits)),
                "evm_percent": round(float(dm.get("evm_percent", float("nan"))), 2)
                if "evm_percent" in dm else None,
                "bits_preview": bits[:64].tolist(),
                "tone_freqs_hz": dm.get("tone_freqs"),
            }
            results["_demod_raw"] = dm  # arrays for the GUI (constellation etc.)
        except Exception as exc:  # keep the pipeline robust in a demo setting
            results["demod"] = {"error": str(exc)}

    return results


def format_report(results: dict[str, Any]) -> str:
    """Human-readable text report (CLI output / log)."""
    s = results["signal"]
    sv = results["survey"]
    p = results["parameters"]
    m = results["modulation"]
    lines = [
        "=" * 60,
        "SPECTRA analysis report",
        "=" * 60,
        f"source            : {s['source']}",
        f"samples           : {s['samples']:,}  ({s['duration_s']} s)",
        f"sample rate       : {s['sample_rate_hz']:.1f} Hz",
        f"real-valued input : {s['real_valued']}",
        "-" * 60,
        "SURVEY",
        f"  occupied BW     : {sv['occupied_bandwidth_hz']:.1f} Hz",
        f"  band center     : {sv['band_center_hz']:.1f} Hz",
        f"  est. SNR        : {sv['snr_db']:.1f} dB",
        "-" * 60,
        "PARAMETERS",
        f"  CFO             : {p['cfo_hz']:.1f} Hz  (conf {p['cfo_confidence']})",
        f"  symbol rate     : {p['symbol_rate_est_hz']:.1f} Hz  "
        f"(conf {p['symbol_rate_confidence']}, ~{p['sps_est']} sps)",
        "-" * 60,
        "MODULATION",
        f"  prediction      : {m['prediction']}  (conf {m['confidence']}, "
        f"family {m['family']})",
        f"  ranking         : " + ", ".join(f"{k}:{v}" for k, v in m["ranking"][:4]),
    ]
    if "demod" in results and "error" not in results["demod"]:
        d = results["demod"]
        lines += [
            "-" * 60,
            "DEMODULATION",
            f"  modulation used : {d['modulation']}",
            f"  symbols         : {d['n_symbols']:,}",
            f"  bits            : {d['n_bits']:,}",
        ]
        if d.get("evm_percent") is not None:
            lines.append(f"  EVM             : {d['evm_percent']} %")
        if d.get("tone_freqs_hz"):
            tones = ", ".join(f"{t:.0f}" for t in d["tone_freqs_hz"])
            lines.append(f"  tone freqs (Hz) : {tones}")
    lines.append("=" * 60)
    return "\n".join(lines)
