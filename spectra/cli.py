"""Command-line interface: ``python -m spectra.cli <command>``.

Commands
--------
info      show a file's header / summary
analyze   run the full analysis pipeline and print a report
generate  synthesize a labeled signal and write it to disk
suite     list the built-in synthetic demo suite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import loader
from .pipeline import analyze, format_report
from .synth import SignalSpec, default_suite, generate


def _load_any(args) -> "object":
    path = Path(args.path)
    name = path.name.lower()
    if name.endswith(".wav"):
        return loader.load_wav(path)
    if name.endswith(".sigmf-meta"):
        return loader.load_sigmf(path)
    if args.sample_rate is None:
        sys.exit("error: raw IQ requires --sample-rate (not stored in the file)")
    return loader.load_iq(path, sample_rate=args.sample_rate, dtype=args.dtype)


def cmd_info(args) -> None:
    sig = _load_any(args)
    print(json.dumps(sig.summary(), indent=2))


def cmd_analyze(args) -> None:
    sig = _load_any(args)
    results = analyze(
        sig,
        demod=not args.no_demod,
        known_modulation=args.modulation,
        known_symbol_rate=args.symbol_rate,
    )
    print(format_report(results))
    if args.json:
        results.pop("_demod_raw", None)
        Path(args.json).write_text(_to_json(results), encoding="utf-8")
        print(f"\n[written] {args.json}")


def cmd_generate(args) -> None:
    if args.preset:
        suite = default_suite()
        if args.preset not in suite:
            sys.exit(f"unknown preset {args.preset!r}; choose from {sorted(suite)}")
        spec = suite[args.preset]
    else:
        il = None
        if args.interleaver:
            parts = args.interleaver.split(":")
            il = (parts[0], int(parts[1]), int(parts[2]))
        spec = SignalSpec(
            modulation=args.modulation or "QPSK",
            symbol_rate=args.symbol_rate or 2400.0,
            sps=args.sps,
            snr_db=args.snr,
            cfo_hz=args.cfo,
            n_bits=args.bits,
            fec=args.fec,
            interleaver=il,
            seed=args.seed,
        )
    sig, label = generate(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    if fmt == "wav":
        _save_wav(out.with_suffix(".wav"), sig)
        written = out.with_suffix(".wav")
    elif fmt == "sigmf":
        meta, data = loader.save_sigmf(out.with_suffix(""), sig,
                                       extra_global={"spectra:label": _clean_label(label)})
        written = meta
    else:
        loader.save_iq(out.with_suffix(".iq"), sig, dtype="complex64")
        written = out.with_suffix(".iq")
    print(f"[generated] {spec.modulation} @ {spec.symbol_rate} baud, "
          f"{spec.snr_db} dB SNR -> {written}")
    print(f"[label] " + json.dumps(_clean_label(label)))


def cmd_suite(args) -> None:
    for name, spec in default_suite().items():
        print(f"{name:20s} {spec.modulation:6s} {spec.symbol_rate:>8.0f} baud "
              f"{spec.snr_db:>5.0f} dB  fec={spec.fec} il={spec.interleaver}")


def _save_wav(path, sig) -> None:
    from scipy.io import wavfile
    data = np.column_stack([sig.samples.real, sig.samples.imag]).astype(np.float32)
    wavfile.write(path, int(sig.sample_rate), data)


def _clean_label(label: dict) -> dict:
    return {k: v for k, v in label.items()
            if not isinstance(v, np.ndarray)}


def _to_json(obj) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    clean = {k: v for k, v in obj.items() if k != "_demod_raw"}
    return json.dumps(clean, indent=2, default=default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectra", description="SPECTRA signal analysis")
    p.add_argument("--version", action="version", version=f"SPECTRA {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common_load = argparse.ArgumentParser(add_help=False)
    common_load.add_argument("path", help="input file (.wav / .sigmf-meta / raw .iq)")
    common_load.add_argument("--sample-rate", type=float, default=None,
                             help="required for raw IQ (Hz)")
    common_load.add_argument("--dtype", default="complex64",
                             help="on-disk IQ dtype (int8/int16/float32/complex64)")

    pi = sub.add_parser("info", parents=[common_load], help="show file summary")
    pi.set_defaults(func=cmd_info)

    pa = sub.add_parser("analyze", parents=[common_load], help="run analysis")
    pa.add_argument("--modulation", default=None, help="force modulation label")
    pa.add_argument("--symbol-rate", type=float, default=None, help="force symbol rate")
    pa.add_argument("--no-demod", action="store_true", help="skip demodulation")
    pa.add_argument("--json", default=None, help="write results JSON to this path")
    pa.set_defaults(func=cmd_analyze)

    pg = sub.add_parser("generate", help="synthesize a labeled signal")
    pg.add_argument("out", help="output path (extension set by --format)")
    pg.add_argument("--preset", default=None, help="use a named suite preset")
    pg.add_argument("--modulation", default=None)
    pg.add_argument("--symbol-rate", type=float, default=None)
    pg.add_argument("--sps", type=int, default=8)
    pg.add_argument("--snr", type=float, default=20.0)
    pg.add_argument("--cfo", type=float, default=0.0)
    pg.add_argument("--bits", type=int, default=2048)
    pg.add_argument("--fec", default="none", choices=["none", "conv", "hamming"])
    pg.add_argument("--interleaver", default=None,
                    help="block:ROWS:COLS or conv:BRANCHES:DEPTH")
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--format", default="sigmf", choices=["sigmf", "iq", "wav"])
    pg.set_defaults(func=cmd_generate)

    ps = sub.add_parser("suite", help="list built-in demo presets")
    ps.set_defaults(func=cmd_suite)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
