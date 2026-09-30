"""Loaders that converge WAV / raw IQ / SigMF onto :class:`spectra.core.Signal`.

Design notes
------------
* **WAV** carries its sample rate in the header. Stereo WAV is treated as
  interleaved IQ (I = left, Q = right) - the common SDR convention. Mono WAV is
  treated as a real recording and converted to an analytic signal (Hilbert), so
  downstream code always sees complex baseband.
* **Raw .iq** has *no* header. The caller must supply ``sample_rate`` and the
  on-disk ``dtype``; we never guess these silently.
* **SigMF** (``.sigmf-meta`` + ``.sigmf-data``) is the metadata-bearing path and
  is preferred whenever available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

from ..core import Signal

# Map friendly dtype names -> (numpy dtype, is_already_complex).
IQ_DTYPES: dict[str, tuple[np.dtype, bool]] = {
    "int8": (np.dtype("int8"), False),
    "uint8": (np.dtype("uint8"), False),
    "int16": (np.dtype("int16"), False),
    "float32": (np.dtype("float32"), False),
    "float64": (np.dtype("float64"), False),
    "complex64": (np.dtype("complex64"), True),
    "complex128": (np.dtype("complex128"), True),
}

# SigMF datatype string -> our IQ_DTYPES key. Only little-endian handled here.
_SIGMF_DTYPE = {
    "ci8": "int8", "cu8": "uint8", "ci16_le": "int16",
    "cf32_le": "float32", "cf64_le": "float64",
    "ri8": "int8", "ru8": "uint8", "ri16_le": "int16",
    "rf32_le": "float32", "rf64_le": "float64",
}


def _scale_int(raw: np.ndarray) -> np.ndarray:
    """Scale integer PCM to roughly [-1, 1] float32."""
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) - 127.5) / 127.5
    info = np.iinfo(raw.dtype)
    return raw.astype(np.float32) / float(max(abs(info.min), info.max))


def load_wav(path: str | os.PathLike, *, force_real: bool = False) -> Signal:
    """Load a WAV file.

    Stereo -> IQ (I=ch0, Q=ch1). Mono -> real signal converted to analytic.
    ``force_real`` treats even a 2-channel file as two independent real channels
    (ch0 only is used) rather than IQ.
    """
    path = Path(path)
    fs, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        data = _scale_int(data)
    else:
        data = data.astype(np.float32)

    meta: dict[str, Any] = {"wav_dtype": str(data.dtype), "loader": "wav"}
    if data.ndim == 2 and data.shape[1] >= 2 and not force_real:
        samples = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
        meta["wav_interpretation"] = "iq_stereo"
    else:
        mono = data[:, 0] if data.ndim == 2 else data
        # Analytic signal keeps spectral content on the positive side only.
        samples = hilbert(mono.astype(np.float64)).astype(np.complex64)
        meta["wav_interpretation"] = "real_analytic"
    return Signal(samples, sample_rate=float(fs), source=str(path), meta=meta)


def load_iq(
    path: str | os.PathLike,
    sample_rate: float,
    dtype: str = "complex64",
    *,
    center_freq: float = 0.0,
    offset_samples: int = 0,
    max_samples: int | None = None,
) -> Signal:
    """Load a raw, header-less IQ file.

    Parameters
    ----------
    dtype:
        On-disk element type (see :data:`IQ_DTYPES`). Real dtypes are read as
        interleaved I,Q,I,Q,... and combined into complex samples.
    """
    if dtype not in IQ_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(IQ_DTYPES)}")
    np_dtype, already_complex = IQ_DTYPES[dtype]

    itemsize = np_dtype.itemsize
    # For interleaved real data, one complex sample = 2 elements.
    elems_per_sample = 1 if already_complex else 2
    byte_offset = offset_samples * elems_per_sample * itemsize
    count = -1 if max_samples is None else max_samples * elems_per_sample

    raw = np.fromfile(path, dtype=np_dtype, count=count, offset=byte_offset)
    if already_complex:
        samples = raw.astype(np.complex64)
    else:
        if np.issubdtype(np_dtype, np.integer):
            raw = _scale_int(raw)
        raw = raw.astype(np.float32)
        if raw.size % 2 == 1:
            raw = raw[:-1]  # drop dangling sample
        samples = raw[0::2] + 1j * raw[1::2]

    return Signal(
        samples,
        sample_rate=float(sample_rate),
        center_freq=float(center_freq),
        source=str(path),
        meta={"loader": "iq", "on_disk_dtype": dtype},
    )


def load_sigmf(meta_path: str | os.PathLike) -> Signal:
    """Load a SigMF pair given the ``.sigmf-meta`` path.

    The data file is resolved by swapping the extension to ``.sigmf-data``.
    Reads sample rate, center frequency and datatype from the metadata.
    """
    meta_path = Path(meta_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    global_meta = meta.get("global", {})
    datatype = global_meta.get("core:datatype", "cf32_le")
    sample_rate = float(global_meta.get("core:sample_rate", 1.0))

    center_freq = 0.0
    captures = meta.get("captures", [])
    if captures:
        center_freq = float(captures[0].get("core:frequency", 0.0))

    if datatype not in _SIGMF_DTYPE:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}")
    dtype = _SIGMF_DTYPE[datatype]

    data_path = meta_path.with_suffix("")  # strip .sigmf-meta -> .sigmf
    data_path = data_path.with_suffix(".sigmf-data")
    if not data_path.exists():
        # some tools name it foo.sigmf-data next to foo.sigmf-meta
        alt = Path(str(meta_path).replace(".sigmf-meta", ".sigmf-data"))
        data_path = alt

    sig = load_iq(data_path, sample_rate=sample_rate, dtype=dtype,
                  center_freq=center_freq)
    sig.source = str(meta_path)
    sig.meta.update({"loader": "sigmf", "sigmf_global": global_meta})
    return sig


def load(path: str | os.PathLike, **kwargs: Any) -> Signal:
    """Dispatch on file extension.

    * ``.wav`` -> :func:`load_wav`
    * ``.sigmf-meta`` -> :func:`load_sigmf`
    * ``.iq`` / ``.dat`` / ``.cfile`` / ``.bin`` -> :func:`load_iq`
      (requires ``sample_rate`` and optionally ``dtype`` in kwargs)
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".wav"):
        return load_wav(p, **{k: v for k, v in kwargs.items() if k == "force_real"})
    if name.endswith(".sigmf-meta"):
        return load_sigmf(p)
    if name.endswith((".iq", ".dat", ".cfile", ".bin", ".cf32", ".c64")):
        if "sample_rate" not in kwargs:
            raise ValueError("raw IQ requires sample_rate=... (not stored in file)")
        return load_iq(p, **kwargs)
    raise ValueError(f"cannot infer format for {p.name!r}; use a specific loader")


def save_iq(path: str | os.PathLike, signal: Signal, dtype: str = "complex64") -> None:
    """Write a Signal as raw IQ (for round-tripping / fixtures)."""
    if dtype not in IQ_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}")
    np_dtype, already_complex = IQ_DTYPES[dtype]
    if already_complex:
        signal.samples.astype(np_dtype).tofile(path)
    else:
        inter = np.empty(len(signal) * 2, dtype=np.float32)
        inter[0::2] = signal.samples.real
        inter[1::2] = signal.samples.imag
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            inter = np.clip(inter * info.max, info.min, info.max)
        inter.astype(np_dtype).tofile(path)


def save_sigmf(
    base_path: str | os.PathLike,
    signal: Signal,
    *,
    datatype: str = "cf32_le",
    extra_global: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a SigMF pair. ``base_path`` should have no extension.

    Returns (meta_path, data_path).
    """
    base = Path(base_path)
    dtype = _SIGMF_DTYPE.get(datatype)
    if dtype is None:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}")
    data_path = base.with_suffix(".sigmf-data")
    meta_path = base.with_suffix(".sigmf-meta")
    save_iq(data_path, signal, dtype=dtype)

    global_meta = {
        "core:datatype": datatype,
        "core:sample_rate": signal.sample_rate,
        "core:version": "1.0.0",
    }
    if extra_global:
        global_meta.update(extra_global)
    doc = {
        "global": global_meta,
        "captures": [{"core:sample_start": 0, "core:frequency": signal.center_freq}],
        "annotations": [],
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return meta_path, data_path
