"""Bitstream correlation utilities for framing / header-payload identification."""

from __future__ import annotations

import numpy as np


def _pm1(bits: np.ndarray) -> np.ndarray:
    """Map {0,1} bits to {-1,+1} for correlation."""
    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64).ravel()


def xcorr_bits(a: np.ndarray, b: np.ndarray) -> dict:
    """Slide the shorter pattern ``b`` across ``a`` and score agreement.

    Returns the best lag and a normalized score in [-1, 1] (1 = exact match,
    -1 = exact complement). Also returns the full normalized correlation.
    """
    a1, b1 = _pm1(a), _pm1(b)
    if len(b1) > len(a1):
        a1, b1 = b1, a1
    corr = np.correlate(a1, b1, mode="valid") / len(b1)
    best_lag = int(np.argmax(np.abs(corr)))
    best_score = float(corr[best_lag])
    return {
        "lag": best_lag,
        "score": best_score,
        "inverted": best_score < 0,
        "corr": corr,
    }


def find_sync(bits: np.ndarray, sync: np.ndarray,
              min_match: float = 0.9) -> list[dict]:
    """Find all positions where ``sync`` appears in ``bits``.

    A hit is reported where the fraction of matching bits (or matching after
    inversion) is >= ``min_match``. Useful for locating preambles / sync words.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    sync = np.asarray(sync, dtype=np.uint8).ravel()
    m = len(sync)
    if m == 0 or len(bits) < m:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(bits, m)
    matches = np.count_nonzero(windows == sync, axis=1) / m
    inv_matches = np.count_nonzero(windows == (1 - sync), axis=1) / m
    hits = []
    for pos in np.where((matches >= min_match) | (inv_matches >= min_match))[0]:
        frac = float(matches[pos])
        inv = inv_matches[pos] > matches[pos]
        hits.append({
            "position": int(pos),
            "match": max(frac, float(inv_matches[pos])),
            "inverted": bool(inv),
        })
    return hits


def frame_segments(bits: np.ndarray, sync: np.ndarray, header_len: int,
                   min_match: float = 0.9) -> list[dict]:
    """Split ``bits`` into frames at sync-word boundaries.

    Each frame is described by its sync position, header slice and payload slice
    (payload runs up to the next sync). This is the raw material the analyst uses
    to separate header from payload.
    """
    hits = find_sync(bits, sync, min_match=min_match)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    m = len(sync)
    frames = []
    for i, hit in enumerate(hits):
        start = hit["position"]
        end = hits[i + 1]["position"] if i + 1 < len(hits) else len(bits)
        body = bits[start + m:end]
        frames.append({
            "sync_position": start,
            "header": body[:header_len].copy(),
            "payload": body[header_len:].copy(),
            "length": int(end - start),
            "inverted": hit["inverted"],
        })
    return frames


def estimate_period(bits: np.ndarray, max_period: int = 4096) -> dict:
    """Blind periodicity detection via bit autocorrelation.

    Returns the most likely repetition period (e.g. frame length) and a
    confidence in [0, 1]. Used when no known sync word is available.
    """
    x = _pm1(bits)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return {"period": 0, "confidence": 0.0}
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0] if ac[0] != 0 else ac
    hi = min(max_period, n - 1)
    if hi < 2:
        return {"period": 0, "confidence": 0.0}
    lag = int(np.argmax(ac[1:hi])) + 1
    return {"period": lag, "confidence": float(np.clip(ac[lag], 0.0, 1.0))}
