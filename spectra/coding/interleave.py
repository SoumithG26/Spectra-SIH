"""Interleavers and their inverses.

Interleaving reorders bits so that a burst of channel errors is spread across
many codewords after de-interleaving, which FEC can then correct. We implement
the two most common structured families with exact inverses:

* **Block** interleaver: write row-wise into an R x C matrix, read column-wise.
* **Convolutional** (Forney) interleaver: ``branches`` shift registers with
  linearly increasing delay ``depth``.

Diagonal and pseudo-random interleavers are special cases of a general
permutation; :func:`permute` / :func:`invert_permutation` cover those and are
used by the (roadmap) blind hypothesis search.
"""

from __future__ import annotations

import numpy as np


def _as_bits(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint8).ravel()


def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Block-interleave: fill rows, read columns. Pads with zeros to R*C."""
    bits = _as_bits(bits)
    n = rows * cols
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    out = bits.reshape(-1, rows, cols).transpose(0, 2, 1).reshape(-1)
    return out.astype(np.uint8)


def block_deinterleave(bits: np.ndarray, rows: int, cols: int,
                       original_len: int | None = None) -> np.ndarray:
    """Invert :func:`block_interleave`. Trims to ``original_len`` if given."""
    bits = _as_bits(bits)
    n = rows * cols
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    out = bits.reshape(-1, cols, rows).transpose(0, 2, 1).reshape(-1)
    if original_len is not None:
        out = out[:original_len]
    return out.astype(np.uint8)


def conv_interleave(bits: np.ndarray, branches: int, depth: int) -> np.ndarray:
    """Forney convolutional interleaver.

    Branch ``i`` (0-indexed) has a shift register of length ``i * depth``. Input
    symbols are fed round-robin across branches. Output length equals input
    length; register contents start at zero, producing a fixed end-to-end delay
    of ``branches * (branches - 1) * depth`` symbols after the matching
    de-interleaver (see :func:`conv_deinterleave`).
    """
    bits = _as_bits(bits)
    regs = [np.zeros(i * depth, dtype=np.uint8) for i in range(branches)]
    out = np.empty(len(bits), dtype=np.uint8)
    for k, b in enumerate(bits):
        i = k % branches
        reg = regs[i]
        if reg.size == 0:
            out[k] = b
        else:
            out[k] = reg[-1]
            regs[i] = np.concatenate(([b], reg[:-1]))
    return out


def conv_deinterleave(bits: np.ndarray, branches: int, depth: int) -> np.ndarray:
    """Matching de-interleaver: branch ``i`` delay is ``(branches-1-i) * depth``."""
    bits = _as_bits(bits)
    regs = [np.zeros((branches - 1 - i) * depth, dtype=np.uint8)
            for i in range(branches)]
    out = np.empty(len(bits), dtype=np.uint8)
    for k, b in enumerate(bits):
        i = k % branches
        reg = regs[i]
        if reg.size == 0:
            out[k] = b
        else:
            out[k] = reg[-1]
            regs[i] = np.concatenate(([b], reg[:-1]))
    return out


def conv_delay(branches: int, depth: int) -> int:
    """End-to-end delay (in symbols) of the conv interleave+deinterleave pair."""
    return branches * (branches - 1) * depth


def permute(bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """General block permutation over blocks of ``len(perm)`` (diagonal / PRN)."""
    bits = _as_bits(bits)
    p = np.asarray(perm, dtype=int)
    n = len(p)
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits.reshape(-1, n)
    return blocks[:, p].reshape(-1).astype(np.uint8)


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    """Return ``inv`` such that ``permute(permute(x, perm), inv) == x``."""
    p = np.asarray(perm, dtype=int)
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv
