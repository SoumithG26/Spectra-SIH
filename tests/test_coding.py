"""Deterministic tests for interleaving, FEC and correlation."""

import numpy as np

from spectra.coding import (
    ConvCode,
    block_deinterleave,
    block_interleave,
    conv_deinterleave,
    conv_interleave,
    find_sync,
    hamming74_decode,
    hamming74_encode,
    viterbi_decode,
)
from spectra.coding.interleave import conv_delay

rng = np.random.default_rng(0)


def test_conv_viterbi_clean_roundtrip():
    code = ConvCode()
    msg = rng.integers(0, 2, 60).astype(np.uint8)
    dec = viterbi_decode(code, code.encode(msg))
    assert np.array_equal(dec[: len(msg)], msg)


def test_conv_viterbi_corrects_scattered_errors():
    code = ConvCode()
    msg = rng.integers(0, 2, 200).astype(np.uint8)
    cw = code.encode(msg)
    r = cw.copy()
    idx = rng.choice(len(r), size=8, replace=False)
    r[idx] ^= 1
    dec = viterbi_decode(code, r)
    assert np.array_equal(dec[: len(msg)], msg)


def test_hamming_single_error_per_block():
    data = rng.integers(0, 2, 4 * 30).astype(np.uint8)
    enc = hamming74_encode(data).reshape(-1, 7)
    for blk in enc:
        blk[rng.integers(0, 7)] ^= 1
    dec = hamming74_decode(enc.reshape(-1), data_len=len(data))
    assert np.array_equal(dec, data)


def test_block_interleave_is_exact_inverse():
    x = rng.integers(0, 2, 128).astype(np.uint8)
    y = block_interleave(x, 8, 16)
    z = block_deinterleave(y, 8, 16, original_len=len(x))
    assert np.array_equal(z, x)


def test_conv_interleave_inverse_with_known_delay():
    N, D = 3, 2
    x = rng.integers(0, 2, 300).astype(np.uint8)
    y = conv_interleave(x, N, D)
    z = conv_deinterleave(y, N, D)
    d = conv_delay(N, D)
    assert np.array_equal(z[d:], x[: len(x) - d])


def test_interleaving_spreads_burst_for_fec():
    """A burst error in the channel becomes correctable after de-interleaving."""
    code = ConvCode()
    msg = rng.integers(0, 2, 240).astype(np.uint8)
    cw = code.encode(msg)
    il = block_interleave(cw, 8, 16)
    il[50:60] ^= 1  # 10-bit burst in the channel
    de = block_deinterleave(il, 8, 16, original_len=len(cw))
    dec = viterbi_decode(code, de)
    assert np.array_equal(dec[: len(msg)], msg)


def test_find_sync_locates_preamble():
    sync = np.array([1, 0, 1, 1, 0, 0, 1, 0], np.uint8)
    payload = rng.integers(0, 2, 100).astype(np.uint8)
    stream = np.concatenate(
        [rng.integers(0, 2, 20).astype(np.uint8), sync, payload]
    )
    hits = find_sync(stream, sync, min_match=1.0)
    assert 20 in [h["position"] for h in hits]
