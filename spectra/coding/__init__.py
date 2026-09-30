"""Bit-domain processing: interleaving, FEC, and bitstream correlation."""

from __future__ import annotations

from .correlate import find_sync, frame_segments, xcorr_bits
from .fec import (
    ConvCode,
    hamming74_decode,
    hamming74_encode,
    viterbi_decode,
)
from .interleave import (
    block_deinterleave,
    block_interleave,
    conv_deinterleave,
    conv_interleave,
)

__all__ = [
    "ConvCode",
    "viterbi_decode",
    "hamming74_encode",
    "hamming74_decode",
    "block_interleave",
    "block_deinterleave",
    "conv_interleave",
    "conv_deinterleave",
    "xcorr_bits",
    "find_sync",
    "frame_segments",
]
