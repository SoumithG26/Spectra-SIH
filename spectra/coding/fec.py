"""Forward error correction: convolutional codes with Viterbi, and Hamming(7,4).

The convolutional encoder / hard-decision Viterbi decoder cover the problem
statement's "short-constraint convolution codes with Viterbi decoding" and are
the flagship FEC path. Reed-Solomon / LDPC / concatenated codes are on the
roadmap (see README); the interfaces here are written so those decoders can slot
into the same hypothesis-and-validate engine.
"""

from __future__ import annotations

import numpy as np


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


class ConvCode:
    """Rate-1/n feed-forward convolutional code.

    Parameters
    ----------
    gens:
        Generator polynomials in octal (default = industry-standard K=7,
        rate-1/2 code with G = [171, 133] octal, as used by many HF/satellite
        links).
    K:
        Constraint length (register length including the current bit).
    """

    def __init__(self, gens: tuple[int, ...] = (0o171, 0o133), K: int = 7) -> None:
        self.gens = tuple(gens)
        self.K = int(K)
        self.n = len(self.gens)
        self.mask = (1 << (self.K - 1)) - 1
        self.n_states = 1 << (self.K - 1)
        self.rate = (1, self.n)
        # Precompute the trellis: trans[state][input] = (next_state, out_bits).
        self.trans: list[list[tuple[int, np.ndarray]]] = []
        for s in range(self.n_states):
            row = []
            for b in (0, 1):
                reg = (b << (self.K - 1)) | s
                out = np.array([_parity(reg & g) for g in self.gens], dtype=np.uint8)
                ns = (reg >> 1) & self.mask
                row.append((ns, out))
            self.trans.append(row)

    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Encode a message. If ``terminate``, append K-1 zero tail bits so the
        trellis returns to the all-zero state (improves decoder tail accuracy)."""
        seq = [int(b) & 1 for b in np.asarray(bits).ravel()]
        if terminate:
            seq = seq + [0] * (self.K - 1)
        state = 0
        out = np.empty(len(seq) * self.n, dtype=np.uint8)
        for k, b in enumerate(seq):
            reg = (b << (self.K - 1)) | state
            for j, g in enumerate(self.gens):
                out[k * self.n + j] = _parity(reg & g)
            state = (reg >> 1) & self.mask
        return out


def viterbi_decode(code: ConvCode, coded: np.ndarray,
                   terminated: bool = True) -> np.ndarray:
    """Hard-decision Viterbi decoder.

    Returns the recovered message bits (tail removed when ``terminated``).
    """
    coded = np.asarray(coded, dtype=np.uint8).ravel()
    n = code.n
    steps = len(coded) // n
    if steps == 0:
        return np.zeros(0, dtype=np.uint8)
    recv = coded[: steps * n].reshape(steps, n)

    INF = 1 << 30
    pm = np.full(code.n_states, INF, dtype=np.int64)
    pm[0] = 0
    prev_state = np.full((steps, code.n_states), -1, dtype=np.int32)
    prev_input = np.zeros((steps, code.n_states), dtype=np.uint8)

    for t in range(steps):
        r = recv[t]
        newpm = np.full(code.n_states, INF, dtype=np.int64)
        for s in range(code.n_states):
            base = pm[s]
            if base >= INF:
                continue
            for b in (0, 1):
                ns, out = code.trans[s][b]
                bm = int(np.count_nonzero(out != r))
                cand = base + bm
                if cand < newpm[ns]:
                    newpm[ns] = cand
                    prev_state[t, ns] = s
                    prev_input[t, ns] = b
        pm = newpm

    final = 0 if terminated else int(np.argmin(pm))
    decoded = np.empty(steps, dtype=np.uint8)
    s = final
    for t in range(steps - 1, -1, -1):
        decoded[t] = prev_input[t, s]
        s = int(prev_state[t, s])
    if terminated:
        decoded = decoded[: steps - (code.K - 1)]
    return decoded


# --- Hamming(7,4) ----------------------------------------------------------

# Systematic generator G = [I4 | P]; parity-check H = [P^T | I3].
_P = np.array([[1, 1, 0],
               [0, 1, 1],
               [1, 1, 1],
               [1, 0, 1]], dtype=np.uint8)
_H = np.hstack([_P.T, np.eye(3, dtype=np.uint8)])  # 3 x 7
# Syndrome (as a column of H) -> error position.
_SYNDROME_TO_POS = {tuple(_H[:, p]): p for p in range(7)}


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """Encode data in blocks of 4 -> 7 (single-error-correcting)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    data = bits.reshape(-1, 4)
    parity = (data @ _P) % 2
    return np.hstack([data, parity]).reshape(-1).astype(np.uint8)


def hamming74_decode(bits: np.ndarray, data_len: int | None = None) -> np.ndarray:
    """Decode blocks of 7 -> 4, correcting a single bit error per block."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = (len(bits) // 7) * 7
    words = bits[:n].reshape(-1, 7).copy()
    for w in words:
        syn = tuple((_H @ w) % 2)
        if any(syn):
            pos = _SYNDROME_TO_POS.get(syn)
            if pos is not None:
                w[pos] ^= 1
    out = words[:, :4].reshape(-1).astype(np.uint8)
    if data_len is not None:
        out = out[:data_len]
    return out
