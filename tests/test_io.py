"""Round-trip tests for the WAV / IQ / SigMF loaders."""

import numpy as np
from scipy.io import wavfile

from spectra.io import loader
from spectra.synth import SignalSpec, generate


def test_iq_roundtrip(tmp_path):
    sig, _ = generate(SignalSpec("QPSK", 2400, 8, snr_db=40, n_bits=512, seed=3))
    p = tmp_path / "x.iq"
    loader.save_iq(p, sig, "complex64")
    sig2 = loader.load_iq(p, sample_rate=sig.sample_rate, dtype="complex64")
    assert len(sig2) == len(sig)
    assert np.allclose(sig2.samples, sig.samples, atol=1e-4)


def test_sigmf_roundtrip(tmp_path):
    sig, _ = generate(SignalSpec("BPSK", 1200, 8, snr_db=40, n_bits=256, seed=4))
    meta, _data = loader.save_sigmf(tmp_path / "rec", sig)
    sig2 = loader.load_sigmf(meta)
    assert abs(sig2.sample_rate - sig.sample_rate) < 1e-6
    assert len(sig2) == len(sig)


def test_wav_stereo_is_iq(tmp_path):
    sig, _ = generate(SignalSpec("QPSK", 2400, 8, snr_db=40, n_bits=256, seed=5))
    data = np.column_stack([sig.samples.real, sig.samples.imag]).astype(np.float32)
    p = tmp_path / "y.wav"
    wavfile.write(p, int(sig.sample_rate), data)
    sig2 = loader.load_wav(p)
    assert sig2.meta["wav_interpretation"] == "iq_stereo"
    assert len(sig2) == len(sig)


def test_wav_mono_is_analytic(tmp_path):
    fs = 8000
    t = np.arange(fs) / fs
    tone = np.cos(2 * np.pi * 1000 * t).astype(np.float32)
    p = tmp_path / "m.wav"
    wavfile.write(p, fs, tone)
    sig = loader.load_wav(p)
    assert sig.meta["wav_interpretation"] == "real_analytic"
    assert len(sig) == fs
