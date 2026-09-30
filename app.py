"""SPECTRA - Streamlit analyst workbench (GUI).

Run:  streamlit run app.py

A thin visual layer over the ``spectra`` package. Load a recording
(.wav / raw .iq / SigMF) or synthesize a labeled one, then run the analysis
pipeline and inspect the results: spectral survey (PSD + waterfall), estimated
parameters with confidence, modulation classification, constellation / demod,
and bit-structure correlation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from spectra import __version__
from spectra.coding.correlate import estimate_period, find_sync, xcorr_bits
from spectra.core import Signal
from spectra.dsp.modem import SUPPORTED_LINEAR, constellation
from spectra.dsp.spectral import waterfall, welch_psd
from spectra.io import loader
from spectra.pipeline import analyze, format_report
from spectra.synth import SignalSpec, default_suite, generate

st.set_page_config(page_title="SPECTRA workbench", layout="wide")

_PLOT_STYLE = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#dddddd",
    "axes.titlecolor": "#ffffff",
    "text.color": "#dddddd",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "grid.color": "#333333",
    "figure.autolayout": True,
}
plt.rcParams.update(_PLOT_STYLE)

# --------------------------------------------------------------------------- #
# Signal acquisition
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def synth_signal(preset, mod, sr, sps, snr, cfo, nbits, fec, il, seed):
    """Generate a labeled synthetic recording (cached on its parameters)."""
    if preset:
        spec = default_suite()[preset]
    else:
        spec = SignalSpec(modulation=mod, symbol_rate=sr, sps=sps, snr_db=snr,
                          cfo_hz=cfo, n_bits=nbits, fec=fec, interleaver=il,
                          seed=seed)
    sig, label = generate(spec)
    return sig, label, spec


def load_upload(files, sample_rate, dtype):
    """Reconstruct a :class:`Signal` from uploaded file(s) in a temp dir.

    Handles a SigMF pair (.sigmf-meta + .sigmf-data), a WAV, or raw IQ. The
    files are written side by side so the SigMF loader can find its data file.
    """
    tmp = Path(tempfile.mkdtemp(prefix="spectra_"))
    saved = {}
    for f in files:
        p = tmp / f.name
        p.write_bytes(f.getbuffer())
        saved[f.name.lower()] = p

    meta = next((p for n, p in saved.items() if n.endswith(".sigmf-meta")), None)
    wav = next((p for n, p in saved.items() if n.endswith(".wav")), None)
    raw = next((p for n, p in saved.items()
                if n.rsplit(".", 1)[-1] in {"iq", "dat", "bin", "cfile", "fc32"}), None)
    if meta is not None:
        return loader.load_sigmf(meta)
    if wav is not None:
        return loader.load_wav(wav)
    target = raw if raw is not None else next(iter(saved.values()))
    if not sample_rate:
        raise ValueError("Raw IQ has no embedded sample rate - set one in the sidebar.")
    return loader.load_iq(target, sample_rate=float(sample_rate), dtype=dtype)


@st.cache_data(show_spinner=False)
def run_pipeline(sample_bytes, sample_rate, source, demod, known_mod, known_sr):
    """Cached wrapper around :func:`spectra.pipeline.analyze` keyed on raw IQ."""
    samples = np.frombuffer(sample_bytes, dtype=np.complex64)
    sig = Signal(samples.copy(), sample_rate, source=source)
    return analyze(sig, demod=demod,
                   known_modulation=known_mod or None,
                   known_symbol_rate=known_sr or None)

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _fig(w=6.5, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_psd(sig):
    f, psd = welch_psd(sig, nperseg=min(2048, max(64, len(sig))))
    fig, ax = _fig()
    ax.plot(f / 1e3, psd, color="#4ea3ff", lw=1.0)
    ax.set_xlabel("baseband frequency (kHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Power spectral density (Welch)")
    return fig


def plot_waterfall(sig):
    f, t, p = waterfall(sig, nperseg=min(256, max(32, len(sig) // 16)))
    fig, ax = _fig(6.5, 3.2)
    im = ax.pcolormesh(t * 1e3, f / 1e3, p, shading="auto", cmap="magma")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title("Waterfall (time-frequency)")
    fig.colorbar(im, ax=ax, label="dB")
    return fig


def plot_constellation(symbols, mod):
    fig, ax = _fig(4.4, 4.4)
    s = np.asarray(symbols)
    if np.iscomplexobj(s) and s.size:
        ax.scatter(s.real, s.imag, s=7, alpha=0.35, color="#4ea3ff",
                   edgecolors="none")
        if mod.upper() in SUPPORTED_LINEAR:
            pts, _, _ = constellation(mod.upper())
            ax.scatter(pts.real, pts.imag, s=90, marker="x", color="#ff6b6b",
                       linewidths=2, label="ideal")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
        lim = max(1.5, float(np.percentile(np.abs(s), 99)) * 1.15)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"Constellation - {mod}")
    return fig

def plot_fsk(dm):
    fig, ax = _fig(4.8, 3.2)
    means = np.asarray(dm.get("symbols", []), dtype=float)
    if means.size:
        ax.hist(means / 1e3, bins=60, color="#4ea3ff", alpha=0.85)
    for c in dm.get("tone_freqs", []) or []:
        ax.axvline(float(c) / 1e3, color="#ff6b6b", ls="--", lw=1.2)
    ax.set_xlabel("instantaneous frequency (kHz)")
    ax.set_ylabel("symbol count")
    ax.set_title("FSK tone histogram (dashed = tone centers)")
    return fig


def plot_ranking(ranking):
    fig, ax = _fig(4.6, 3.2)
    names = [k for k, _ in ranking][:6][::-1]
    scores = [v for _, v in ranking][:6][::-1]
    colors = ["#4ea3ff" if i == len(names) - 1 else "#3a6ea5"
              for i in range(len(names))]
    ax.barh(names, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("Modulation ranking")
    return fig


def bits_to_grid(bits, width=64, rows=16):
    b = np.asarray(bits, dtype=np.uint8).ravel()[: width * rows]
    if b.size == 0:
        return "(no bits recovered)"
    lines = []
    for i in range(0, b.size, width):
        chunk = "".join(str(int(x)) for x in b[i:i + width])
        lines.append(f"{i:6d} | {chunk}")
    return "\n".join(lines)


def _parse_bits(text):
    cleaned = "".join(c for c in text if c in "01")
    if not cleaned:
        return None
    return np.array([int(c) for c in cleaned], dtype=np.uint8)

# --------------------------------------------------------------------------- #
# Sidebar - source + analysis options
# --------------------------------------------------------------------------- #
st.title("SPECTRA")
st.caption(
    "Signal Parameter Extraction, Classification, Transformation & Recovery "
    f"Analyzer  ·  v{__version__}  ·  SIH 2026 / PS 26147"
)

MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK"]
sig = None
label = None
err = None

with st.sidebar:
    st.header("1 · Signal source")
    mode = st.radio("Input", ["Synthetic demo", "Upload recording"], index=0)

    if mode == "Synthetic demo":
        use_preset = st.toggle("Use a suite preset", value=True)
        preset = mod = None
        sr = sps = snr = cfo = nbits = seed = 0
        fec, il = "none", None
        if use_preset:
            preset = st.selectbox("Preset", list(default_suite().keys()), index=1)
        else:
            mod = st.selectbox("Modulation", MODS, index=1)
            sr = st.number_input("Symbol rate (baud)", 300, 200_000, 4800, step=300)
            sps = st.slider("Samples/symbol", 4, 16, 8)
            snr = st.slider("SNR (dB)", -5, 40, 20)
            cfo = st.number_input("CFO (Hz)", -5000, 5000, 0, step=50)
            nbits = st.number_input("Message bits", 256, 20_000, 2048, step=256)
            fec = st.selectbox("FEC", ["none", "conv", "hamming"])
            il_kind = st.selectbox("Interleaver", ["none", "block", "conv"])
            if il_kind == "block":
                il = ("block", st.number_input("rows", 2, 64, 8),
                      st.number_input("cols", 2, 64, 16))
            elif il_kind == "conv":
                il = ("conv", st.number_input("branches", 2, 16, 4),
                      st.number_input("depth", 1, 16, 2))
            seed = st.number_input("Seed", 0, 9999, 0)
        try:
            sig, label, _spec = synth_signal(preset, mod, float(sr), int(sps),
                                             float(snr), float(cfo), int(nbits),
                                             fec, il, int(seed))
        except Exception as exc:  # noqa: BLE001
            err = f"generation failed: {exc}"

    else:  # Upload recording
        files = st.file_uploader(
            "Recording file(s)",
            type=["wav", "iq", "dat", "bin", "cfile", "fc32", "sigmf-meta",
                  "sigmf-data"],
            accept_multiple_files=True,
            help="WAV, raw IQ, or a SigMF pair (upload both .sigmf-meta and "
                 ".sigmf-data together).",
        )
        st.caption("Raw IQ needs the two fields below; WAV/SigMF ignore them.")
        up_sr = st.number_input("Sample rate (Hz)", 0, 100_000_000, 0, step=1000)
        up_dtype = st.selectbox(
            "Raw IQ dtype",
            ["complex64", "int8", "int16", "float32", "uint8", "complex128"],
        )
        if files:
            try:
                sig = load_upload(files, up_sr, up_dtype)
            except Exception as exc:  # noqa: BLE001
                err = f"load failed: {exc}"

    st.header("2 · Analysis options")
    do_demod = st.toggle("Run demodulation", value=True)
    known_mod = st.selectbox("Force modulation", ["auto", *MODS], index=0)
    known_sr = st.number_input("Force symbol rate (Hz, 0=auto)", 0, 500_000, 0,
                               step=100)
    known_mod_val = None if known_mod == "auto" else known_mod

# --------------------------------------------------------------------------- #
# Main - guard clauses
# --------------------------------------------------------------------------- #
if err:
    st.error(err)
    st.stop()
if sig is None:
    st.info("Choose a synthetic preset or upload a recording from the sidebar "
            "to begin.")
    st.stop()

with st.spinner("Analyzing..."):
    results = run_pipeline(sig.samples.tobytes(), sig.sample_rate, sig.source,
                           do_demod, known_mod_val, float(known_sr))

# --------------------------------------------------------------------------- #
# Main - header metrics
# --------------------------------------------------------------------------- #
s = results["signal"]
sv = results["survey"]
p = results["parameters"]
m = results["modulation"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples", f"{s['samples']:,}")
c2.metric("Sample rate", f"{s['sample_rate_hz']/1e3:.1f} kHz")
c3.metric("Duration", f"{s['duration_s']*1e3:.1f} ms")
c4.metric("Occupied BW", f"{sv['occupied_bandwidth_hz']/1e3:.2f} kHz")
c5.metric("Est. SNR", f"{sv['snr_db']:.1f} dB")

if label is not None:
    with st.expander("Ground truth (synthetic label)", expanded=False):
        truth = {k: v for k, v in label.items()
                 if not isinstance(v, np.ndarray)}
        gc = st.columns(4)
        gc[0].metric("True modulation", truth.get("modulation", "?"),
                     help="what the generator produced")
        gc[1].metric("True symbol rate", f"{truth.get('symbol_rate', 0):.0f} Hz")
        gc[2].metric("True CFO", f"{truth.get('cfo_hz', 0):.0f} Hz")
        gc[3].metric("True SNR", f"{truth.get('snr_db', 0):.0f} dB")
        st.json(truth, expanded=False)

tab_spec, tab_mod, tab_bits, tab_report = st.tabs(
    ["Spectrum", "Modulation & demod", "Bitstream", "Full report"]
)

# --- Spectrum tab ---------------------------------------------------------- #
with tab_spec:
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_psd(sig), use_container_width=True)
    with right:
        st.pyplot(plot_waterfall(sig), use_container_width=True)
    st.markdown(
        f"**Occupied bandwidth** {sv['occupied_bandwidth_hz']:.1f} Hz around "
        f"**{sv['band_center_hz']:.1f} Hz**  ·  **noise floor** "
        f"{sv['noise_floor']:.1f} dB  ·  **est. SNR** {sv['snr_db']:.1f} dB"
    )

# --- Modulation & demod tab ------------------------------------------------ #
with tab_mod:
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("CFO estimate", f"{p['cfo_hz']:.1f} Hz",
               help=f"confidence {p['cfo_confidence']}")
    pc2.metric("Symbol rate estimate", f"{p['symbol_rate_est_hz']:.0f} Hz",
               help=f"confidence {p['symbol_rate_confidence']}  ·  "
                    f"~{p['sps_est']} sps")
    pc3.metric("Predicted modulation", m["prediction"],
               help=f"confidence {m['confidence']}  ·  family {m['family']}")

    mleft, mright = st.columns([1, 1])
    with mleft:
        st.pyplot(plot_ranking(m["ranking"]), use_container_width=True)
        st.progress(min(1.0, float(m["confidence"])),
                    text=f"classifier confidence {m['confidence']:.0%}")
    with mright:
        dm = results.get("_demod_raw")
        used = results.get("modulation_used", m["prediction"])
        if dm is None:
            st.info("Demodulation is turned off (enable it in the sidebar).")
        elif "FSK" in str(used).upper():
            st.pyplot(plot_fsk(dm), use_container_width=True)
        else:
            st.pyplot(plot_constellation(dm.get("symbols", []), used),
                      use_container_width=True)
        d = results.get("demod", {})
        if d.get("evm_percent") is not None:
            st.metric("EVM (demod quality)", f"{d['evm_percent']:.2f} %")

# --- Bitstream tab --------------------------------------------------------- #
with tab_bits:
    dm = results.get("_demod_raw") or {}
    bits = np.asarray(dm.get("bits", []), dtype=np.uint8).ravel()
    d = results.get("demod", {})
    st.subheader("Recovered bitstream")
    if bits.size == 0:
        st.warning("No bits recovered (demodulation off or failed).")
    else:
        bc1, bc2 = st.columns(2)
        bc1.metric("Symbols", f"{d.get('n_symbols', 0):,}")
        bc2.metric("Bits", f"{d.get('n_bits', bits.size):,}")
        st.code(bits_to_grid(bits), language="text")
        st.download_button("Download bits (.txt)",
                           "".join(str(int(b)) for b in bits),
                           file_name="spectra_bits.txt")

        st.divider()
        st.subheader("Bit-structure correlation")
        sc1, sc2 = st.columns(2)
        with sc1:
            patt = st.text_input("Sync / preamble pattern (binary)",
                                 value="10110001")
            min_match = st.slider("Min match", 0.5, 1.0, 0.9, 0.01)
            sync = _parse_bits(patt)
            if sync is not None:
                hits = find_sync(bits, sync, min_match=min_match)
                st.write(f"**{len(hits)}** hit(s) for a {sync.size}-bit pattern")
                if hits:
                    st.dataframe(hits[:50], use_container_width=True)
        with sc2:
            per = estimate_period(bits)
            st.metric("Detected period", f"{per['period']} bits",
                      help=f"autocorrelation confidence {per['confidence']:.2f}")
            if label is not None and isinstance(label.get("tx_symbol_bits"), np.ndarray):
                truth_bits = label["tx_symbol_bits"]
                xc = xcorr_bits(bits, truth_bits)
                st.metric("Match vs transmitted bits",
                          f"{abs(xc['score']):.1%}",
                          help="peak normalized correlation of recovered vs "
                               "true channel bits (1.0 = perfect, before FEC "
                               "decode); phase/rotation may invert polarity")

# --- Full report tab ------------------------------------------------------- #
with tab_report:
    st.code(format_report(results), language="text")

    import json

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)

    clean = {k: v for k, v in results.items() if k != "_demod_raw"}
    payload = json.dumps(clean, indent=2, default=_default)
    st.download_button("Download analysis JSON", payload,
                       file_name="spectra_analysis.json", mime="application/json")
    with st.expander("Raw results dict"):
        st.json(clean, expanded=False)

st.caption(
    "Estimates are confidence-scored, not asserted as ground truth. "
    "Tier-1 (parameter/AMC/demod) is automated; blind FEC/interleaver recovery "
    "and RS/LDPC decoding are on the roadmap - see the README."
)

