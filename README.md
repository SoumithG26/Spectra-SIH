# SPECTRA

**S**ignal **P**arameter **E**xtraction, **C**lassification, **T**ransformation & **R**ecovery **A**nalyzer

An AI-assisted analyst workbench for off-air signal recordings (`.wav` / raw `.iq` / SigMF). SPECTRA ingests a recording, surveys its spectrum, estimates its front-end parameters, classifies its modulation, demodulates it to symbols and bits, and helps locate frame structure — every automated estimate carried with a **confidence**, never asserted as ground truth.

> Prototype for **Smart India Hackathon 2026 · Problem Statement 26147** (NTRO) — *"Automated model for analysis of .IQ and .wav files along with signal parameter extraction."* Category: Software · Theme: Space Technology.

---

## The problem

Given a raw signal capture, an analyst must recover, with no reliable side information:

1. **Signal parameters** — sampling frequency, modulation, FEC scheme, interleaving.
2. **Demodulation** — FSK / PSK / QAM back to a symbol stream.
3. **De-interleaving** — block / convolutional / diagonal / pseudo-random.
4. **FEC decoding** — convolutional (Viterbi), Reed–Solomon, concatenated, LDPC.
5. **Bitstream correlation** — locate sync words, separate header from payload.

The honest engineering reality is that this is **underdetermined** from samples alone. SPECTRA is therefore built around a *feasibility triage*, not a promise of one-click decode.

## Approach — confidence-scored feasibility triage

Every capability is placed in one of three tiers, and the UI reports confidence rather than a bare answer:

| Tier | Meaning | Examples in SPECTRA |
|------|---------|---------------------|
| **1 — Reliable** | Solvable from samples with good accuracy | sample-rate/BW/SNR survey, symbol-rate & CFO estimation, modulation family/order, PSK/QAM/FSK demod |
| **2 — Assisted** | Solvable *with* a confidence score or a human-in-the-loop hint | exact modulation order at low SNR, FEC/interleaver decode **given** the scheme, sync-word search |
| **3 — Underdetermined** | Not blindly recoverable without metadata or a codebook | blind LDPC/RS parameters, unknown scrambler/whitening, arbitrary proprietary framing |

This is the core differentiator: SPECTRA behaves like an **analyst's assistant that shows its work**, not an oracle that over-claims.

## What works today

- **Ingest** — WAV (stereo→IQ, mono→analytic via Hilbert), raw IQ (int8/int16/float32/complex64/…), and SigMF (`.sigmf-meta` + `.sigmf-data`).
- **Spectral survey** — Welch PSD, time-frequency waterfall, occupied bandwidth, SNR, band center.
- **Front-end estimation** — symbol rate (cyclostationary line in |x|²) and CFO (M-th-power spectral line), each with a confidence.
- **Modulation classification** — feature-based AMC across BPSK / QPSK / 8PSK / 16QAM / 64QAM / 2FSK / 4FSK, returning a ranked list.
- **Demodulation** — CFO correction → resample → RRC matched filter → timing recovery → carrier recovery → slicing; reports EVM and a constellation. FSK via instantaneous-frequency clustering.
- **Coding toolkit** — convolutional encode + **Viterbi** decode (K=7, 171/133), Hamming(7,4), block & convolutional (de)interleavers, bitstream cross-correlation / sync search / periodicity.
- **Synthetic digital twin** — a labeled signal generator (message → FEC → interleave → modulate → CFO → phase noise → AWGN) that doubles as the test oracle and demo data.
- **Two front-ends** — a Streamlit GUI and a CLI, both thin layers over one analysis pipeline.

## Architecture

One pipeline, two front-ends. The GUI and CLI both call `spectra.pipeline.analyze()`, so there is a single code path and a single place where decisions are scored.

```
              ┌─────────────────────────────────────────────────────────┐
  .wav / .iq  │  INGEST → SURVEY → FRONT-END → CLASSIFY → DEMOD → CORREL. │  report +
  / SigMF ───▶│  loader   PSD/SNR   sym-rate    AMC       symbols  sync/  │─▶ JSON +
              │           waterfall  CFO                   bits    period │   plots
              └─────────────────────────────────────────────────────────┘
                         every estimate carries a confidence
```

### Repository layout

```
spectra/
  core.py            Signal dataclass — the universal complex-baseband representation
  pipeline.py        analyze() + format_report() — the orchestrator
  cli.py             `python -m spectra.cli` (info / analyze / generate / suite)
  io/loader.py       WAV / raw-IQ / SigMF load + save
  dsp/
    spectral.py      Welch PSD, waterfall
    params.py        occupied BW, SNR, CFO, symbol-rate estimation (+confidence)
    modem.py         constellations, Gray maps, RRC — generator↔demod single source of truth
    modulation.py    feature-based modulation classifier
    demod.py         PSK/QAM/FSK demodulation chain
  coding/
    fec.py           ConvCode + Viterbi, Hamming(7,4)
    interleave.py    block & convolutional (de)interleavers
    correlate.py     xcorr, sync search, framing, periodicity
  synth/generator.py SignalSpec + generate() + default_suite() — the labeled digital twin
app.py               Streamlit GUI
tests/               pytest suite (io / dsp / coding / pipeline)
```

## Install

Requires **Python 3.10+** (developed on 3.13). From the repo root:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -e .            # installs spectra + the `spectra` CLI
# or, without packaging:
pip install -r requirements.txt
```

## Quickstart — GUI

```bash
streamlit run app.py
```

Then, in the browser:

1. Pick a **synthetic preset** (e.g. `qpsk_cfo`) from the sidebar, or **upload** a `.wav` / raw `.iq` / SigMF pair.
2. Read the survey metrics, PSD and waterfall on the **Spectrum** tab.
3. See the parameter estimates, modulation ranking and constellation on **Modulation & demod**.
4. Inspect recovered bits and run a sync-word / periodicity search on **Bitstream**.
5. Export the text report or full JSON on **Full report**.

For synthetic inputs the true label is shown so estimates can be checked against ground truth on the spot.

## Quickstart — CLI

```bash
# list the built-in labeled demo suite
python -m spectra.cli suite

# synthesize a labeled recording (SigMF by default; also iq / wav)
python -m spectra.cli generate samples/demo --preset qpsk_cfo --format sigmf

# analyze any recording; --json writes a machine-readable result
python -m spectra.cli analyze samples/demo.sigmf-meta --json out.json

# raw IQ must be told its sample rate (it is not stored in the bytes)
python -m spectra.cli analyze capture.iq --sample-rate 2000000 --dtype complex64

# quick header/summary
python -m spectra.cli info recording.wav
```

## Python API

```python
from spectra.io import loader
from spectra.pipeline import analyze, format_report

sig = loader.load("capture.sigmf-meta")        # or load_wav / load_iq
results = analyze(sig)                          # full pipeline, all estimates scored
print(format_report(results))
print(results["parameters"]["symbol_rate_est_hz"],
      results["modulation"]["prediction"],
      results["modulation"]["confidence"])
```

Generate labeled test signals directly:

```python
from spectra.synth import SignalSpec, generate

sig, label = generate(SignalSpec("16QAM", symbol_rate=9600, snr_db=28,
                                 cfo_hz=250, fec="conv",
                                 interleaver=("block", 8, 16), seed=1))
```

## Testing

```bash
python -m pytest -q
```

The suite (`tests/`) covers IO round-trips, symbol-rate/CFO estimation accuracy, modulation-family classification across the whole demo suite, demod EVM bounds, and Viterbi/interleaver correctness — all validated against the synthetic generator's ground-truth labels.

## Roadmap (Tier 2/3, not yet implemented)

- **FEC:** Reed–Solomon and LDPC decoders; concatenated (RS + conv) chains; soft-decision Viterbi.
- **Interleaving:** diagonal and pseudo-random de-interleavers; **blind** interleaver-parameter recovery.
- **Blind coding recovery:** rank/rate detection, a hypothesis-and-validate engine for unknown FEC/interleaver.
- **Front-end:** full cyclostationary (spectral-correlation) analysis; higher-order cumulants for AMC.
- **ML classifier:** a learned AMC model trained on the synthetic corpus, benchmarked against the feature-based one.
- **Scrambler/whitening** detection and descrambling.

## Design notes

- **Pure NumPy/SciPy** DSP (no GNU Radio dependency) for a reliable, portable prototype.
- `dsp/modem.py` is the **single source of truth** for constellations, Gray maps and RRC pulses, so the generator and demodulator can never drift apart.
- The synthetic generator is deliberately the same code the tests score against — the "digital twin" is both the demo data and the correctness oracle.
- Estimates are **confidence-scored by construction**; the workbench is honest about Tier-3 limits rather than hiding them.

