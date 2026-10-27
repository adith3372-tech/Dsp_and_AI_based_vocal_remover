# Classical DSP Vocal Remover & Karaoke Track Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Flask Web UI](https://img.shields.io/badge/Web%20UI-Flask-brightgreen.svg)](app.py)
[![Tests: 31 Passed](https://img.shields.io/badge/tests-31%20passed-success.svg)](test_pipeline.py)
[![DSP: Pure Python](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-orange.svg)](main.py)

A complete, software-based **Digital Signal Processing (DSP) pipeline** and **interactive Web Application** designed to isolate vocal content and produce high-fidelity karaoke backing tracks from stereo commercial music mixes.

Developed as the **S5 DSP Course Project (PBECT504)** at **SCT College of Engineering, Thiruvananthapuram (APJ Abdul Kalam Technological University)**.

---

## Table of Contents

- [Overview](#overview)
- [DSP Pipeline Architecture](#dsp-pipeline-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Web Interface (Recommended)](#1-web-interface-recommended)
  - [2. Command-Line Interface (CLI)](#2-command-line-interface-cli)
- [Tuning Presets & Guidelines](#tuning-presets--guidelines)
- [Testing](#testing)
- [Deep Learning Engine (Optional)](#deep-learning-engine-optional)
- [Team & Acknowledgments](#team--acknowledgments)
- [License](#license)

---

## Overview

Traditional vocal cancellation techniques relying solely on simple center-channel subtraction ($\text{Left} - \text{Right}$) suffer from severe side-effects: bass drums and central basslines are lost, reverberant vocal tails remain audible, and comb filtering colors the instrumental sound.

This project implements a multi-stage classical DSP pipeline combining:
1. **Mid/Side Sub-band Decomposition** preserving bass (<80 Hz) and treble (>4 kHz).
2. **Polyphase Multirate Decimation ($M=2$)** reducing processing complexity by 50%.
3. **STFT Wiener Soft Spectral Masking** driven by inter-channel phase coherence.
4. **256-tap Normalized Least Mean Squares (NLMS)** adaptive cancellation.
5. **iSTFT Overlap-Add (OLA) Synthesis** with DC-offset removal and transparent soft limiting.
6. **Side-by-side Demucs v4 AI Mode** for objective comparison against deep learning source separation.

---

## DSP Pipeline Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Input: stereo_mix.wav               │
                  └──────────────────────┬───────────────────────┘
                                         │
                   [Stage 1: Preprocessing & Filter Banks]
                                         │
        ┌────────────────────────────────┼──────────────────────────────┐
        │ (<80 Hz)                       │ (80 Hz - 4 kHz)              │ (>4 kHz)
        ▼                                ▼                              ▼
  Bass Sub-band                    Vocal Sub-band                 Treble Sub-band
(8th-order Butterworth SOS)      (1001-tap Kaiser FIR)          (8th-order Butterworth SOS)
        │                                │                              │
        │                  [Stage 2: Multirate Decimation]              │
        │                                │                              │
        │                  Polyphase Decimation (M=2)                   │
        │                                │                              │
        │                  [Stage 3: STFT & Spectral Masking]           │
        │                                │                              │
        │                  Mid/Side Phase Coherence Mask                │
        │                  + Wiener Soft Spectral Mask                  │
        │                                │                              │
        │                  [Stage 4: Adaptive Filtering]                │
        │                                │                              │
        │                  256-Tap Block-NLMS Filter                    │
        │                  (Residual Vocal Cancellation)                │
        │                                │                              │
        │                  [Stage 5: Synthesis & Recombination]         │
        │                                │                              │
        │                  Polyphase Interpolation (L=2)                │
        │                  + iSTFT Overlap-Add (OLA)                    │
        │                                │                              │
        └───────────────────────────────►◄──────────────────────────────┘
                                         │
                         DC Removal & Peak Normalization
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │  Outputs:                                    │
                  │   1. karaoke.wav (Instrumental)              │
                  │   2. vocals.wav  (Isolated vocals)           │
                  │   3. pipeline_results.png (6-panel analysis) │
                  └──────────────────────────────────────────────┘
```

---

## Project Structure

```
vocal_remover/
├── app.py                 # Flask web server & asynchronous job worker
├── main.py                # Top-level CLI pipeline orchestrator
├── preprocessing.py       # Stage 1: IIR Butterworth SOS & Kaiser FIR filter banks
├── multirate.py           # Stage 2: Polyphase decimation and interpolation
├── spectral.py            # Stage 3: STFT, Mid/Side decomposition & Wiener masking
├── adaptive.py            # Stage 4: 256-tap sample and block NLMS adaptive filter
├── synthesis.py           # Stage 5: iSTFT OLA, band recombination, soft limiter
├── evaluation.py          # Stage 6: Blind SDR, SIR estimation & 6-panel plotting
├── ai_separator.py        # Optional Meta Demucs v4 deep learning integration
├── test_pipeline.py       # Complete pytest test suite (31 unit tests)
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore file excluding temporary outputs and audio
├── LICENSE                # MIT License
├── templates/
│   └── index.html         # Responsive web dashboard
├── static/
│   ├── css/
│   │   └── style.css      # Custom dark-theme styling
│   ├── js/
│   │   └── app.js         # Async job polling, audio players & waveform UI
│   └── outputs/           # Dynamic processing outputs (.gitkeep)
└── uploads/               # Temporary song uploads (.gitkeep)
```

---

## Getting Started

### Prerequisites
- Python **3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- Windows, macOS, or Linux

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/vocal-remover.git
   cd vocal-remover
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Web Interface (Recommended)

Start the local Flask application:
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

- Drag and drop any stereo audio file (`.wav`, `.mp3`, `.flac`, `.ogg`).
- Select parameter presets (Ballad, Pop, Rock, EDM) or fine-tune sliders:
  - **Decimation Factor ($M$)**: `2` or `4`
  - **NLMS Step Size ($\mu$)**: `0.05` to `0.80`
  - **Mask Blend Weight**: `0.0` to `1.0`
- Preview and download isolated **Karaoke** and **Vocal** tracks with the interactive dual audio player.
- Inspect the generated **6-panel DSP diagnostic plot** showing convergence curves, filter magnitude responses, and spectrograms.

---

### 2. Command-Line Interface (CLI)

Run separation directly from the terminal:

```bash
# Basic run with default parameters
python main.py --input song.wav

# Custom parameters
python main.py --input song.wav --decimation 2 --nlms_mu 0.35 --stft_blend 0.65 --nlms_blend 0.35

# Generate a synthetic stereo test demo (no audio file required)
python main.py --demo
```

#### CLI Options:
| Flag | Default | Description |
|---|---|---|
| `--input` | None | Path to input stereo audio file (`.wav`, `.mp3`, `.flac`) |
| `--output_dir` | `./output` | Output directory for separated stems and plots |
| `--decimation` | `2` | Polyphase decimation factor $M \in \{2, 4\}$ |
| `--n_fft` | `2048` | STFT analysis window size |
| `--hop_length` | `512` | STFT hop size (75% overlap) |
| `--nlms_order` | `256` | FIR filter tap length for NLMS |
| `--nlms_mu` | `0.3` | NLMS step size $\mu$ |
| `--stft_blend` | `0.6` | STFT Wiener estimate blend weight |
| `--nlms_blend` | `0.4` | NLMS adaptive estimate blend weight |
| `--plot` | `True` | Generate 6-panel evaluation plot PNG |
| `--demo` | `False` | Run self-contained synthetic benchmark demo |

---

## Tuning Presets & Guidelines

| Genre / Track Type | Decimation ($M$) | NLMS Step ($\mu$) | Mask Blend | Rationale |
|---|:---:|:---:|:---:|---|
| **Acoustic / Ballad** | 2 | `0.30 - 0.35` | `0.60 - 0.65` | Smooth vocal transitions, piano/acoustic reverb preservation |
| **Standard Pop** | 2 | `0.35 - 0.40` | `0.70` | Balanced vocal cancellation with stable stereo drum imaging |
| **Rock / High Tempo** | 2 | `0.40 - 0.50` | `0.75` | Fast tracking for dynamic vocal bursts and heavy guitars |
| **EDM / Dense Synth** | 2 | `0.20 - 0.30` | `0.55 - 0.60` | Lower step size prevents modulation distortion on wide synths |

---

## Testing

Run the automated test suite covering all DSP stages:
```bash
python -m pytest test_pipeline.py -v
```

Tests include:
- FIR bandpass symmetry, linear phase, and stopband attenuation
- IIR Butterworth cutoff attenuation and stability
- Polyphase multirate round-trip reconstruction energy
- STFT/iSTFT perfect reconstruction within machine precision
- NLMS filter convergence and stability on transients
- Normalization, soft limiting, and DC offset removal

---

## Deep Learning Engine (Optional)

The project also includes an optional comparison mode powered by **Meta's Demucs v4 (`htdemucs`)**:
```bash
# Install PyTorch and Demucs
pip install torch torchaudio demucs
```
When Demucs is installed, the web interface automatically enables the **AI Mode** toggle, allowing users to benchmark classical DSP results against deep neural source separation.

---

## Team & Acknowledgments

### Project Team (Batch 2024–2028, ECE)
- **Adith R** (SCT24EC016)
- **Gautham S Nair** (SCT24EC052)
- **Abhishek S Shenoy** (SCT24EC013)
- **Adithya Siva** (CHN24EC011)

### Faculty Mentors
- **Course Guide:** Dr. Nelwin Raj N. R., Associate Professor, Dept. of ECE
- **Head of Department:** Prof. Sajith Sethu P., Dept. of ECE

**Institution:** SCT College of Engineering, Pappanamcode, Thiruvananthapuram, Kerala.  
*Affiliated to APJ Abdul Kalam Technological University (KTU).*

---

## License

This project is open-source under the [MIT License](LICENSE).
