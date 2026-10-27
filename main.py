"""
=============================================================================
main.py - AI-Assisted Vocal Remover & Karaoke Track Generator
=============================================================================

University ECE Project - DSP Pipeline Orchestrator
----------------------------------------------------
This script is the top-level entry point that wires together every DSP stage:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  INPUT: stereo_mix.wav (44.1 kHz, 16/24-bit PCM)                    │
    │                                                                     │
    │  Stage 1: Pre-Processing                                            │
    │    └─► FIR band-pass (80 Hz – 4 kHz) + IIR bass & treble split     │
    │                                                                     │
    │  Stage 2: Multi-Rate DSP                                            │
    │    └─► Polyphase Decimation (÷D) -> analyse at reduced rate          │
    │                                                                     │
    │  Stage 3: STFT Analysis & Spectral Masking                          │
    │    └─► STFT -> Mid/Side -> Phase Coherence -> Wiener Soft Mask         │
    │                                                                     │
    │  Stage 4: Adaptive Filtering (NLMS)                                 │
    │    └─► Block-NLMS on vocal band -> residual cancellation             │
    │                                                                     │
    │  Stage 5: Synthesis                                                 │
    │    └─► ISTFT (OLA) -> blend STFT + NLMS -> recombine bands            │
    │                                                                     │
    │  Stage 6: Evaluation                                                │
    │    └─► Blind SDR metrics + 6-panel visualisation PNG                │
    │                                                                     │
    │  OUTPUT: karaoke.wav + vocals.wav + pipeline_results.png            │
    └─────────────────────────────────────────────────────────────────────┘

Usage
-----
    python main.py --input your_song.wav [--output_dir ./out] [--decimation 2]

Arguments
---------
    --input        : Path to input stereo WAV file (required)
    --output_dir   : Directory for output files (default: ./output)
    --decimation   : Decimation factor D ∈ {2, 4} (default: 2)
    --n_fft        : STFT window size in samples (default: 2048)
    --hop_length   : STFT hop size in samples (default: 512)
    --nlms_order   : NLMS filter order (default: 256)
    --nlms_mu      : NLMS step size mu (default: 0.3)
    --stft_blend   : STFT estimate blend weight (default: 0.6)
    --nlms_blend   : NLMS estimate blend weight (default: 0.4)
    --plot         : Save visualisation PNG (flag, default: True)
    --demo         : Generate a synthetic test signal for demo (no input needed)
"""

import argparse
import os
import sys
import time
import numpy as np
import librosa
import soundfile as sf

# ── Local pipeline modules ─────────────────────────────────────────────────
from preprocessing import preprocess_audio
from multirate     import polyphase_decimate, polyphase_interpolate, align_length
from spectral      import spectral_separate
from adaptive      import apply_nlms_vocal_cancel, NLMSConfig
from synthesis     import synthesise_output, write_wav
from evaluation    import plot_pipeline_results, print_metrics_report


# ===========================================================================
# Utility: Load Audio
# ===========================================================================

def load_audio(filepath: str, target_sr: int = None) -> tuple:
    """
    Load a stereo audio file and return (stereo_ndarray, sample_rate).

    Parameters
    ----------
    filepath  : Path to WAV / FLAC / MP3 / OGG file
    target_sr : Resample to this rate if not None

    Returns
    -------
    (stereo, fs) : ndarray shape (2, N), float
    """
    # Configure imageio-ffmpeg static binaries for audioread / librosa backend
    try:
        import imageio_ffmpeg
        import shutil
        bin_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        src = imageio_ffmpeg.get_ffmpeg_exe()
        
        dst = os.path.join(bin_dir, "ffmpeg.exe")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            
        dst_probe = os.path.join(bin_dir, "ffprobe.exe")
        if not os.path.exists(dst_probe):
            shutil.copy(src, dst_probe)
            
        if bin_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + bin_dir
    except Exception as e:
        print(f"[Warning] Could not set up imageio-ffmpeg backend path: {e}")

    print(f"[Load] Reading: {filepath}")
    y, fs = librosa.load(
        filepath,
        sr=target_sr,
        mono=False,     # keep stereo
        dtype=np.float32,
    )

    # librosa returns mono as (N,) and stereo as (2, N)
    if y.ndim == 1:
        print("[Load] Mono file detected - duplicating to stereo.")
        y = np.stack([y, y])

    # Trim silence at the very start / end (helps with edge effects)
    # We trim at -70 dBFS which is effectively silence
    # (operate on mono mix for trimming decisions)
    mono_mix = y.mean(axis=0)
    _, (start, end) = librosa.effects.trim(mono_mix, top_db=70)
    y = y[:, start:end]

    duration = y.shape[-1] / fs
    print(f"[Load] Loaded {duration:.1f} s at {fs} Hz, shape={y.shape}")
    return y, fs


# ===========================================================================
# Utility: Generate Synthetic Demo Signal
# ===========================================================================

def generate_demo_signal(duration: float = 30.0, fs: int = 44100) -> np.ndarray:
    """
    Synthesise a demo stereo mix containing a centre-panned 'vocal' sine
    sweep and stereo-panned instrument signals.

    Signal structure
    ----------------
    • Vocal : Sine sweep 200 Hz -> 800 Hz, centred (L = R), AM-modulated
    • Bass  : 80 Hz sine, centred, amplitude 0.4
    • Drums : White noise bursts every 0.5 s, centred, amplitude 0.2
    • Guitar: 500 Hz sine, panned L+R with 20° phase offset, amplitude 0.3
    • Pad   : 1 kHz sine, strongly panned right, amplitude 0.15

    Returns
    -------
    stereo : ndarray, shape (2, N)
    """
    N = int(duration * fs)
    t = np.arange(N) / fs

    # ── Vocal: centre-panned sine sweep + AM envelope ─────────────────────
    f_sweep = np.linspace(200, 800, N)
    phase_vocal = 2 * np.pi * np.cumsum(f_sweep) / fs
    vocal_am = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)   # slow tremolo
    vocal_mono = 0.45 * np.sin(phase_vocal) * vocal_am

    # ── Bass: 80 Hz, centred ──────────────────────────────────────────────
    bass_mono = 0.40 * np.sin(2 * np.pi * 80 * t)

    # ── Kick drum: narrow noise burst every 0.5 s, centred ───────────────
    kick = np.zeros(N)
    for k in range(int(duration * 2)):
        idx = int(k * 0.5 * fs)
        burst_len = int(0.05 * fs)
        if idx + burst_len < N:
            env = np.exp(-40 * np.arange(burst_len) / fs)
            kick[idx:idx + burst_len] += 0.20 * env * np.random.randn(burst_len)

    # ── Guitar: stereo panned at ±20° ─────────────────────────────────────
    guitar_centre = 0.30 * np.sin(2 * np.pi * 500 * t)
    pan_angle = np.pi / 9       # 20 degrees
    guitar_L  = guitar_centre * np.cos(pan_angle + 0.3)
    guitar_R  = guitar_centre * np.cos(pan_angle - 0.3)

    # ── Pad: strongly right-panned ─────────────────────────────────────────
    pad_mono = 0.15 * np.sin(2 * np.pi * 1000 * t)
    pad_L = pad_mono * 0.1
    pad_R = pad_mono * 0.9

    # ── Mix ────────────────────────────────────────────────────────────────
    centre = vocal_mono + bass_mono + kick
    L = centre + guitar_L + pad_L
    R = centre + guitar_R + pad_R

    stereo = np.stack([L.astype(np.float32), R.astype(np.float32)])

    # Normalise
    peak = np.max(np.abs(stereo))
    stereo /= peak + 1e-8

    print(f"[Demo] Generated {duration:.0f} s synthetic mix at {fs} Hz")
    return stereo


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_pipeline(args, progress_cb=None) -> dict:
    """
    Execute the full DSP separation pipeline.

    Parameters
    ----------
    args        : Parsed argparse namespace
    progress_cb : Function callback(stage_name, progress_percent) (optional)

    Returns
    -------
    results : dict with 'karaoke', 'vocals', 'metrics', 'timing'
    """
    os.makedirs(args.output_dir, exist_ok=True)
    timing = {}

    def update_progress(stage_name, percent):
        if progress_cb is not None:
            try:
                progress_cb(stage_name, percent)
            except Exception as e:
                print(f"[Warning] Progress callback error: {e}")

    # ── Stage 0: Load / Generate Input ────────────────────────────────────
    update_progress("Loading Audio", 5)
    t0 = time.perf_counter()
    if args.demo:
        print("\n" + "=" * 60)
        print("  DEMO MODE - Generating synthetic test signal")
        print("=" * 60)
        stereo = generate_demo_signal(duration=30.0, fs=44100)
        fs = 44100
    else:
        if not os.path.isfile(args.input):
            sys.exit(f"[Error] Input file not found: {args.input}")
        stereo, fs = load_audio(args.input)
    timing["load"] = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"  Input: {stereo.shape[-1]} samples @ {fs} Hz")
    print(f"  Duration: {stereo.shape[-1] / fs:.2f} s")
    print(f"{'=' * 60}\n")

    # ── Stage 1: Pre-Processing & Filtering ───────────────────────────────
    update_progress("Pre-Processing & Band Splitting", 15)
    print("\n[STAGE 1] Pre-Processing & Filtering")
    print("-" * 40)
    t1 = time.perf_counter()
    preproc = preprocess_audio(stereo, fs, f_vocal_low=80.0, f_vocal_high=4000.0)
    timing["preprocess"] = time.perf_counter() - t1

    vocal_band = preproc["vocal_band"]   # shape (2, N) - vocal frequency range

    # ── Stage 2: Multi-Rate DSP - Decimation ──────────────────────────────
    update_progress("Decimating Audio (Downsampling)", 30)
    print("\n[STAGE 2] Multi-Rate DSP - Polyphase Decimation")
    print("-" * 40)
    D = args.decimation
    print(f"[MultiRate] Decimation factor D={D}, "
          f"rate: {fs:.0f} Hz -> {fs/D:.0f} Hz")
    t2 = time.perf_counter()
    vocal_band_dec = polyphase_decimate(vocal_band, D, fs)
    fs_dec = fs / D
    print(f"[MultiRate] Decimated shape: {vocal_band_dec.shape}")
    timing["decimation"] = time.perf_counter() - t2

    # ── Stage 3: STFT Analysis & Spectral Masking ─────────────────────────
    update_progress("STFT Analysis & Wiener Soft Masking", 40)
    print("\n[STAGE 3] STFT Analysis & Spectral Masking")
    print("-" * 40)
    # Adjust STFT params for the decimated sample rate
    n_fft      = args.n_fft
    hop_length = args.hop_length
    t3 = time.perf_counter()
    spectral = spectral_separate(
        vocal_band_dec,
        fs_dec,
        n_fft=n_fft,
        hop_length=hop_length,
        wiener_power=2.0,
        coherence_threshold=0.70,
    )
    timing["spectral"] = time.perf_counter() - t3

    # ── Stage 4: Adaptive Filtering (NLMS) ────────────────────────────────
    update_progress("Running Adaptive Filter (NLMS)", 60)
    print("\n[STAGE 4] Adaptive Filtering - NLMS")
    print("-" * 40)

    # CRITICAL FIX: Compute time-domain vocal reference from the STFT
    # vocal estimate.  The old code used cross-channel prediction (L vs R)
    # which cancelled ALL common content, not just vocals.
    from spectral import compute_istft
    vocal_ref_mono = compute_istft(
        spectral["vocal_mid_stft"],
        hop_length=spectral["hop_length"],
        n_fft=spectral["n_fft"],
        length=spectral["n_samples"],
    )
    print(f"[NLMS] Vocal reference signal: {vocal_ref_mono.shape[0]} samples")

    nlms_cfg = NLMSConfig(
        filter_order=args.nlms_order,
        mu=args.nlms_mu,
        eps=1e-4,
        leakage=1.0,
        block_size=64,            # block NLMS for speed
    )
    t4 = time.perf_counter()
    nlms = apply_nlms_vocal_cancel(
        vocal_band_dec, vocal_ref_mono, cfg=nlms_cfg, use_block=True
    )
    timing["nlms"] = time.perf_counter() - t4

    # ── Stage 5: Synthesis - Interpolate back & Recombine ─────────────────
    update_progress("Reconstructing Tracks & Interpolating", 80)
    print("\n[STAGE 5] Synthesis & Reconstruction")
    print("-" * 40)
    t5 = time.perf_counter()

    # Use STFT mask outputs directly as the clean primary separation.
    # The old NLMS blending was corrupting the output by mixing a
    # fundamentally different signal type (filter error) into the STFT estimate.
    from spectral import compute_istft
    from synthesis import recombine_bands, remove_dc_offset, soft_limiter, peak_normalise

    print("[Synthesis] ISTFT - karaoke L (music) ...")
    karaoke_mid_L = compute_istft(
        spectral["music_L_stft"],
        hop_length=spectral["hop_length"],
        n_fft=spectral["n_fft"],
        length=spectral["n_samples"],
    )
    print("[Synthesis] ISTFT - karaoke R (music) ...")
    karaoke_mid_R = compute_istft(
        spectral["music_R_stft"],
        hop_length=spectral["hop_length"],
        n_fft=spectral["n_fft"],
        length=spectral["n_samples"],
    )
    print("[Synthesis] ISTFT - vocals L ...")
    vocals_mid_L = compute_istft(
        spectral["vocals_L_stft"],
        hop_length=spectral["hop_length"],
        n_fft=spectral["n_fft"],
        length=spectral["n_samples"],
    )
    print("[Synthesis] ISTFT - vocals R ...")
    vocals_mid_R = compute_istft(
        spectral["vocals_R_stft"],
        hop_length=spectral["hop_length"],
        n_fft=spectral["n_fft"],
        length=spectral["n_samples"],
    )

    karaoke_mid_dec = np.stack([karaoke_mid_L, karaoke_mid_R])
    vocals_mid_dec  = np.stack([vocals_mid_L,  vocals_mid_R])

    # Interpolate separated mid-band back to original rate
    print(f"[MultiRate] Interpolating back to {fs:.0f} Hz ...")
    karaoke_mid_full = polyphase_interpolate(karaoke_mid_dec, D, fs_dec)
    vocals_mid_full  = polyphase_interpolate(vocals_mid_dec,  D, fs_dec)

    # Align lengths with bass/treble bands
    bass_band   = preproc["bass_band"]
    treble_band = preproc["treble_band"]
    karaoke_mid_full, bass_band   = align_length(karaoke_mid_full, bass_band)
    karaoke_mid_full, treble_band = align_length(karaoke_mid_full, treble_band)
    vocals_mid_full, _            = align_length(vocals_mid_full,  bass_band)

    N_full = karaoke_mid_full.shape[-1]

    # Karaoke = bass + instrumental mid + treble  (full-band instrumental)
    karaoke_full = recombine_bands(
        bass_band[:, :N_full],
        karaoke_mid_full,
        treble_band[:, :N_full],
        fs,
    )
    # Vocals = bass + vocal mid + treble  (full-band — fixes thin/high-pitched sound)
    vocals_full = recombine_bands(
        bass_band[:, :N_full],
        vocals_mid_full,
        treble_band[:, :N_full],
        fs,
    )

    # Final post-processing at full rate
    karaoke_full = remove_dc_offset(karaoke_full, fs)
    vocals_full  = remove_dc_offset(vocals_full,  fs)
    karaoke_full = soft_limiter(karaoke_full)
    vocals_full  = soft_limiter(vocals_full)
    karaoke_full = peak_normalise(karaoke_full, -1.0)
    vocals_full  = peak_normalise(vocals_full,  -1.0)
    timing["synthesis"] = time.perf_counter() - t5

    # ── Write outputs ─────────────────────────────────────────────────────
    update_progress("Writing output WAV files", 90)
    karaoke_path = os.path.join(args.output_dir, "karaoke.wav")
    vocals_path  = os.path.join(args.output_dir, "vocals.wav")
    write_wav(karaoke_full, fs, karaoke_path)
    write_wav(vocals_full,  fs, vocals_path)

    # ── Stage 6: Evaluation & Visualisation ───────────────────────────────
    update_progress("Evaluating Metrics & Generating Plot", 95)
    print("\n[STAGE 6] Evaluation & Visualisation")
    print("-" * 40)
    t6 = time.perf_counter()

    # Mono mix for blind SDR
    mono_mix    = stereo.mean(axis=0)
    mono_kar    = karaoke_full.mean(axis=0)
    mono_voc    = vocals_full.mean(axis=0)

    metrics = print_metrics_report(mono_kar, mono_voc, mono_mix, fs)

    if args.plot:
        plot_path = os.path.join(args.output_dir, "pipeline_results.png")
        # Trim mse arrays to manageable size for plotting
        mse_L = nlms["mse_L"]
        mse_R = nlms["mse_R"]
        plot_pipeline_results(
            original_stereo=stereo,
            karaoke=karaoke_full,
            vocals=vocals_full,
            mse_curve_L=mse_L,
            mse_curve_R=mse_R,
            spectral_mask=spectral["mask"],
            fs=fs_dec,       # mask was computed at decimated rate
            output_path=plot_path,
        )
    timing["evaluation"] = time.perf_counter() - t6
    update_progress("Process Complete", 100)

    # ── Summary ───────────────────────────────────────────────────────────
    total = sum(timing.values())
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - TIMING SUMMARY")
    print("=" * 60)
    for stage, dt in timing.items():
        print(f"  {stage:<20} {dt:6.2f} s  ({dt/total*100:5.1f} %)")
    print(f"  {'TOTAL':<20} {total:6.2f} s")
    print(f"\n  Outputs saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60 + "\n")

    return {
        "karaoke": karaoke_full,
        "vocals":  vocals_full,
        "metrics": metrics,
        "timing":  timing,
    }


# ===========================================================================
# AI Pipeline
# ===========================================================================

def run_ai_pipeline(args) -> dict:
    """
    Execute the Demucs AI-based source separation pipeline.

    Uses Meta's htdemucs model to produce clean 4-stem separation:
      vocals | drums | bass | other
    The non-vocal stems are summed into the final karaoke backing track.
    """
    from ai_separator import ai_separate, normalise
    import soundfile as sf

    os.makedirs(args.output_dir, exist_ok=True)
    timing = {}

    # ── Load audio ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if args.demo:
        print("\n" + "=" * 60)
        print("  AI MODE - DEMO: Generating synthetic test signal")
        print("=" * 60)
        stereo = generate_demo_signal(duration=10.0, fs=44100)
        fs = 44100
    else:
        if not os.path.isfile(args.input):
            sys.exit(f"[Error] Input file not found: {args.input}")
        stereo, fs = load_audio(args.input)
    timing["load"] = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"  [AI MODE] Demucs htdemucs source separation")
    print(f"  Input: {stereo.shape[-1]} samples @ {fs} Hz")
    print(f"  Duration: {stereo.shape[-1] / fs:.2f} s")
    print(f"{'=' * 60}\n")

    # ── Run Demucs ────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    stems, out_sr = ai_separate(
        audio      = stereo,
        sr         = int(fs),
        model_name = getattr(args, "ai_model",  "htdemucs"),
        device     = "auto",
        shifts     = getattr(args, "ai_shifts",  1),
        segment    = getattr(args, "ai_segment", 7.8),
    )
    timing["ai_inference"] = time.perf_counter() - t1

    # ── Post-process & save ───────────────────────────────────────────────
    t2 = time.perf_counter()
    karaoke = normalise(stems["karaoke"], headroom_db=-1.0)   # (2, N)
    vocals  = normalise(stems["vocals"],  headroom_db=-1.0)   # (2, N)

    karaoke_path = os.path.join(args.output_dir, "karaoke.wav")
    vocals_path  = os.path.join(args.output_dir, "vocals.wav")

    # soundfile expects (N, channels)
    sf.write(karaoke_path, karaoke.T, out_sr, subtype="PCM_16")
    sf.write(vocals_path,  vocals.T,  out_sr, subtype="PCM_16")
    print(f"\n[AI] Saved karaoke track -> {karaoke_path}")
    print(f"[AI] Saved vocal track   -> {vocals_path}")

    # Save individual stems too (drums, bass, other)
    for stem_name in ["drums", "bass", "other"]:
        if stem_name in stems:
            stem_norm = normalise(stems[stem_name], headroom_db=-2.0)
            stem_path = os.path.join(args.output_dir, f"{stem_name}.wav")
            sf.write(stem_path, stem_norm.T, out_sr, subtype="PCM_16")
            print(f"[AI] Saved {stem_name} stem     -> {stem_path}")
    timing["save"] = time.perf_counter() - t2

    # ── Evaluation & Visualisation ────────────────────────────────────────
    t3 = time.perf_counter()
    mono_mix = stereo.mean(axis=0)

    # Align lengths (Demucs may change N slightly due to padding)
    N = min(mono_mix.shape[0], karaoke.shape[1], vocals.shape[1])
    mono_kar = karaoke[:, :N].mean(axis=0)
    mono_voc = vocals[:, :N].mean(axis=0)
    mono_mix = mono_mix[:N]

    metrics = print_metrics_report(mono_kar, mono_voc, mono_mix, out_sr)

    if args.plot:
        _plot_ai_results(
            original = stereo,
            karaoke  = karaoke,
            vocals   = vocals,
            stems    = stems,
            fs       = out_sr,
            output_path = os.path.join(args.output_dir, "ai_pipeline_results.png"),
        )
    timing["evaluation"] = time.perf_counter() - t3

    # ── Timing summary ────────────────────────────────────────────────────
    total = sum(timing.values())
    print("\n" + "=" * 60)
    print("  AI PIPELINE COMPLETE - TIMING SUMMARY")
    print("=" * 60)
    for stage, dt in timing.items():
        print(f"  {stage:<22} {dt:6.2f} s  ({dt/total*100:5.1f} %)")
    print(f"  {'TOTAL':<22} {total:6.2f} s")
    print(f"\n  Outputs saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60 + "\n")

    return {"karaoke": karaoke, "vocals": vocals, "metrics": metrics, "timing": timing}


def _plot_ai_results(original, karaoke, vocals, stems, fs, output_path):
    """Generate a 6-panel diagnostic plot for the AI separation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.signal import stft as scipy_stft

    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    def _spectrogram(ax, audio_mono, title, sr, cmap="magma"):
        f, t, Zxx = scipy_stft(audio_mono, fs=sr, nperseg=2048, noverlap=1536)
        db = 20 * np.log10(np.abs(Zxx) + 1e-9)
        im = ax.imshow(db, aspect="auto", origin="lower", cmap=cmap,
                       extent=[t[0], t[-1], f[0]/1000, f[-1]/1000],
                       vmin=-80, vmax=0)
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#e6edf3", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        ax.set_ylabel("Freq (kHz)", color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        plt.colorbar(im, ax=ax, pad=0.02).ax.tick_params(colors="#8b949e")

    orig_mono    = original.mean(axis=0) if original.ndim == 2 else original
    karaoke_mono = karaoke.mean(axis=0)  if karaoke.ndim  == 2 else karaoke
    vocals_mono  = vocals.mean(axis=0)   if vocals.ndim   == 2 else vocals

    N = min(len(orig_mono), len(karaoke_mono), len(vocals_mono))

    panels = [
        (orig_mono[:N],    "Original Mix",      "magma"),
        (karaoke_mono[:N], "Karaoke (AI)",      "viridis"),
        (vocals_mono[:N],  "Vocals (AI)",       "plasma"),
    ]

    # Add individual stems
    stem_panels = []
    for sname, cmap in [("drums","Blues"), ("bass","Greens"), ("other","Oranges")]:
        if sname in stems:
            smono = stems[sname].mean(axis=0)[:N]
            stem_panels.append((smono, f"Stem: {sname.capitalize()}", cmap))

    all_panels = panels + stem_panels

    positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for (row, col), (data, title, cmap) in zip(positions, all_panels):
        _spectrogram(fig.add_subplot(gs[row, col]), data, title, fs, cmap)

    # Waveform comparison (bottom row col 2)
    ax_wave = fig.add_subplot(gs[2, :])
    ax_wave.set_facecolor("#161b22")
    t_axis = np.linspace(0, N/fs, N)
    ax_wave.plot(t_axis, orig_mono[:N],    color="#58a6ff", lw=0.4, alpha=0.7, label="Original")
    ax_wave.plot(t_axis, karaoke_mono[:N], color="#3fb950", lw=0.4, alpha=0.8, label="Karaoke")
    ax_wave.plot(t_axis, vocals_mono[:N],  color="#f78166", lw=0.4, alpha=0.8, label="Vocals")
    ax_wave.set_title("Waveform Comparison: Original vs. Karaoke vs. Vocals",
                      color="#e6edf3", fontsize=10, fontweight="bold")
    ax_wave.set_xlabel("Time (s)", color="#8b949e")
    ax_wave.set_ylabel("Amplitude", color="#8b949e")
    ax_wave.tick_params(colors="#8b949e", labelsize=7)
    ax_wave.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
    for sp in ax_wave.spines.values(): sp.set_edgecolor("#30363d")

    fig.suptitle("AI Source Separation — Demucs htdemucs Results",
                 color="#f0f6fc", fontsize=14, fontweight="bold", y=0.97)
    plt.savefig(output_path, dpi=150, facecolor="#0d1117")
    plt.close(fig)
    print(f"[AI] Saved diagnostic plot -> {output_path}")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vocal Remover & Karaoke Generator - DSP + AI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Shared arguments ───────────────────────────────────────────────────
    p.add_argument("--input",      type=str, default="",       help="Input stereo WAV/MP3 file")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory")
    p.add_argument("--demo",       action="store_true", default=False,
                   help="Run with a synthetic test signal (no input file needed)")
    p.add_argument("--plot",       action="store_true", default=True,
                   help="Generate evaluation/diagnostic plots")
    p.add_argument("--no-plot",    action="store_false", dest="plot",
                   help="Do not generate plots")
    p.add_argument("--mode",       type=str, default="dsp", choices=["dsp", "ai"],
                   help="Separation mode: 'dsp' = classical pipeline, 'ai' = Demucs AI model")

    # ── DSP-mode arguments (ignored in AI mode) ────────────────────────────
    dsp = p.add_argument_group("DSP mode options")
    dsp.add_argument("--decimation",  type=int,   default=2,    choices=[1, 2, 4],
                     help="Polyphase decimation factor D")
    dsp.add_argument("--n_fft",       type=int,   default=2048, help="STFT window size")
    dsp.add_argument("--hop_length",  type=int,   default=512,  help="STFT hop length")
    dsp.add_argument("--nlms_order",  type=int,   default=256,  help="NLMS filter taps")
    dsp.add_argument("--nlms_mu",     type=float, default=0.3,  help="NLMS step size")
    dsp.add_argument("--stft_blend",  type=float, default=0.6,
                     help="Weight of STFT estimate in final mix")
    dsp.add_argument("--nlms_blend",  type=float, default=0.4,
                     help="Weight of NLMS estimate in final mix")

    # ── AI-mode arguments (ignored in DSP mode) ────────────────────────────
    ai = p.add_argument_group("AI mode options")
    ai.add_argument("--ai_model",   type=str,   default="htdemucs",
                    help="Demucs model name (htdemucs | htdemucs_ft | mdx_extra)")
    ai.add_argument("--ai_shifts",  type=int,   default=1,
                    help="Test-time shifts (1=fast, 4=higher quality)")
    ai.add_argument("--ai_segment", type=float, default=7.8,
                    help="Segment length in seconds for chunked inference")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-enable demo if no input given
    if not args.input and not args.demo:
        print("[Info] No --input provided. Running in --demo mode.")
        args.demo = True

    if args.mode == "ai":
        print(f"\n[Mode] AI-Based Separation (Demucs '{args.ai_model}')")
        results = run_ai_pipeline(args)
    else:
        print(f"\n[Mode] Classical DSP Pipeline")
        results = run_pipeline(args)
