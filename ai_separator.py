"""
=============================================================================
  ai_separator.py  —  Demucs v4 AI-Based Source Separation
=============================================================================
  Uses Meta's Demucs (htdemucs model) to separate a stereo audio track into
  4 independent stems:
      vocals | drums | bass | other

  The non-vocal stems are mixed back together to form the karaoke backing
  track, giving near-perfect vocal removal unavailable from classical DSP.

  Model: htdemucs  (Hybrid Transformer Demucs, v4)
    • State-of-the-art source separation (SDR ~8–9 dB on MUSDB18)
    • Operates in both time-domain and STFT-domain simultaneously
    • Auto-downloads weights (~130 MB) on first run, cached afterwards

  GPU Acceleration:
    • Automatically uses CUDA if an NVIDIA GPU is detected
    • Falls back to CPU if no GPU is available (slower but still works)

  Reference:
    Défossez, A. et al. (2021). "Hybrid Spectrogram and Waveform Source
    Separation". arXiv:2111.03600
=============================================================================
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Lazy-import guard: demucs is an optional dependency
# ---------------------------------------------------------------------------
def _check_demucs():
    try:
        import torch        # noqa: F401
        import demucs       # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Core AI separation function
# ---------------------------------------------------------------------------

def ai_separate(audio: np.ndarray,
                sr:    int,
                model_name: str  = "htdemucs",
                device:     str  = "auto",
                segment:    float = 5.0,
                overlap:    float = 0.1,
                shifts:     int   = 1) -> dict:
    """
    Separate a stereo audio signal into 4 stems using Demucs v4.

    Parameters
    ----------
    audio      : np.ndarray  shape (2, N) or (N, 2)  — stereo float audio
    sr         : int  — sample rate of `audio` (Demucs internally resamples
                        to 44100 Hz if needed)
    model_name : str  — Demucs model ID (default: 'htdemucs')
                        Other options: 'htdemucs_ft', 'mdx_extra'
    device     : str  — 'auto' | 'cuda' | 'cpu'
    segment    : float — segment length in seconds for chunked inference
                        (reduce if you get CUDA OOM errors)
    overlap    : float — overlap ratio between segments [0, 1)
    shifts     : int  — number of random shifts for test-time augmentation
                        (1 = fast, 4 = slightly better quality)

    Returns
    -------
    dict with keys: 'vocals', 'drums', 'bass', 'other', 'karaoke'
      Each value is a np.ndarray of shape (2, N) — stereo float32 audio
      'karaoke' = drums + bass + other  (vocal-free backing track)
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # ── Device selection ──────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [AI] Using device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [AI] GPU: {gpu_name}")

    # ── Normalise audio shape to (2, N) ──────────────────────────────────
    if audio.ndim == 2 and audio.shape[1] != 2 and audio.shape[0] == 2:
        pass  # already (2, N)
    elif audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.T  # (N, 2) → (2, N)
    elif audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)  # mono → stereo
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    audio = audio.astype(np.float32)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"  [AI] Loading model '{model_name}' ...")
    print(f"       (Weights auto-download ~130 MB on first run)")
    model = get_model(model_name)
    model.to(device)
    model.eval()
    print(f"  [AI] Model loaded. Stems: {model.sources}")

    # Demucs expects 44100 Hz; resample if needed
    model_sr = getattr(model, "samplerate", 44100)
    if sr != model_sr:
        import torchaudio.functional as F_audio
        wav_t = torch.from_numpy(audio)
        wav_t = F_audio.resample(wav_t, orig_freq=sr, new_freq=model_sr)
        audio  = wav_t.numpy()
        print(f"  [AI] Resampled audio {sr} Hz → {model_sr} Hz")
        sr = model_sr

    # ── Run inference ─────────────────────────────────────────────────────
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, 2, N)
    audio_tensor = audio_tensor.to(device)

    print(f"  [AI] Running inference  "
          f"(duration={audio.shape[1]/sr:.1f}s  shifts={shifts}) ...")

    with torch.no_grad():
        sources = apply_model(
            model,
            audio_tensor,
            device    = device,
            shifts    = shifts,
            split     = True,
            overlap   = overlap,
            segment   = segment,
            progress  = True,
        )  # → Tensor (1, n_stems, 2, N)

    sources = sources.squeeze(0).cpu().numpy()  # (n_stems, 2, N)

    # ── Map stems to dict ─────────────────────────────────────────────────
    stem_names = model.sources   # e.g. ['drums', 'bass', 'other', 'vocals']
    stems: dict = {}
    for i, name in enumerate(stem_names):
        stems[name] = sources[i].astype(np.float32)   # (2, N)

    # ── Build karaoke mix (all stems except vocals) ───────────────────────
    non_vocal_keys = [k for k in stem_names if k != "vocals"]
    karaoke = sum(stems[k] for k in non_vocal_keys).astype(np.float32)
    stems["karaoke"] = karaoke

    # ── Print per-stem RMS ────────────────────────────────────────────────
    for name, data in stems.items():
        rms = np.sqrt(np.mean(data ** 2))
        print(f"  [AI] {name:<12} RMS = {rms:.6f}")

    return stems, sr


# ---------------------------------------------------------------------------
# Utility: mix multiple stereo stems
# ---------------------------------------------------------------------------

def mix_stems(stem_arrays: list) -> np.ndarray:
    """Sum a list of (2, N) stem arrays into one stereo mix."""
    result = sum(s.astype(np.float64) for s in stem_arrays)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Utility: normalise to peak headroom
# ---------------------------------------------------------------------------

def normalise(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Peak-normalise stereo (2, N) or mono (N,) audio."""
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    target = 10 ** (headroom_db / 20.0)
    return (audio / peak * target).astype(np.float32)
