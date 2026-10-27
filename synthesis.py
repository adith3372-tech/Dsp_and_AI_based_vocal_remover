"""
=============================================================================
synthesis.py - Band Recombination & Final Output Synthesis
=============================================================================

DSP Rationale:
    After separation, we have three frequency zones:
        1. bass_band   - original, unprocessed (< 80 Hz)
        2. vocal_band  - split by STFT + NLMS into music & vocal estimates
        3. treble_band - original, unprocessed (> 4 kHz)

    The final outputs are assembled by summing the appropriate bands:

        karaoke(n) = bass_band(n) + music_vocal_band(n) + treble_band(n)
        vocals(n)  = vocal_estimate_band(n)

    Band summation is valid because the FIR / IIR filters are complementary
    and designed to have minimal overlap in their transition regions.

    After summing, we apply:
    - Normalisation  : Scale to prevent clipping (peak normalise to –1 dBFS)
    - DC offset removal: High-pass at 10 Hz to remove any processing-induced DC
    - Gentle limiting: Soft-knee limiter to catch rare transient peaks
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf


# ---------------------------------------------------------------------------
# 1.  DC Offset Removal
# ---------------------------------------------------------------------------

def remove_dc_offset(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove DC offset introduced by processing by high-passing at 10 Hz.

    A processing pipeline can introduce a small, constant offset in the output
    due to filter group delays, edge effects, or numerical accumulation.
    Even a tiny DC offset causes audible thumps on play/pause and wastes
    headroom.

    Parameters
    ----------
    signal : ndarray, shape (2, N) or (N,)
    fs     : Sample rate in Hz

    Returns
    -------
    signal with DC removed
    """
    sos = butter(2, 10.0 / (fs / 2.0), btype="high", output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal)
    return np.stack([sosfiltfilt(sos, signal[ch]) for ch in range(signal.shape[0])])


# ---------------------------------------------------------------------------
# 2.  Peak Normalisation
# ---------------------------------------------------------------------------

def peak_normalise(signal: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """
    Scale signal so that its peak amplitude is exactly `target_dbfs` dBFS.

    target_dbfs = 0.0 -> full scale (dangerous near clipping)
    target_dbfs = -1.0 -> 1 dB headroom (recommended for final output)

    Parameters
    ----------
    signal      : ndarray, any shape
    target_dbfs : Target peak level in dBFS

    Returns
    -------
    normalised ndarray
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-9:
        return signal   # silent signal - do nothing
    target_linear = 10 ** (target_dbfs / 20.0)
    return signal * (target_linear / peak)


# ---------------------------------------------------------------------------
# 3.  Soft-Knee Limiter
# ---------------------------------------------------------------------------

def soft_limiter(signal: np.ndarray, threshold: float = 0.95, knee: float = 0.1) -> np.ndarray:
    """
    Apply soft-knee amplitude limiting to catch rare transient peaks.

    The transfer function is:
        - Linear below (threshold - knee/2)
        - Soft-knee cubic spline between (threshold ± knee/2)
        - Clamped at threshold above the knee

    This prevents hard clipping distortion while leaving quiet passages
    untouched.

    Parameters
    ----------
    signal    : ndarray - audio signal
    threshold : Hard limiting ceiling (0–1)
    knee      : Width of the soft transition region

    Returns
    -------
    limited signal (same shape)
    """
    out = signal.copy()
    lo  = threshold - knee / 2.0
    hi  = threshold + knee / 2.0

    abs_sig = np.abs(signal)
    # Soft-knee region: apply quadratic interpolation
    in_knee = (abs_sig > lo) & (abs_sig < hi)
    if np.any(in_knee):
        # Normalise to [0, 1] across the knee
        t = (abs_sig[in_knee] - lo) / knee
        # Quadratic: g(t) = lo + knee·t - 0.5·knee·t²
        gain_factor = (lo + knee * t - 0.5 * knee * t**2) / (abs_sig[in_knee] + 1e-12)
        out[in_knee] = signal[in_knee] * gain_factor

    # Hard limit above knee
    over = abs_sig >= hi
    if np.any(over):
        out[over] = np.sign(signal[over]) * threshold

    return out


# ---------------------------------------------------------------------------
# 4.  Band Recombination
# ---------------------------------------------------------------------------

def recombine_bands(
    bass_band:   np.ndarray,
    music_mid:   np.ndarray,
    treble_band: np.ndarray,
    fs:          float,
) -> np.ndarray:
    """
    Recombine bass, separated mid, and treble into a full-bandwidth track.

    Parameters
    ----------
    bass_band   : ndarray, shape (2, N) - below 80 Hz
    music_mid   : ndarray, shape (2, N) - separated mid band (music or vocal)
    treble_band : ndarray, shape (2, N) - above 4 kHz
    fs          : Sample rate in Hz

    Returns
    -------
    combined : ndarray, shape (2, N)
    """
    # Align lengths (small differences can occur after polyphase processing)
    n = min(bass_band.shape[-1], music_mid.shape[-1], treble_band.shape[-1])
    combined = bass_band[:, :n] + music_mid[:, :n] + treble_band[:, :n]
    return combined


# ---------------------------------------------------------------------------
# 5.  Full Synthesis Pipeline
# ---------------------------------------------------------------------------

def synthesise_output(
    spectral_result: dict,
    nlms_result:     dict,
    preproc_result:  dict,
    original_stereo: np.ndarray,
    fs:              float,
    stft_blend:      float = 0.6,
    nlms_blend:      float = 0.4,
) -> dict:
    """
    Synthesise the final karaoke and vocals tracks from all pipeline stages.

    Blending Strategy
    -----------------
    Both the STFT masking stage and the NLMS stage produce independent
    estimates.  We blend them with configurable weights:

        music_est = stft_blend · music_stft_istft  +  nlms_blend · music_nlms
        vocal_est = stft_blend · vocal_stft_istft  +  nlms_blend · vocal_nlms

    The STFT mask tends to be more globally accurate; NLMS is better at
    tracking rapid vocal transients.  An equal blend is a safe default.

    Parameters
    ----------
    spectral_result : Output of spectral.spectral_separate()
    nlms_result     : Output of adaptive.apply_nlms_to_stereo()
    preproc_result  : Output of preprocessing.preprocess_audio()
    original_stereo : Raw stereo input for final alignment
    fs              : Sample rate in Hz
    stft_blend      : Weight for STFT estimate (0–1)
    nlms_blend      : Weight for NLMS estimate (0–1)

    Returns
    -------
    dict:
        'karaoke' : ndarray, shape (2, N) - instrumental track
        'vocals'  : ndarray, shape (2, N) - vocal track
    """
    from spectral import compute_istft

    n_fft      = spectral_result["n_fft"]
    hop_length = spectral_result["hop_length"]
    n_samples  = spectral_result["n_samples"]

    # -- STFT-domain reconstructions (per-channel L/R) -------------------
    # CRITICAL FIX: The old code ran ISTFT on a single mid-channel mono
    # signal and duplicated it to fake stereo.  This discarded all stereo
    # information.  We now reconstruct L and R channels independently.
    print("[Synthesis] ISTFT - vocal L estimate ...")
    vocal_stft_L = compute_istft(
        spectral_result["vocals_L_stft"],
        hop_length=hop_length, n_fft=n_fft, length=n_samples,
    )
    print("[Synthesis] ISTFT - vocal R estimate ...")
    vocal_stft_R = compute_istft(
        spectral_result["vocals_R_stft"],
        hop_length=hop_length, n_fft=n_fft, length=n_samples,
    )
    print("[Synthesis] ISTFT - music L estimate ...")
    music_stft_L = compute_istft(
        spectral_result["music_L_stft"],
        hop_length=hop_length, n_fft=n_fft, length=n_samples,
    )
    print("[Synthesis] ISTFT - music R estimate ...")
    music_stft_R = compute_istft(
        spectral_result["music_R_stft"],
        hop_length=hop_length, n_fft=n_fft, length=n_samples,
    )

    # Stack into proper stereo (preserving original panning)
    vocal_stft_stereo = np.stack([vocal_stft_L, vocal_stft_R])
    music_stft_stereo = np.stack([music_stft_L, music_stft_R])

    # -- NLMS time-domain estimates (already stereo) ----------------------
    N = n_samples
    vocal_nlms = np.stack([
        nlms_result["vocal_L"][:N],
        nlms_result["vocal_R"][:N],
    ])
    music_nlms = np.stack([
        nlms_result["music_L"][:N],
        nlms_result["music_R"][:N],
    ])

    # -- Align lengths -----------------------------------------------------
    n_min = min(
        vocal_stft_stereo.shape[-1],
        music_stft_stereo.shape[-1],
        vocal_nlms.shape[-1],
        music_nlms.shape[-1],
    )
    vocal_stft_stereo = vocal_stft_stereo[:, :n_min]
    music_stft_stereo = music_stft_stereo[:, :n_min]
    vocal_nlms = vocal_nlms[:, :n_min]
    music_nlms = music_nlms[:, :n_min]

    # -- Blend STFT and NLMS estimates ------------------------------------
    vocal_mid_band = stft_blend * vocal_stft_stereo + nlms_blend * vocal_nlms
    music_mid_band = stft_blend * music_stft_stereo + nlms_blend * music_nlms

    # -- Recombine with bass and treble -----------------------------------
    bass   = preproc_result["bass_band"]
    treble = preproc_result["treble_band"]

    print("[Synthesis] Recombining bands ...")
    karaoke_raw = recombine_bands(bass, music_mid_band, treble, fs)
    vocals_raw  = vocal_mid_band   # vocals = mid-band only (no bass/treble)

    # -- Post-processing --------------------------------------------------
    print("[Synthesis] DC removal, normalisation, limiting ...")
    karaoke = remove_dc_offset(karaoke_raw, fs)
    vocals  = remove_dc_offset(vocals_raw,  fs)

    karaoke = soft_limiter(karaoke)
    vocals  = soft_limiter(vocals)

    karaoke = peak_normalise(karaoke, target_dbfs=-1.0)
    vocals  = peak_normalise(vocals,  target_dbfs=-1.0)

    return {"karaoke": karaoke, "vocals": vocals}


# ---------------------------------------------------------------------------
# 6.  Write WAV Output
# ---------------------------------------------------------------------------

def write_wav(signal: np.ndarray, fs: float, filepath: str) -> None:
    """
    Write a stereo signal to a WAV file using soundfile (libsndfile).

    soundfile uses 32-bit float PCM by default which provides 144 dB of
    dynamic range - more than sufficient for any perceptual audio task.

    Parameters
    ----------
    signal   : ndarray, shape (2, N) - stereo signal
    fs       : Sample rate in Hz
    filepath : Output file path (e.g. 'karaoke.wav')
    """
    # soundfile expects (N, channels) layout
    data = signal.T.astype(np.float32)
    # Clip to [-1, 1] as a final safety guard
    data = np.clip(data, -1.0, 1.0)
    sf.write(filepath, data, int(fs), subtype="PCM_24")
    print(f"[Output] Saved: {filepath}  ({data.shape[0]} samples, {fs} Hz)")
