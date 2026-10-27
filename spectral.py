"""
=============================================================================
spectral.py - STFT Analysis, Centre-Pan Cancellation & Soft Masking
=============================================================================

DSP Rationale:
    Vocals in commercial mixes are almost universally panned to the centre
    (equal amplitude in both channels, in-phase).  The key insight is:

        mid  = (L + R) / 2   ← contains centre-panned content (vocals + kick)
        side = (L - R) / 2   ← contains stereo-panned content (guitars, pads)

    A simple L - R subtraction in the time domain removes centre-panned content.
    But this also removes kick drums and lead synths.  We improve this by:

    1. Computing the STFT of both channels
       X(k, m) = FFT{ x(n) · w(n - m·H) }

       where w is a Hann window of length N_fft, H is the hop size,
       k is frequency bin, m is frame index.

    2. Building a *soft* Wiener mask that attenuates only those TF bins where
       |X_L(k,m)| ≈ |X_R(k,m)| AND ∠X_L ≈ ∠X_R (i.e., truly centre-panned):

            M_vocal(k,m) = |X_mid(k,m)|^p / (|X_mid(k,m)|^p + |X_side(k,m)|^p + eps)

       p controls softness (p=2 -> Wiener, p->∞ -> binary)

    3. Subtracting the masked vocal spectrogram from the full mix to get
       the instrumental spectrogram.

    Reference: Zafar Rafii & Bryan Pardo, "Music/Voice Separation Using the
    Similarity Matrix", ISMIR 2012.
"""

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# 1.  STFT / ISTFT Wrappers
# ---------------------------------------------------------------------------

def compute_stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT).

    X(k, m) = Σ_{n=0}^{N-1} x(n + mH) · w(n) · e^{-j2πkn/N}

    where N = n_fft, H = hop_length, w(n) = Hann window.

    Parameters
    ----------
    signal     : ndarray, shape (samples,) - single channel
    n_fft      : FFT size N (determines frequency resolution Δf = fs/N)
    hop_length : Frame shift H in samples (determines time resolution)
    win_length : Analysis window length (defaults to n_fft)
    window     : Window type (Hann is 50 % overlap-compatible)

    Returns
    -------
    X : ndarray, shape (n_fft//2+1, T), complex - one-sided STFT
    """
    win_length = win_length or n_fft
    X = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,        # zero-pad so first frame is centred on sample 0
    )
    return X


def compute_istft(
    X: np.ndarray,
    hop_length: int = 512,
    win_length: int = None,
    n_fft: int = 2048,
    length: int = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute the Inverse STFT using overlap-add (OLA) reconstruction.

    x̂(n) = Σ_m [ IFFT{X(·,m)} · w(n) ] overlapped and summed.

    The Hann window with 50 % overlap satisfies the Constant OverLap-Add
    (COLA) condition, guaranteeing perfect reconstruction:
        Σ_m w²(n - mH) = const   for H = N/2

    Parameters
    ----------
    X          : Complex STFT array, shape (n_fft//2+1, T)
    hop_length : Must match the hop used during analysis
    win_length : Must match the window length used during analysis
    n_fft      : FFT size used during analysis
    length     : Trim / zero-pad output to this many samples (optional)
    window     : Window type (must match analysis)

    Returns
    -------
    x : ndarray, shape (samples,) - reconstructed time-domain signal
    """
    win_length = win_length or n_fft
    x = librosa.istft(
        X,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )
    return x


# ---------------------------------------------------------------------------
# 2.  Mid-Side Decomposition in the STFT Domain
# ---------------------------------------------------------------------------

def midside_decompose(
    X_L: np.ndarray,
    X_R: np.ndarray,
) -> tuple:
    """
    Convert Left/Right STFT spectra to Mid/Side.

        X_mid  = (X_L + X_R) / √2   - centre-panned content
        X_side = (X_L - X_R) / √2   - stereo-panned content

    The √2 normalisation preserves total power:
        |X_mid|² + |X_side|² = |X_L|² + |X_R|²

    Parameters
    ----------
    X_L, X_R : complex ndarrays, shape (F, T)

    Returns
    -------
    (X_mid, X_side) : complex ndarrays, shape (F, T)
    """
    X_mid  = (X_L + X_R) / np.sqrt(2.0)
    X_side = (X_L - X_R) / np.sqrt(2.0)
    return X_mid, X_side


def midside_recompose(
    X_mid: np.ndarray,
    X_side: np.ndarray,
) -> tuple:
    """
    Reconstruct Left/Right STFT from Mid/Side.

        X_L = (X_mid + X_side) / √2
        X_R = (X_mid - X_side) / √2

    Parameters
    ----------
    X_mid, X_side : complex ndarrays, shape (F, T)

    Returns
    -------
    (X_L, X_R) : complex ndarrays, shape (F, T)
    """
    X_L = (X_mid + X_side) / np.sqrt(2.0)
    X_R = (X_mid - X_side) / np.sqrt(2.0)
    return X_L, X_R


# ---------------------------------------------------------------------------
# 3.  Soft Wiener Mask
# ---------------------------------------------------------------------------

def compute_wiener_mask(
    X_vocal_est: np.ndarray,
    X_music_est: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a soft (Wiener-filter) spectral mask.

    The Wiener mask trades off between the estimated vocal and music
    magnitudes in each time-frequency bin:

        M(k,m) = |X_vocal(k,m)|^p
                 ─────────────────────────────────────────
                 |X_vocal(k,m)|^p + |X_music(k,m)|^p + eps

    Properties:
    - When vocal energy dominates:  M(k,m) -> 1  (pass vocal)
    - When music energy dominates:  M(k,m) -> 0  (suppress vocal)
    - p = 1 -> gentle soft mask
    - p = 2 -> Wiener filter (minimum mean-square-error optimal estimator)
    - p -> ∞ -> binary mask

    The eps term provides numerical stability and sets a noise floor.

    Parameters
    ----------
    X_vocal_est : Estimated vocal STFT magnitude, shape (F, T)
    X_music_est : Estimated music STFT magnitude, shape (F, T)
    power       : Exponent p controlling mask softness
    eps         : Small regularisation term

    Returns
    -------
    mask : real-valued ndarray in [0, 1], shape (F, T)
    """
    V = np.abs(X_vocal_est) ** power
    M = np.abs(X_music_est) ** power
    mask = V / (V + M + eps)
    return mask.astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Phase-Coherence Mask (Lateral Suppression)
# ---------------------------------------------------------------------------

def compute_phase_coherence_mask(
    X_L: np.ndarray,
    X_R: np.ndarray,
    threshold: float = 0.70,
) -> np.ndarray:
    """
    Identify TF bins where L and R are highly phase-coherent and centre-panned.

    We compute phase alignment cos(θ_L - θ_R) and amplitude similarity:

        cos_phase = Real(X_L · X_R*) / (|X_L| · |X_R| + eps)
        amp_sim   = (2 · |X_L| · |X_R|) / (|X_L|² + |X_R|² + eps)
        γ(k,m)    = max(0, cos_phase) · amp_sim

    γ = 1 means identical phase and amplitude (truly centre-panned vocal).
    γ < 1 for stereo-panned or phase-spread instruments (piano, violin, reverb).

    Parameters
    ----------
    X_L, X_R  : Complex STFT arrays, shape (F, T)
    threshold : Minimum coherence to be considered 'vocal' bin (0–1)

    Returns
    -------
    coherence_mask : real-valued ndarray in [0, 1], shape (F, T)
    """
    eps = 1e-8
    mag_L = np.abs(X_L)
    mag_R = np.abs(X_R)

    # Phase alignment: cos(theta_L - theta_R)
    cross_real = np.real(X_L * np.conj(X_R))
    cos_phase  = cross_real / (mag_L * mag_R + eps)

    # Amplitude similarity: 2*|L|*|R| / (|L|^2 + |R|^2)
    amp_sim    = (2.0 * mag_L * mag_R) / (mag_L**2 + mag_R**2 + eps)

    # True panning & phase coherence
    gamma      = np.maximum(0.0, cos_phase) * amp_sim

    # Gate: only keep bins above the threshold
    coherence_mask = np.where(gamma > threshold, gamma, 0.0).astype(np.float32)
    return coherence_mask


# ---------------------------------------------------------------------------
# 5.  Combined Spectral Separation
# ---------------------------------------------------------------------------

def spectral_separate(
    stereo: np.ndarray,
    fs: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    wiener_power: float = 2.0,
    coherence_threshold: float = 0.70,
    eps: float = 1e-8,
) -> dict:
    """
    Full STFT-domain vocal/instrumental separation.

    Algorithm
    ---------
    1. Compute STFT for L and R channels.
    2. Decompose to Mid/Side in the STFT domain.
    3. Compute phase coherence mask C(k,m) - high where L≈R (vocal bins).
    4. Estimate vocal spectrum:  V̂ = |X_mid| · C
    5. Estimate music spectrum:  M̂ = |X_mid| · (1 - C)
    6. Build Wiener mask from V̂ and M̂.
    7. Apply combined mask = Wiener × C to the mid channel.
    8. ISTFT to reconstruct vocals and (by subtraction) instrumentals.

    Parameters
    ----------
    stereo               : ndarray, shape (2, N)
    fs                   : Sample rate in Hz
    n_fft                : STFT window size
    hop_length           : STFT hop size
    wiener_power         : Wiener mask exponent p
    coherence_threshold  : Phase coherence gate threshold
    eps                  : Regularisation floor

    Returns
    -------
    dict with keys:
        'vocals_stft'     : Complex STFT of estimated vocal track (mid ch)
        'music_stft'      : Complex STFT of estimated music track (mid ch)
        'mask'            : Combined soft mask (F, T)
        'X_L', 'X_R'     : Original channel STFTs
        'hop_length'      : Stored for ISTFT
        'n_fft'           : Stored for ISTFT
        'n_samples'       : Original sample count for length-exact ISTFT
    """
    L, R = stereo[0], stereo[1]
    n_samples = L.shape[0]

    print("[Spectral] Computing STFT ...")
    X_L = compute_stft(L, n_fft=n_fft, hop_length=hop_length)
    X_R = compute_stft(R, n_fft=n_fft, hop_length=hop_length)

    print("[Spectral] Mid-Side decomposition ...")
    X_mid, X_side = midside_decompose(X_L, X_R)

    print("[Spectral] Computing phase coherence mask ...")
    C = compute_phase_coherence_mask(X_L, X_R, threshold=coherence_threshold)

    # Estimated vocal energy = mid channel gated by coherence
    V_est = np.abs(X_mid) * C
    # Estimated music energy = mid channel with vocal bins suppressed
    M_est = np.abs(X_mid) * (1.0 - C) + np.abs(X_side)

    print("[Spectral] Computing Wiener soft mask ...")
    W = compute_wiener_mask(V_est, M_est, power=wiener_power, eps=eps)

    # Combined mask: high where coherent AND where mid energy dominates
    combined_mask = W * C

    # ── Per-channel separation preserving stereo field ──────────────────
    # CRITICAL FIX: Apply the vocal mask to the ORIGINAL L/R channel STFTs
    # instead of just the mid channel.  The old code discarded the entire
    # side channel (all stereo-panned instruments), causing the music
    # estimate to lose guitars, pads, and any wide-panned content.
    vocals_L_stft = combined_mask * X_L
    vocals_R_stft = combined_mask * X_R
    music_L_stft  = (1.0 - combined_mask) * X_L
    music_R_stft  = (1.0 - combined_mask) * X_R

    # Mono vocal estimate from mid channel — used as the NLMS reference
    # signal for targeted vocal cancellation in the adaptive stage.
    mag_mid   = np.abs(X_mid)
    phase_mid = np.exp(1j * np.angle(X_mid))
    vocal_mid_stft = (mag_mid * combined_mask) * phase_mid

    print("[Spectral] Separation complete.")
    return {
        "vocals_L_stft":  vocals_L_stft,
        "vocals_R_stft":  vocals_R_stft,
        "music_L_stft":   music_L_stft,
        "music_R_stft":   music_R_stft,
        "vocal_mid_stft": vocal_mid_stft,
        "mask":           combined_mask,
        "X_L":            X_L,
        "X_R":            X_R,
        "X_mid":          X_mid,
        "X_side":         X_side,
        "hop_length":     hop_length,
        "n_fft":          n_fft,
        "n_samples":      n_samples,
    }
