"""
=============================================================================
preprocessing.py - Pre-Processing & Filtering Stage
=============================================================================

DSP Rationale:
    Vocals occupy roughly 80 Hz–4 kHz in a typical mix.  Everything outside
    that band (sub-bass kick, hi-hats, cymbals) is largely instrument energy.
    By splitting the signal into three frequency zones *before* the separation
    stage, we avoid accidentally removing kick/sub-bass (center-panned but NOT
    vocal) while also preserving the air/presence above 4 kHz.

    Flow:
        raw stereo ──► [anti-alias LPF] ──► [polyphase decimation]
                                                         │
              ┌──────────────────────────────────────────┘
              │
              ├─ bass_band  (0 – 80 Hz)   : simple sum, passed straight to output
              ├─ vocal_band (80 Hz – 4 kHz): handed to STFT + NLMS pipeline
              └─ treble_band (4 kHz – Nyq): simple sum, passed straight to output
"""

import numpy as np
from scipy.signal import firwin, butter, sosfilt, sosfiltfilt, kaiserord


# ---------------------------------------------------------------------------
# 1.  FIR Band-Pass Filter - Vocal Frequency Range
# ---------------------------------------------------------------------------

def design_fir_bandpass(
    f_low: float,
    f_high: float,
    fs: float,
    ripple_db: float = 60.0,
    width_hz: float = 40.0,
) -> np.ndarray:
    """
    Design a Type-I linear-phase FIR band-pass filter using a Kaiser window.

    The Kaiser window is preferred over Hamming/Hann here because it gives
    direct control over both the stop-band attenuation and the transition-band
    width via the Kaiser β parameter, making filter order predictable.

    Math background
    ---------------
    The ideal band-pass impulse response is the difference of two sinc kernels:

        h_bp[n] = h_lp_high[n] - h_lp_low[n]
                = 2·f_high·sinc(2·f_high·n) - 2·f_low·sinc(2·f_low·n)

    The Kaiser window w[n] = I₀(β√(1-(2n/M-1)²)) / I₀(β) is applied
    to truncate the ideal infinite-length filter.

    Parameters
    ----------
    f_low    : Lower cut-off frequency in Hz (e.g. 80 Hz)
    f_high   : Upper cut-off frequency in Hz (e.g. 4000 Hz)
    fs       : Sampling rate in Hz
    ripple_db: Minimum stop-band attenuation in dB (>= 21 dB)
    width_hz : Width of the transition band in Hz

    Returns
    -------
    h : ndarray, shape (N+1,)
        FIR filter coefficients (symmetric, Type-I).
    """
    # Normalise transition width relative to Nyquist (0–1 scale)
    nyq = fs / 2.0
    width_norm = width_hz / nyq

    # kaiserord returns (N, beta) that satisfy the attenuation and width specs
    N, beta = kaiserord(ripple_db, width_norm)
    # Ensure odd filter length (Type-I: linear phase, zero-phase capable)
    if N % 2 == 0:
        N += 1

    # firwin with two cut-offs produces a band-pass when pass_zero=False
    h = firwin(
        N,
        [f_low / nyq, f_high / nyq],
        window=("kaiser", beta),
        pass_zero=False,   # band-pass (not low-pass)
    )
    return h


# ---------------------------------------------------------------------------
# 2.  IIR Butterworth Filters - Bass / Treble Separation
# ---------------------------------------------------------------------------

def design_iir_lowpass(cutoff_hz: float, fs: float, order: int = 8) -> np.ndarray:
    """
    Design a Butterworth low-pass IIR filter (SOS form).

    Butterworth is chosen for its maximally flat passband - important for bass
    reproduction where amplitude coloration is undesirable.

    The Butterworth polynomial poles are at angles:
        θ_k = π(2k + n - 1) / (2n),  k = 1, ..., n

    Parameters
    ----------
    cutoff_hz : -3 dB cut-off frequency in Hz
    fs        : Sampling rate in Hz
    order     : Filter order (higher -> steeper roll-off, more group delay)

    Returns
    -------
    sos : ndarray, shape (order/2, 6)
        Second-order sections representation for numerically stable filtering.
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sos


def design_iir_highpass(cutoff_hz: float, fs: float, order: int = 8) -> np.ndarray:
    """
    Design a Butterworth high-pass IIR filter (SOS form).

    Parameters
    ----------
    cutoff_hz : -3 dB cut-off frequency in Hz
    fs        : Sampling rate in Hz
    order     : Filter order

    Returns
    -------
    sos : ndarray, shape (order/2, 6)
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sos


# ---------------------------------------------------------------------------
# 3.  Anti-Alias Filter - Required Before Decimation
# ---------------------------------------------------------------------------

def design_antialiasing_filter(
    decimation_factor: int,
    fs: float,
    order: int = 10,
) -> np.ndarray:
    """
    Design a low-pass anti-aliasing IIR filter for pre-decimation.

    Before downsampling by factor D, any frequency component above
    f_s / (2D) will alias back into the baseband and corrupt the signal.
    The anti-alias filter attenuates these components below a safe threshold.

    Cutoff is set conservatively at 0.9 × Nyquist_new = 0.9 × fs / (2D)
    to leave a small guard band.

    Parameters
    ----------
    decimation_factor : Integer downsampling ratio D
    fs               : Original sampling rate in Hz
    order            : IIR filter order

    Returns
    -------
    sos : ndarray - SOS coefficients of the anti-alias filter
    """
    new_nyq = fs / (2.0 * decimation_factor)
    cutoff = 0.9 * new_nyq          # 10 % guard band
    sos = butter(order, cutoff / (fs / 2.0), btype="low", output="sos")
    return sos


# ---------------------------------------------------------------------------
# 4.  Apply Filters to a Stereo Signal
# ---------------------------------------------------------------------------

def apply_fir_filter(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Apply an FIR filter to a 1-D or 2-D (channels × samples) signal
    using overlap-save via scipy's fftconvolve (implicitly).

    We use sosfiltfilt (zero-phase) for the IIR filters below, but for the
    FIR we use direct-form convolution via np.convolve on each channel.
    For long signals this is handled chunk-by-chunk inside numpy's FFT backend.

    Parameters
    ----------
    signal : ndarray, shape (samples,) or (2, samples)
    h      : FIR coefficient vector

    Returns
    -------
    filtered : ndarray, same shape as signal
    """
    from scipy.signal import fftconvolve

    if signal.ndim == 1:
        return fftconvolve(signal, h, mode="same")
    else:
        return np.stack(
            [fftconvolve(signal[ch], h, mode="same") for ch in range(signal.shape[0])]
        )


def apply_sos_filter(
    signal: np.ndarray,
    sos: np.ndarray,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply an IIR SOS filter to a 1-D or 2-D (channels × samples) signal.

    Zero-phase filtering (filtfilt) is used by default to avoid group-delay
    distortion that would misalign bass and treble bands relative to the vocal
    band - critical for correct band summation at the output stage.

    Parameters
    ----------
    signal     : ndarray, shape (samples,) or (2, samples)
    sos        : SOS filter coefficients
    zero_phase : If True, use sosfiltfilt; otherwise causal sosfilt.

    Returns
    -------
    filtered : ndarray, same shape as signal
    """
    fn = sosfiltfilt if zero_phase else sosfilt

    if signal.ndim == 1:
        return fn(sos, signal)
    else:
        return np.stack([fn(sos, signal[ch]) for ch in range(signal.shape[0])])


# ---------------------------------------------------------------------------
# 5.  Full Pre-Processing Pipeline
# ---------------------------------------------------------------------------

def preprocess_audio(
    stereo: np.ndarray,
    fs: float,
    f_vocal_low: float = 80.0,
    f_vocal_high: float = 4000.0,
) -> dict:
    """
    Execute the complete pre-processing stage.

    Returns a dictionary with:
        'vocal_band'  – band-passed signal fed to the separation pipeline
        'bass_band'   – low-passed below f_vocal_low (bypasses separation)
        'treble_band' – high-passed above f_vocal_high (bypasses separation)
        'fir_h'       – the FIR coefficients (for inspection/logging)

    The band splitting uses complementary filters so that:
        bass_band + vocal_band + treble_band ≈ original  (modulo transition bands)

    Parameters
    ----------
    stereo       : ndarray, shape (2, N) - left/right channels
    fs           : Sampling rate in Hz
    f_vocal_low  : Lower vocal band edge in Hz
    f_vocal_high : Upper vocal band edge in Hz
    """
    print("[PreProcess] Designing FIR band-pass filter ...")
    fir_h = design_fir_bandpass(f_vocal_low, f_vocal_high, fs)

    print("[PreProcess] Designing IIR low-pass (bass) filter ...")
    sos_lp = design_iir_lowpass(f_vocal_low, fs, order=8)

    print("[PreProcess] Designing IIR high-pass (treble) filter ...")
    sos_hp = design_iir_highpass(f_vocal_high, fs, order=8)

    print("[PreProcess] Applying filters ...")
    vocal_band  = apply_fir_filter(stereo, fir_h)
    bass_band   = apply_sos_filter(stereo, sos_lp)
    treble_band = apply_sos_filter(stereo, sos_hp)

    print(f"[PreProcess] Done - signal shape {stereo.shape}, fs={fs} Hz")
    return {
        "vocal_band":   vocal_band,
        "bass_band":    bass_band,
        "treble_band":  treble_band,
        "fir_h":        fir_h,
    }
