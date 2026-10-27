"""
=============================================================================
multirate.py - Multi-Rate DSP: Polyphase Decimation & Interpolation
=============================================================================

DSP Rationale:
    STFT-based spectral analysis of a full 44.1 kHz or 48 kHz signal is
    computationally expensive.  The vocal band (80 Hz – 4 kHz) is comfortably
    captured at a reduced sample rate (e.g. 11–12 kHz after 4× decimation).

    Polyphase decomposition avoids computing a full-rate filter output and then
    discarding D-1 out of every D samples - instead it rearranges the filter
    into D sub-filters that each operate at the lower rate directly.

    For a length-L FIR filter h[n], the k-th polyphase component is:
        E_k[n] = h[nD + k],  k = 0, 1, ..., D-1

    Decimation  : y[n] = Σ_k E_k[n] * x[nD - k]  (computed at low rate)
    Interpolation: the Noble Identity means we can move the upsample before
                   the filter bank and run filters at the high rate efficiently.

    For simplicity and reliability, this implementation wraps
    scipy.signal.decimate / resample_poly, which internally uses polyphase
    filtering.  The anti-alias filter is applied explicitly *before* calling
    decimate so the user has full visibility and control over it.
"""

import numpy as np
from scipy.signal import decimate, resample_poly, firwin


# ---------------------------------------------------------------------------
# Helper: Build a Polyphase Anti-Alias FIR
# ---------------------------------------------------------------------------

def _make_polyphase_fir(D: int, fs: float, num_taps_per_phase: int = 32) -> np.ndarray:
    """
    Build the prototype low-pass FIR used in polyphase decomposition.

    The prototype filter cuts off at f_c = 0.9 / D (normalised, 0–1 scale)
    to leave a 10 % guard band against aliasing.

    Total taps = num_taps_per_phase × D (rounded up to odd).

    Parameters
    ----------
    D                 : Decimation / interpolation factor
    fs               : Original sample rate (used only for logging)
    num_taps_per_phase: Taps per polyphase branch; more -> sharper roll-off

    Returns
    -------
    h : ndarray - prototype FIR coefficients
    """
    total_taps = num_taps_per_phase * D
    if total_taps % 2 == 0:
        total_taps += 1            # enforce Type-I (odd-length)

    cutoff_norm = 0.9 / D         # normalised cut-off (Nyquist = 1.0)
    h = firwin(total_taps, cutoff_norm, window="hamming")
    return h


# ---------------------------------------------------------------------------
# Polyphase Decimation
# ---------------------------------------------------------------------------

def polyphase_decimate(
    signal: np.ndarray,
    D: int,
    fs: float,
    sos_antialiase=None,
) -> np.ndarray:
    """
    Downsample `signal` by integer factor D using polyphase filtering.

    Steps
    -----
    1. If the caller already applied an anti-alias filter (sos_antialiase is
       not None), we skip the internal anti-alias step.  Otherwise, we build
       and apply a prototype polyphase FIR here.
    2. Resample using scipy.signal.resample_poly(signal, 1, D) which
       implements the Noble-Identity polyphase structure internally.

    The output sample rate is:  fs_out = fs / D

    Parameters
    ----------
    signal        : ndarray, shape (samples,) or (2, samples)
    D             : Integer decimation factor (must be >= 2)
    fs            : Input sample rate in Hz
    sos_antialiase: Pre-computed SOS anti-alias filter (optional).  If given,
                    it has already been applied upstream; skip internal filter.

    Returns
    -------
    decimated : ndarray, same number of channels but N/D samples
    """
    if D < 2:
        raise ValueError(f"Decimation factor D must be >= 2, got {D}")

    # Build prototype FIR for polyphase structure
    h_proto = _make_polyphase_fir(D, fs)

    if signal.ndim == 1:
        # resample_poly applies polyphase FIR internally with the given filter
        return resample_poly(signal, 1, D, window=h_proto)
    else:
        channels = [
            resample_poly(signal[ch], 1, D, window=h_proto)
            for ch in range(signal.shape[0])
        ]
        return np.stack(channels)


# ---------------------------------------------------------------------------
# Polyphase Interpolation
# ---------------------------------------------------------------------------

def polyphase_interpolate(
    signal: np.ndarray,
    D: int,
    fs_low: float,
) -> np.ndarray:
    """
    Upsample `signal` by integer factor D using polyphase filtering.

    This is the exact inverse of polyphase_decimate.  The prototype FIR
    acts as an interpolation (reconstruction) filter: after inserting D-1
    zeros between each sample, the FIR smoothes the result.

    In the polyphase interpretation:
        y[n] = Σ_k E_k[n//D] * δ(n mod D - k)
    which avoids computing the zeros explicitly.

    The output sample rate is:  fs_out = fs_low × D

    Parameters
    ----------
    signal : ndarray, shape (samples,) or (2, samples) at fs_low
    D      : Integer interpolation factor
    fs_low : Sample rate of the input signal in Hz

    Returns
    -------
    interpolated : ndarray with D × more samples at fs_low × D
    """
    if D < 2:
        raise ValueError(f"Interpolation factor D must be >= 2, got {D}")

    fs_high = fs_low * D
    h_proto = _make_polyphase_fir(D, fs_high)

    if signal.ndim == 1:
        return resample_poly(signal, D, 1, window=h_proto)
    else:
        channels = [
            resample_poly(signal[ch], D, 1, window=h_proto)
            for ch in range(signal.shape[0])
        ]
        return np.stack(channels)


# ---------------------------------------------------------------------------
# Length-Alignment Utility
# ---------------------------------------------------------------------------

def align_length(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Trim both arrays to the same length along the last axis.

    Necessary after polyphase operations which may introduce minor
    off-by-one differences depending on boundary conditions.

    Parameters
    ----------
    a, b : ndarrays with equal ndim, shapes differing only in last dim

    Returns
    -------
    (a_trimmed, b_trimmed) : both share the same last-axis length
    """
    n = min(a.shape[-1], b.shape[-1])
    if a.ndim == 1:
        return a[:n], b[:n]
    return a[:, :n], b[:, :n]
