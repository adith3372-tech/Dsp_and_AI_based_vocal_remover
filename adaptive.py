"""
=============================================================================
adaptive.py - NLMS Adaptive Filter for Residual Vocal Cancellation
=============================================================================

DSP Rationale:
    After STFT-domain masking, residual vocal "bleed" often remains in the
    instrumental track because the soft mask is not perfect.  An adaptive
    filter provides a second, complementary cancellation stage that operates
    in the time domain and can track non-stationary (time-varying) signals.

    Model
    -----
    We treat the RIGHT channel as the reference signal r(k) and the LEFT
    channel as the desired signal d(k).  The adaptive filter attempts to
    model the common vocal component - present in both channels - and
    subtract it.

        e(k) = d(k) - ŷ(k)
             = d(k) - w^H(k) · r(k)

    The error e(k) is the estimated "music-only" component in d(k).

    NLMS Update Rule
    ----------------
    The Normalised LMS (NLMS) rule divides the standard LMS step by the
    instantaneous power of the reference vector, giving a data-dependent
    step size.  This ensures the algorithm adapts faster when the input
    is loud and slower when quiet - important for music signals with large
    dynamic range.

        mu_eff(k) = mu / (||r(k)||² + eps)       ← normalised step size

        w(k+1) = w(k) + mu_eff(k) · e*(k) · r(k)

    Convergence & Stability
    -----------------------
    NLMS converges in the mean-square sense if  0 < mu < 2.
    Typically mu = 0.1 – 0.5 provides a good trade-off between convergence
    speed and steady-state misadjustment noise.

    The eps term (regularisation) prevents division by zero on silent frames
    and bounds the effective step size on loud transients:
        mu_eff ≤ mu / eps  (guaranteed upper bound)

    Leaky NLMS (optional)
    ---------------------
    A leakage factor λ < 1 is applied to w(k) before each update to prevent
    coefficient blow-up on long correlated signals:
        w(k+1) = λ·w(k) + mu_eff(k) · e*(k) · r(k)
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class NLMSConfig:
    """
    Configuration parameters for the NLMS adaptive filter.

    Attributes
    ----------
    filter_order : Length of the adaptive FIR filter (number of coefficients)
    mu           : NLMS step size mu ∈ (0, 2) for guaranteed convergence
    eps          : Regularisation constant eps to prevent zero-denominator
    leakage      : Leakage factor λ ∈ (0, 1]; 1.0 -> no leakage (standard NLMS)
    block_size   : Number of samples to process per update block (1 = sample-by-sample)
    """
    filter_order: int   = 256
    mu:           float = 0.3
    eps:          float = 1e-4
    leakage:      float = 1.0
    block_size:   int   = 1


# ---------------------------------------------------------------------------
# NLMS Core Implementation
# ---------------------------------------------------------------------------

def nlms_filter(
    d: np.ndarray,
    r: np.ndarray,
    cfg: NLMSConfig = None,
) -> tuple:
    """
    Run the NLMS adaptive filter over the full signal.

    Parameters
    ----------
    d   : Desired signal (left channel), shape (N,)
    r   : Reference signal (right channel), shape (N,)
    cfg : NLMSConfig instance; uses defaults if None

    Returns
    -------
    e          : Error signal ≈ music (vocal-cancelled), shape (N,)
    y          : Filter output ≈ vocal estimate, shape (N,)
    w_history  : Weight vector snapshots, shape (num_snapshots, filter_order)
                 Stored every 1000 samples for analysis.
    mse_curve  : Per-sample squared error |e(k)|², shape (N,)

    Notes
    -----
    This implementation is sample-by-sample (block_size=1), which gives the
    clearest illustration of the NLMS algorithm.  For production use, a
    block NLMS (sub-band NLMS) would be more efficient.
    """
    if cfg is None:
        cfg = NLMSConfig()

    N  = len(d)
    M  = cfg.filter_order
    mu = cfg.mu
    eps = cfg.eps
    lam = cfg.leakage

    # Initialise filter weights to zero (standard initialisation)
    w = np.zeros(M, dtype=np.float64)

    # Output buffers
    e         = np.zeros(N, dtype=np.float64)   # error / music estimate
    y         = np.zeros(N, dtype=np.float64)   # filter output / vocal estimate
    mse_curve = np.zeros(N, dtype=np.float64)   # instantaneous squared error

    # Weight snapshots - log every 'snapshot_interval' samples
    snapshot_interval = max(1, N // 200)
    w_history = []

    # Pre-fill a buffer for the reference signal
    r_buf = np.zeros(M, dtype=np.float64)

    print(f"[NLMS] Running adaptive filter: N={N}, order={M}, mu={mu}, eps={eps}")

    for k in range(N):
        # Shift reference buffer and insert latest sample
        # r_buf = [r(k), r(k-1), ..., r(k-M+1)]
        r_buf = np.roll(r_buf, 1)
        r_buf[0] = r[k]

        # Filter output: ŷ(k) = w^H · r(k)  (real signals -> w^T · r)
        y_k = np.dot(w, r_buf)

        # Error: e(k) = d(k) - ŷ(k)
        e_k = d[k] - y_k

        # NLMS normalised step size
        power = np.dot(r_buf, r_buf)        # ||r(k)||²
        mu_eff = mu / (power + eps)

        # Leaky NLMS weight update
        # w(k+1) = λ·w(k) + mu_eff · e(k) · r(k)
        # (For real signals, complex conjugate e*(k) reduces to e(k))
        w = lam * w + mu_eff * e_k * r_buf

        # Store outputs
        y[k]         = y_k
        e[k]         = e_k
        mse_curve[k] = e_k ** 2

        # Snapshot weight vector periodically
        if k % snapshot_interval == 0:
            w_history.append(w.copy())

    print(f"[NLMS] Done. Final MSE: {np.mean(mse_curve[-N//10:]):.6f}")
    return e, y, np.array(w_history), mse_curve


# ---------------------------------------------------------------------------
# Block-NLMS (Efficient Alternative)
# ---------------------------------------------------------------------------

def block_nlms_filter(
    d: np.ndarray,
    r: np.ndarray,
    cfg: NLMSConfig = None,
) -> tuple:
    """
    Block NLMS - processes signals in chunks of cfg.block_size samples.

    The weight vector is updated once per block using the time-averaged
    gradient across the block:

        w(k+1) = λ·w(k) + (mu/B) · Σ_{i∈block} e(i) · r_buf(i)

    This reduces the number of weight updates by factor B, lowering
    computational cost while retaining the same steady-state performance
    as sample-by-sample NLMS (for stationary signals).

    Parameters
    ----------
    d, r : Desired and reference signals, shape (N,)
    cfg  : NLMSConfig with block_size > 1

    Returns
    -------
    Same as nlms_filter: (e, y, w_history, mse_curve)
    """
    if cfg is None:
        cfg = NLMSConfig(block_size=64)

    N  = len(d)
    M  = cfg.filter_order
    B  = cfg.block_size
    mu = cfg.mu
    eps = cfg.eps
    lam = cfg.leakage

    w         = np.zeros(M, dtype=np.float64)
    e         = np.zeros(N, dtype=np.float64)
    y         = np.zeros(N, dtype=np.float64)
    mse_curve = np.zeros(N, dtype=np.float64)
    w_history = []

    snapshot_interval = max(1, (N // B) // 200)
    r_buf = np.zeros(M, dtype=np.float64)

    # Gradient accumulator for block update
    grad_acc = np.zeros(M, dtype=np.float64)
    power_acc = 0.0

    print(f"[Block-NLMS] Running: N={N}, order={M}, block={B}, mu={mu}")

    block_idx = 0
    for k in range(N):
        r_buf    = np.roll(r_buf, 1)
        r_buf[0] = r[k]

        y_k = np.dot(w, r_buf)
        e_k = d[k] - y_k

        y[k]         = y_k
        e[k]         = e_k
        mse_curve[k] = e_k ** 2

        # Accumulate gradient
        grad_acc  += e_k * r_buf
        power_acc += np.dot(r_buf, r_buf)
        block_idx += 1

        if block_idx == B:
            # Normalised block update
            mu_eff  = mu / (power_acc / B + eps)
            w       = lam * w + mu_eff * (grad_acc / B)
            grad_acc[:] = 0.0
            power_acc   = 0.0
            block_idx   = 0

            snap_k = k // B
            if snap_k % snapshot_interval == 0:
                w_history.append(w.copy())

    print(f"[Block-NLMS] Done. Final MSE: {np.mean(mse_curve[-N//10:]):.6f}")
    return e, y, np.array(w_history) if w_history else np.zeros((1, M)), mse_curve


# ---------------------------------------------------------------------------
# Stereo NLMS Wrapper
# ---------------------------------------------------------------------------

def apply_nlms_to_stereo(
    vocal_band: np.ndarray,
    cfg: NLMSConfig = None,
    use_block: bool = True,
) -> dict:
    """
    Apply NLMS adaptive cancellation to the vocal band stereo signal.

    The Left channel is treated as 'desired' d(k) and the Right channel
    as 'reference' r(k).  The error signal e(k) ≈ music component in L.

    We produce a stereo music estimate by pairing:
        music_L = e_L  (error from L desired, R reference)
        music_R = e_R  (error from R desired, L reference)

    and similarly for the vocal estimate.

    Parameters
    ----------
    vocal_band : ndarray, shape (2, N) - pre-filtered stereo vocal band
    cfg        : NLMSConfig
    use_block  : Use block NLMS (faster) vs sample-by-sample

    Returns
    -------
    dict with:
        'music_L', 'music_R' : Instrumental estimates per channel
        'vocal_L', 'vocal_R' : Vocal estimates per channel
        'mse_L', 'mse_R'     : Per-sample MSE curves for plotting
        'w_history_L'        : Weight history for L filter
    """
    if cfg is None:
        cfg = NLMSConfig()

    L = vocal_band[0].astype(np.float64)
    R = vocal_band[1].astype(np.float64)

    fn = block_nlms_filter if use_block else nlms_filter

    print("[NLMS] Processing Left channel (d=L, r=R) ...")
    e_L, y_L, w_hist_L, mse_L = fn(L, R, cfg)

    print("[NLMS] Processing Right channel (d=R, r=L) ...")
    e_R, y_R, w_hist_R, mse_R = fn(R, L, cfg)

    return {
        "music_L":     e_L,
        "music_R":     e_R,
        "vocal_L":     y_L,
        "vocal_R":     y_R,
        "mse_L":       mse_L,
        "mse_R":       mse_R,
        "w_history_L": w_hist_L,
        "w_history_R": w_hist_R,
    }


# ---------------------------------------------------------------------------
# Targeted Vocal Cancellation using STFT Vocal Estimate
# ---------------------------------------------------------------------------

def apply_nlms_vocal_cancel(
    vocal_band: np.ndarray,
    vocal_ref: np.ndarray,
    cfg: NLMSConfig = None,
    use_block: bool = True,
) -> dict:
    """
    Apply NLMS to cancel a known vocal estimate from each stereo channel.

    CRITICAL FIX: The old apply_nlms_to_stereo() used one audio channel to
    predict the other.  Since most energy in a stereo mix is common to both
    channels (not just vocals), the filter learned to predict EVERYTHING,
    driving the error signal (music estimate) toward zero.

    This function instead uses the STFT-derived mono vocal estimate as the
    reference signal.  The NLMS adapts this reference to match the actual
    vocal component in each channel, and the error signal is the residual
    music-only content:

        e_L(k) = L(k) - w_L^T(k) · vocal_ref(k)   ≈ music in L
        e_R(k) = R(k) - w_R^T(k) · vocal_ref(k)   ≈ music in R

    Parameters
    ----------
    vocal_band : ndarray, shape (2, N) - original stereo vocal band
    vocal_ref  : ndarray, shape (N,) - mono vocal estimate from STFT stage
    cfg        : NLMSConfig
    use_block  : Use block NLMS (faster) vs sample-by-sample

    Returns
    -------
    dict with:
        'music_L', 'music_R' : Instrumental estimates per channel (error)
        'vocal_L', 'vocal_R' : Vocal estimates per channel (filter output)
        'mse_L', 'mse_R'     : Per-sample MSE curves for plotting
        'w_history_L'        : Weight history for L filter
    """
    if cfg is None:
        cfg = NLMSConfig()

    L   = vocal_band[0].astype(np.float64)
    R   = vocal_band[1].astype(np.float64)
    ref = vocal_ref.astype(np.float64)

    fn = block_nlms_filter if use_block else nlms_filter

    print("[NLMS] Cancelling vocals from Left channel (d=L, r=vocal_ref) ...")
    e_L, y_L, w_hist_L, mse_L = fn(L, ref, cfg)

    print("[NLMS] Cancelling vocals from Right channel (d=R, r=vocal_ref) ...")
    e_R, y_R, w_hist_R, mse_R = fn(R, ref, cfg)

    return {
        "music_L":     e_L,
        "music_R":     e_R,
        "vocal_L":     y_L,
        "vocal_R":     y_R,
        "mse_L":       mse_L,
        "mse_R":       mse_R,
        "w_history_L": w_hist_L,
        "w_history_R": w_hist_R,
    }
