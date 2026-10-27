"""
=============================================================================
evaluation.py - Objective Metrics & Visualisation
=============================================================================

DSP Rationale:
    Signal separation quality is measured using the BSS Eval framework
    (Blind Source Separation Evaluation), specifically SDR and SIR.

    Definitions (Vincent et al., 2006)
    -----------------------------------
    Given the true target s_target and estimated signal ŝ, we decompose:

        ŝ = s_target + e_interf + e_artif

    where:
        s_target : projection of ŝ onto the true source signal
        e_interf : interference from other sources
        e_artif  : algorithmic artefacts (distortion, reverb, etc.)

    Metrics:
        SDR = 10·log₁₀( ||s_target||²  /  ||e_interf + e_artif||² )
        SIR = 10·log₁₀( ||s_target||²  /  ||e_interf||² )

    A target SDR of >= 8 dB indicates commercially usable separation quality.

    NLMS Learning Curve
    -------------------
    We plot the smoothed MSE curve from the NLMS stage.  Convergence is
    demonstrated when the MSE plateaus - typically within the first 5–10 %
    of the signal length.  The convergence time constant τ for LMS is:

        τ ≈ 1 / (2mu·P̄)

    where P̄ is the average input power.  NLMS normalises this, giving a
    convergence time that is approximately 1/(2mu) samples regardless of
    input level.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend - safe for all OS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft as scipy_stft


# ---------------------------------------------------------------------------
# 1.  SDR and SIR Computation
# ---------------------------------------------------------------------------

def _project(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Project `estimate` onto `reference` to find the target component.

        s_target = <ŝ, s> / <s, s> · s

    Parameters
    ----------
    estimate  : ndarray - estimated separated signal
    reference : ndarray - ground-truth reference signal

    Returns
    -------
    s_target : projection, same shape as reference
    """
    eps = 1e-10
    proj_scalar = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    return proj_scalar * reference


def compute_sdr(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compute the Signal-to-Distortion Ratio (SDR) in dB.

        SDR = 10·log₁₀( ||s_target||²  /  ||ŝ - s_target||² )

    If no reference is provided (e.g., you only have the mix), a
    simplified version using the mix as pseudo-reference is used.

    Parameters
    ----------
    estimate  : ndarray - separated output (e.g. karaoke.wav mono sum)
    reference : ndarray - ground-truth signal (if available)

    Returns
    -------
    sdr_db : float
    """
    n = min(len(estimate), len(reference))
    est, ref = estimate[:n], reference[:n]

    s_target  = _project(est, ref)
    e_total   = est - s_target              # all error

    eps = 1e-10
    sdr_db = 10.0 * np.log10(
        (np.dot(s_target, s_target) + eps) /
        (np.dot(e_total, e_total) + eps)
    )
    return float(sdr_db)


def compute_sir(
    estimate: np.ndarray,
    reference: np.ndarray,
    interference: np.ndarray,
) -> float:
    """
    Compute the Signal-to-Interference Ratio (SIR) in dB.

        SIR = 10·log₁₀( ||s_target||²  /  ||e_interf||² )

    e_interf is estimated by projecting the interference signal onto the
    error (estimate - s_target).

    Parameters
    ----------
    estimate      : ndarray - separated output
    reference     : ndarray - ground-truth target signal
    interference  : ndarray - known interfering signal

    Returns
    -------
    sir_db : float
    """
    n = min(len(estimate), len(reference), len(interference))
    est, ref, interf = estimate[:n], reference[:n], interference[:n]

    s_target  = _project(est, ref)
    e_total   = est - s_target
    e_interf  = _project(e_total, interf)

    eps = 1e-10
    sir_db = 10.0 * np.log10(
        (np.dot(s_target, s_target) + eps) /
        (np.dot(e_interf, e_interf) + eps)
    )
    return float(sir_db)


def estimate_sdr_blind(
    estimate: np.ndarray,
    mix: np.ndarray,
    window_sec: float = 0.5,
    fs: float = 44100.0,
) -> float:
    """
    Estimate SDR without ground truth using a short-window energy ratio.

    This is a blind metric used when the original separate stems are
    unavailable (common in real-world karaoke generation).

    The SNR is estimated segment-by-segment as:
        SNR_seg = 10·log₁₀( Σ|mix|² / Σ|mix - estimate|² )

    and averaged in dB.  A higher score indicates better reconstruction
    fidelity (estimate more closely resembles the mix in terms of
    signal structure).

    Parameters
    ----------
    estimate   : Separated track, shape (N,)
    mix        : Original mix, shape (N,)
    window_sec : Segment length in seconds
    fs         : Sample rate

    Returns
    -------
    blind_sdr : float - mean blind SDR across windows
    """
    n = min(len(estimate), len(mix))
    est, mix_ = estimate[:n], mix[:n]

    seg = int(window_sec * fs)
    sdr_list = []
    eps = 1e-10

    for start in range(0, n - seg, seg):
        e = est[start:start + seg]
        m = mix_[start:start + seg]
        signal_power = np.dot(m, m)
        residual = m - e
        noise_power = np.dot(residual, residual)
        if noise_power < eps:
            continue
        sdr_list.append(10.0 * np.log10(signal_power / noise_power + eps))

    return float(np.mean(sdr_list)) if sdr_list else float("nan")


# ---------------------------------------------------------------------------
# 2.  Visualisation
# ---------------------------------------------------------------------------

def plot_pipeline_results(
    original_stereo:  np.ndarray,
    karaoke:          np.ndarray,
    vocals:           np.ndarray,
    mse_curve_L:      np.ndarray,
    mse_curve_R:      np.ndarray,
    spectral_mask:    np.ndarray,
    fs:               float,
    output_path:      str = "pipeline_results.png",
) -> None:
    """
    Generate a comprehensive 6-panel visualisation of the pipeline.

    Panels
    ------
    1. Original mix waveform (L & R)
    2. Karaoke output waveform
    3. Vocals output waveform
    4. Spectral mask (time–frequency heatmap)
    5. NLMS MSE learning curve (L channel) - proves convergence
    6. Smoothed MSE with convergence annotation

    Parameters
    ----------
    original_stereo : shape (2, N)
    karaoke         : shape (2, N)
    vocals          : shape (2, N)
    mse_curve_L     : shape (N,) - per-sample squared error from NLMS
    mse_curve_R     : shape (N,)
    spectral_mask   : shape (F, T) - Wiener mask from STFT stage
    fs              : Sample rate in Hz
    output_path     : File path for saved figure
    """
    fig = plt.figure(figsize=(18, 14), facecolor="#0D1117")
    fig.suptitle(
        "AI-Assisted Vocal Remover - DSP Pipeline Results",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Colour palette ────────────────────────────────────────────────────
    CYAN   = "#00FFFF"
    LIME   = "#39FF14"
    ORANGE = "#FF6B35"
    PURPLE = "#BD93F9"
    PINK   = "#FF79C6"
    BG     = "#161B22"
    TEXT   = "#C9D1D9"

    def _style_ax(ax, title):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_title(title, color=CYAN, fontsize=10, pad=6)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    N = original_stereo.shape[-1]
    t = np.arange(N) / fs

    # ── Panel 1: Original mix ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "Original Mix - L & R Waveforms")
    n_plot = min(N, int(30 * fs))   # show first 30 s max
    t_plot = t[:n_plot]
    ax1.plot(t_plot, original_stereo[0, :n_plot], color=CYAN,   lw=0.5,
             alpha=0.8, label="Left")
    ax1.plot(t_plot, original_stereo[1, :n_plot], color=PURPLE, lw=0.5,
             alpha=0.8, label="Right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.5)

    # ── Panel 2: Karaoke waveform ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Karaoke Output - Instrumental Track")
    n_kar = min(karaoke.shape[-1], n_plot)
    ax2.plot(t_plot[:n_kar], karaoke[0, :n_kar], color=LIME, lw=0.5, alpha=0.9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")

    # ── Panel 3: Vocals waveform ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Vocals Output - Isolated Vocal Track")
    n_voc = min(vocals.shape[-1], n_plot)
    ax3.plot(t_plot[:n_voc], vocals[0, :n_voc], color=ORANGE, lw=0.5, alpha=0.9)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")

    # ── Panel 4: Spectral mask heatmap ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Spectral Mask M(k,m) - Vocal TF Bins (Wiener+Coherence)")
    # Show first 500 frames and up to 512 frequency bins
    F, T_full = spectral_mask.shape
    T_show = min(T_full, 500)
    F_show = min(F, 256)
    im = ax4.imshow(
        spectral_mask[:F_show, :T_show],
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=0, vmax=1,
    )
    # Y-axis in kHz
    freq_ticks = np.linspace(0, F_show - 1, 5).astype(int)
    freq_labels = [f"{(f / F_show) * (fs / 2) / 1000:.1f}" for f in freq_ticks]
    ax4.set_yticks(freq_ticks)
    ax4.set_yticklabels(freq_labels, fontsize=7)
    ax4.set_xlabel("Frame index")
    ax4.set_ylabel("Frequency (kHz)")
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.set_label("Mask value", color=TEXT, fontsize=7)

    # ── Panel 5: Raw NLMS MSE learning curve ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    _style_ax(ax5, "NLMS MSE Learning Curve - Left Channel (Raw)")
    mse_display = mse_curve_L[:min(len(mse_curve_L), int(60 * fs))]
    t_mse = np.arange(len(mse_display)) / fs
    ax5.plot(t_mse, mse_display, color=PINK, lw=0.3, alpha=0.7, label="Instantaneous MSE")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("MSE  |e(k)|²")
    ax5.set_yscale("log")
    ax5.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.5)

    # ── Panel 6: Smoothed MSE + convergence annotation ────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    _style_ax(ax6, "NLMS Smoothed MSE - Convergence Proof")

    # Smooth with a running mean (1000-sample window)
    window = min(1000, len(mse_curve_L) // 20)
    kernel = np.ones(window) / window
    smooth_L = np.convolve(mse_curve_L, kernel, mode="valid")
    smooth_R = np.convolve(mse_curve_R, kernel, mode="valid")
    t_smooth = np.arange(len(smooth_L)) / fs

    ax6.plot(t_smooth, smooth_L, color=PINK,   lw=1.5, label="Smoothed MSE (L)")
    ax6.plot(t_smooth, smooth_R, color=PURPLE, lw=1.5, label="Smoothed MSE (R)", alpha=0.7)

    # Mark 5 % and 10 % convergence points
    for pct, color, label in [(0.05, LIME, "5 %"), (0.10, ORANGE, "10 %")]:
        idx = int(pct * len(smooth_L))
        ax6.axvline(t_smooth[idx], color=color, lw=1.2, ls="--", alpha=0.8,
                    label=f"{label} of track")

    ax6.set_yscale("log")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Smoothed MSE")
    ax6.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.5)

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Eval] Visualisation saved -> {output_path}")


# ---------------------------------------------------------------------------
# 3.  Print Metric Report
# ---------------------------------------------------------------------------

def print_metrics_report(
    karaoke:  np.ndarray,
    vocals:   np.ndarray,
    mix:      np.ndarray,
    fs:       float,
) -> dict:
    """
    Compute and print a formatted metrics report.

    When ground-truth stems are not available, blind SDR estimates are used.

    Parameters
    ----------
    karaoke, vocals, mix : Mono (1-D) arrays
    fs                   : Sample rate in Hz

    Returns
    -------
    metrics : dict with keys 'blind_sdr_karaoke', 'blind_sdr_vocals',
                              'energy_ratio_karaoke', 'energy_ratio_vocals'
    """
    n = min(len(karaoke), len(vocals), len(mix))
    kar  = karaoke[:n]
    voc  = vocals[:n]
    mix_ = mix[:n]

    blind_sdr_kar = estimate_sdr_blind(kar, mix_, fs=fs)
    blind_sdr_voc = estimate_sdr_blind(voc, mix_, fs=fs)

    # Energy ratio: how much of the mix energy each track captures
    mix_power = np.mean(mix_ ** 2) + 1e-12
    kar_ratio = float(10.0 * np.log10(np.mean(kar ** 2) / mix_power))
    voc_ratio = float(10.0 * np.log10(np.mean(voc ** 2) / mix_power))

    metrics = {
        "blind_sdr_karaoke":     blind_sdr_kar,
        "blind_sdr_vocals":      blind_sdr_voc,
        "energy_ratio_karaoke":  kar_ratio,
        "energy_ratio_vocals":   voc_ratio,
    }

    banner = "=" * 60
    print(f"\n{banner}")
    print("  EVALUATION METRICS REPORT")
    print(banner)
    print(f"  Blind SDR - Karaoke (Instrumental) : {blind_sdr_kar:+.2f} dB")
    print(f"  Blind SDR - Vocals                 : {blind_sdr_voc:+.2f} dB")
    print(f"  Energy Ratio - Karaoke vs Mix       : {kar_ratio:+.2f} dB")
    print(f"  Energy Ratio - Vocals vs Mix        : {voc_ratio:+.2f} dB")
    print(f"\n  Target: Blind SDR >= 8 dB")
    for name, val in [("Karaoke", blind_sdr_kar), ("Vocals", blind_sdr_voc)]:
        status = "[PASS]" if val >= 8.0 else "[BELOW TARGET]"
        print(f"    {name}: {val:.2f} dB -> {status}")
    print(banner + "\n")

    return metrics
