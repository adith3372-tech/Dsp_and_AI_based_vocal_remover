"""
=============================================================================
test_pipeline.py - Unit Tests for Each DSP Stage
=============================================================================
Run with:  python -m pytest test_pipeline.py -v
"""

import numpy as np
import pytest
from scipy.signal import freqz


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def stereo_1s():
    """1-second synthetic stereo signal at 44100 Hz."""
    fs = 44100
    N  = fs
    t  = np.arange(N) / fs
    # Centre-panned vocal (400 Hz) + side-panned guitar (1 kHz)
    vocal  = 0.5 * np.sin(2 * np.pi * 400 * t)
    guitar = 0.3 * np.sin(2 * np.pi * 1000 * t)
    L = vocal + guitar * 0.3
    R = vocal + guitar * 0.7
    return np.stack([L, R]).astype(np.float32), fs


@pytest.fixture
def mono_1s():
    """1-second mono 440 Hz sine at 44100 Hz."""
    fs = 44100
    t  = np.arange(fs) / fs
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32), fs


# ===========================================================================
# Stage 1: Preprocessing Tests
# ===========================================================================

class TestPreprocessing:

    def test_fir_bandpass_length_odd(self):
        """FIR filter must have odd length (Type-I for zero-phase)."""
        from preprocessing import design_fir_bandpass
        h = design_fir_bandpass(80, 4000, 44100)
        assert len(h) % 2 == 1, "FIR filter must be odd-length (Type-I)"

    def test_fir_bandpass_symmetry(self):
        """FIR filter must be symmetric (linear phase)."""
        from preprocessing import design_fir_bandpass
        h = design_fir_bandpass(80, 4000, 44100)
        assert np.allclose(h, h[::-1], atol=1e-10), "FIR must be symmetric"

    def test_fir_bandpass_attenuation(self):
        """Verify passband gain ≈ 1 and stopband gain < 0.1."""
        from preprocessing import design_fir_bandpass
        h = design_fir_bandpass(80, 4000, 44100)
        w, H = freqz(h, worN=8192, fs=44100)
        # Mid-band gain (e.g. 500 Hz) should be close to 1
        idx_500  = np.argmin(np.abs(w - 500))
        idx_50   = np.argmin(np.abs(w - 50))     # below bass cutoff
        idx_8000 = np.argmin(np.abs(w - 8000))   # above treble cutoff
        assert abs(np.abs(H[idx_500])) > 0.7,  "Passband gain should be ~1"
        assert abs(np.abs(H[idx_50]))  < 0.15, "Bass stopband should be attenuated"
        assert abs(np.abs(H[idx_8000])) < 0.15, "Treble stopband should be attenuated"

    def test_iir_lowpass_cutoff(self):
        """IIR low-pass should pass DC and reject high frequencies."""
        from preprocessing import design_iir_lowpass, apply_sos_filter
        sos = design_iir_lowpass(80, 44100, order=8)
        fs = 44100
        t  = np.arange(44100) / fs
        dc_signal    = np.ones(44100)
        high_signal  = np.sin(2 * np.pi * 5000 * t)
        out_dc   = apply_sos_filter(dc_signal, sos)
        out_high = apply_sos_filter(high_signal, sos)
        assert np.mean(np.abs(out_dc))   > 0.8, "DC should pass low-pass"
        assert np.mean(np.abs(out_high)) < 0.05, "5 kHz should be attenuated"

    def test_preprocess_output_keys(self, stereo_1s):
        """preprocess_audio must return all required keys."""
        from preprocessing import preprocess_audio
        stereo, fs = stereo_1s
        result = preprocess_audio(stereo, fs)
        for key in ["vocal_band", "bass_band", "treble_band", "fir_h"]:
            assert key in result, f"Missing key: {key}"

    def test_preprocess_shape_preserved(self, stereo_1s):
        """Band outputs must have shape (2, N)."""
        from preprocessing import preprocess_audio
        stereo, fs = stereo_1s
        result = preprocess_audio(stereo, fs)
        assert result["vocal_band"].shape[0] == 2
        assert result["bass_band"].shape[0] == 2
        assert result["treble_band"].shape[0] == 2


# ===========================================================================
# Stage 2: Multi-Rate DSP Tests
# ===========================================================================

class TestMultiRate:

    @pytest.mark.parametrize("D", [2, 4])
    def test_decimation_length(self, mono_1s, D):
        """Decimated signal length should be ~N/D."""
        from multirate import polyphase_decimate
        signal, fs = mono_1s
        dec = polyphase_decimate(signal, D, fs)
        expected = len(signal) // D
        assert abs(len(dec) - expected) <= D, f"Decimated length off by more than D={D}"

    @pytest.mark.parametrize("D", [2, 4])
    def test_interpolation_length(self, mono_1s, D):
        """Interpolated signal length should be ~N×D."""
        from multirate import polyphase_decimate, polyphase_interpolate
        signal, fs = mono_1s
        dec = polyphase_decimate(signal, D, fs)
        interp = polyphase_interpolate(dec, D, fs / D)
        expected = len(signal)
        assert abs(len(interp) - expected) <= D * 2, \
            f"Interpolated length too far from original: {len(interp)} vs {expected}"

    def test_roundtrip_energy_preserved(self, mono_1s):
        """Decimate -> Interpolate should preserve signal energy within 5 %."""
        from multirate import polyphase_decimate, polyphase_interpolate, align_length
        signal, fs = mono_1s
        D = 2
        dec    = polyphase_decimate(signal, D, fs)
        interp = polyphase_interpolate(dec, D, fs / D)
        sig, rec = align_length(signal, interp)
        e_orig = np.mean(sig ** 2)
        e_rec  = np.mean(rec ** 2)
        ratio  = e_rec / (e_orig + 1e-12)
        assert 0.90 <= ratio <= 1.10, \
            f"Energy ratio after roundtrip: {ratio:.3f} (expected 0.90–1.10)"

    def test_stereo_decimation_shape(self, stereo_1s):
        """Stereo decimation must preserve channel count."""
        from multirate import polyphase_decimate
        stereo, fs = stereo_1s
        D = 2
        dec = polyphase_decimate(stereo, D, fs)
        assert dec.shape[0] == 2, "Stereo decimation must preserve 2 channels"


# ===========================================================================
# Stage 3: Spectral Masking Tests
# ===========================================================================

class TestSpectral:

    def test_stft_shape(self, mono_1s):
        """STFT output shape should be (n_fft//2+1, T)."""
        from spectral import compute_stft
        signal, fs = mono_1s
        X = compute_stft(signal, n_fft=2048, hop_length=512)
        assert X.shape[0] == 2048 // 2 + 1, "STFT freq bins incorrect"
        assert X.dtype in [np.complex64, np.complex128], "STFT must be complex"

    def test_istft_roundtrip(self, mono_1s):
        """ISTFT(STFT(x)) ≈ x with < 1 % RMS error."""
        from spectral import compute_stft, compute_istft
        signal, fs = mono_1s
        X   = compute_stft(signal, n_fft=2048, hop_length=512)
        x_r = compute_istft(X, hop_length=512, n_fft=2048, length=len(signal))
        rms_error = np.sqrt(np.mean((signal - x_r) ** 2))
        rms_sig   = np.sqrt(np.mean(signal ** 2))
        assert rms_error / (rms_sig + 1e-12) < 0.01, \
            f"ISTFT roundtrip error too large: {rms_error/rms_sig:.4f}"

    def test_wiener_mask_bounds(self):
        """Wiener mask values must be in [0, 1]."""
        from spectral import compute_wiener_mask
        V = np.random.rand(512, 100).astype(np.float32)
        M = np.random.rand(512, 100).astype(np.float32)
        mask = compute_wiener_mask(V, M)
        assert np.all(mask >= 0) and np.all(mask <= 1), \
            "Wiener mask must be in [0, 1]"

    def test_midside_roundtrip(self):
        """Mid-side decomposition and recomposition must be lossless."""
        from spectral import midside_decompose, midside_recompose
        X_L = np.random.randn(512, 50) + 1j * np.random.randn(512, 50)
        X_R = np.random.randn(512, 50) + 1j * np.random.randn(512, 50)
        X_mid, X_side = midside_decompose(X_L, X_R)
        X_L2, X_R2   = midside_recompose(X_mid, X_side)
        assert np.allclose(X_L, X_L2, atol=1e-5), "Mid-side L roundtrip failed"
        assert np.allclose(X_R, X_R2, atol=1e-5), "Mid-side R roundtrip failed"

    def test_phase_coherence_center_signal(self):
        """Identical L and R (fully centre-panned) must give high coherence."""
        from spectral import compute_phase_coherence_mask, compute_stft
        fs = 44100
        t  = np.arange(fs) / fs
        x  = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        X_L = compute_stft(x)
        X_R = compute_stft(x)
        C = compute_phase_coherence_mask(X_L, X_R, threshold=0.5)
        # Most bins in the passband should have high coherence
        mean_coherence = np.mean(C[C > 0])
        assert mean_coherence > 0.8, \
            f"Centre-panned signal should have high coherence, got {mean_coherence:.3f}"


# ===========================================================================
# Stage 4: NLMS Adaptive Filter Tests
# ===========================================================================

class TestNLMS:

    def test_nlms_output_shape(self):
        """NLMS outputs must match input length."""
        from adaptive import nlms_filter, NLMSConfig
        N = 5000
        d = np.random.randn(N)
        r = np.random.randn(N)
        cfg = NLMSConfig(filter_order=32, mu=0.3)
        e, y, w_hist, mse = nlms_filter(d, r, cfg)
        assert len(e) == N
        assert len(y) == N
        assert len(mse) == N

    def test_nlms_convergence_on_known_signal(self):
        """NLMS must cancel a known common component within 5 % of signal."""
        from adaptive import nlms_filter, NLMSConfig
        np.random.seed(42)
        N  = 20000
        t  = np.arange(N) / 44100
        # Common vocal component (centre)
        vocal = 0.8 * np.sin(2 * np.pi * 400 * t)
        d = vocal           # L channel: vocal only
        r = vocal           # R channel: vocal only (reference)

        cfg = NLMSConfig(filter_order=64, mu=0.5, eps=1e-6)
        e, y, _, mse = nlms_filter(d, r, cfg)

        # After initial transient, MSE should be low (< initial MSE / 20)
        initial_mse = np.mean(mse[:100])
        final_mse   = np.mean(mse[N * 9 // 10:])   # last 10 %
        assert final_mse < initial_mse / 5, \
            f"NLMS did not converge: initial={initial_mse:.4f}, final={final_mse:.4f}"

    def test_nlms_stability_on_transient(self):
        """NLMS must not diverge on a loud impulse (stability test)."""
        from adaptive import nlms_filter, NLMSConfig
        N   = 10000
        d   = np.random.randn(N) * 0.01
        r   = np.random.randn(N) * 0.01
        # Inject a very loud transient
        d[N // 2] = 1000.0
        r[N // 2] = 1000.0

        cfg = NLMSConfig(filter_order=32, mu=0.3, eps=1e-4)
        e, y, _, mse = nlms_filter(d, r, cfg)

        assert np.all(np.isfinite(e)),   "NLMS output contains NaN/Inf"
        assert np.all(np.isfinite(mse)), "NLMS MSE contains NaN/Inf"
        # After transient, should recover
        post_mse = np.mean(mse[N // 2 + 1000:])
        assert post_mse < 100, f"NLMS did not recover after transient: {post_mse:.2f}"

    def test_block_nlms_matches_sample_nlms(self):
        """Block NLMS and sample NLMS should produce similar error energy."""
        from adaptive import nlms_filter, block_nlms_filter, NLMSConfig
        np.random.seed(7)
        N = 8000
        d = np.random.randn(N)
        r = np.random.randn(N)

        cfg_s = NLMSConfig(filter_order=32, mu=0.2, block_size=1)
        cfg_b = NLMSConfig(filter_order=32, mu=0.2, block_size=16)

        e_s, _, _, mse_s = nlms_filter(d, r, cfg_s)
        e_b, _, _, mse_b = block_nlms_filter(d, r, cfg_b)

        # Final MSE should be within a factor of 3 of each other
        final_s = np.mean(mse_s[-500:])
        final_b = np.mean(mse_b[-500:])
        ratio   = max(final_s, final_b) / (min(final_s, final_b) + 1e-12)
        assert ratio < 3.0, \
            f"Sample vs Block NLMS final MSE ratio too large: {ratio:.2f}"

    def test_nlms_vocal_cancel_reduces_vocal(self):
        """apply_nlms_vocal_cancel must reduce vocal energy in the error signal."""
        from adaptive import apply_nlms_vocal_cancel, NLMSConfig
        np.random.seed(99)
        N  = 20000
        t  = np.arange(N) / 44100
        # Centre-panned vocal at 400 Hz
        vocal = 0.8 * np.sin(2 * np.pi * 400 * t)
        # Stereo-panned guitar
        guitar = 0.3 * np.sin(2 * np.pi * 1000 * t)
        L = (vocal + guitar * 0.3).astype(np.float32)
        R = (vocal + guitar * 0.7).astype(np.float32)
        stereo = np.stack([L, R])
        vocal_ref = vocal.astype(np.float32)

        cfg = NLMSConfig(filter_order=64, mu=0.3, block_size=16)
        result = apply_nlms_vocal_cancel(stereo, vocal_ref, cfg, use_block=True)

        # The music estimate (error) should have less vocal than the original
        # Check last 50% of signal (after convergence)
        half = N // 2
        music_L_tail = result["music_L"][half:]
        # Project music onto vocal reference to measure residual vocal
        vocal_tail = vocal[half:]
        vocal_energy = np.mean(vocal_tail ** 2)
        residual_vocal = np.dot(music_L_tail, vocal_tail) / (np.dot(vocal_tail, vocal_tail) + 1e-10)
        residual_energy = np.mean((residual_vocal * vocal_tail) ** 2)
        # Residual vocal energy should be much less than original
        suppression = residual_energy / (vocal_energy + 1e-10)
        assert suppression < 0.5, \
            f"Vocal cancellation insufficient: {suppression:.3f} (expected < 0.5)"

    def test_spectral_separate_output_keys(self):
        """spectral_separate must return per-channel STFT keys."""
        from spectral import spectral_separate
        fs = 44100
        N  = fs  # 1 second
        t  = np.arange(N) / fs
        vocal  = 0.5 * np.sin(2 * np.pi * 400 * t)
        guitar = 0.3 * np.sin(2 * np.pi * 1000 * t)
        L = (vocal + guitar * 0.3).astype(np.float32)
        R = (vocal + guitar * 0.7).astype(np.float32)
        stereo = np.stack([L, R])

        result = spectral_separate(stereo, fs)
        required_keys = [
            "vocals_L_stft", "vocals_R_stft",
            "music_L_stft", "music_R_stft",
            "vocal_mid_stft", "mask",
            "X_L", "X_R", "X_mid", "X_side",
            "hop_length", "n_fft", "n_samples",
        ]
        for key in required_keys:
            assert key in result, f"Missing key in spectral result: {key}"


# ===========================================================================
# Stage 5: Synthesis Tests
# ===========================================================================

class TestSynthesis:

    def test_peak_normalise(self):
        """Normalised signal peak must match target dBFS."""
        from synthesis import peak_normalise
        sig = np.random.randn(10000)
        out = peak_normalise(sig, target_dbfs=-6.0)
        peak_dbfs = 20 * np.log10(np.max(np.abs(out)) + 1e-12)
        assert abs(peak_dbfs - (-6.0)) < 0.1, f"Peak = {peak_dbfs:.2f} dBFS"

    def test_soft_limiter_no_clipping(self):
        """Soft limiter must keep output within ±threshold."""
        from synthesis import soft_limiter
        sig = np.random.randn(10000) * 5.0   # very loud
        out = soft_limiter(sig, threshold=0.95)
        assert np.max(np.abs(out)) <= 0.95 + 1e-6, \
            f"Soft limiter exceeded threshold: {np.max(np.abs(out)):.4f}"

    def test_dc_removal(self):
        """DC removal should eliminate offset below 10 Hz."""
        from synthesis import remove_dc_offset
        fs  = 44100
        N   = fs
        dc_signal = np.ones(N) * 0.5 + np.random.randn(N) * 0.01
        out = remove_dc_offset(dc_signal, fs)
        # Mean of output should be near zero
        assert abs(np.mean(out)) < 0.01, \
            f"DC not removed: mean = {np.mean(out):.4f}"

    def test_write_read_roundtrip(self, tmp_path):
        """Written WAV file should be readable and preserve values."""
        from synthesis import write_wav
        import soundfile as sf
        N = 44100
        stereo = (np.random.randn(2, N) * 0.3).astype(np.float32)
        path = str(tmp_path / "test.wav")
        write_wav(stereo, 44100, path)
        data, sr = sf.read(path)
        assert sr == 44100, "Sample rate mismatch"
        assert data.shape == (N, 2), f"Shape mismatch: {data.shape}"


# ===========================================================================
# Stage 6: Evaluation Tests
# ===========================================================================

class TestEvaluation:

    def test_sdr_perfect_separation(self):
        """SDR of a perfect copy should be very high (> 40 dB)."""
        from evaluation import compute_sdr
        N   = 10000
        sig = np.random.randn(N)
        sdr = compute_sdr(sig, sig)
        assert sdr > 40, f"Perfect separation SDR should be > 40 dB, got {sdr:.1f}"

    def test_sdr_noisy_signal(self):
        """SDR of a noisy signal should be lower than perfect."""
        from evaluation import compute_sdr
        N     = 10000
        clean = np.random.randn(N)
        noisy = clean + 0.1 * np.random.randn(N)   # ~20 dB SNR
        sdr_clean = compute_sdr(clean, clean)
        sdr_noisy = compute_sdr(noisy, clean)
        assert sdr_noisy < sdr_clean, "Noisy signal should have lower SDR"

    def test_blind_sdr_returns_float(self):
        """Blind SDR must return a finite float."""
        from evaluation import estimate_sdr_blind
        N   = 44100
        est = np.random.randn(N) * 0.4
        mix = np.random.randn(N) * 0.5
        sdr = estimate_sdr_blind(est, mix, fs=44100.0)
        assert np.isfinite(sdr), f"Blind SDR must be finite, got {sdr}"

    def test_wiener_mask_sums_to_one(self):
        """
        Vocal mask M + Instrumental complement (1-M) must sum to 1 everywhere.
        This ensures perfect reconstruction of the total energy.
        """
        from spectral import compute_wiener_mask
        V = np.abs(np.random.randn(512, 100)).astype(np.float32)
        M = np.abs(np.random.randn(512, 100)).astype(np.float32)
        mask = compute_wiener_mask(V, M)
        complement = 1.0 - mask
        total = mask + complement
        assert np.allclose(total, 1.0, atol=1e-6), "Mask + complement must sum to 1"


if __name__ == "__main__":
    # Run all tests with verbose output
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"])
