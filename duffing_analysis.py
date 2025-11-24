import numpy as np
from scipy.integrate import odeint
from scipy.signal import fftconvolve
import pandas as pd

class DuffingAnalysis:
    """
    The Universal Engine for Blind Nonlinear System Identification.
    Implements the 'Ritual of the Hidden Ridge'.
    """

    @staticmethod
    def duffing_oscillator(state, t, zeta, omega0, alpha):
        """
        Equation of motion: x'' + 2*zeta*w0*x' + w0^2*x + alpha*x^3 = 0
        """
        x, v = state
        dxdt = v
        dvdt = -2.0 * zeta * omega0 * v - (omega0**2) * x - alpha * (x**3)
        return [dxdt, dvdt]

    @staticmethod
    def simulate_signal(alpha, snr_db, seed=42, duration=4.0, fs=1000.0, f0=10.0, zeta=0.01, x0=[1.0, 0.0]):
        """
        Generates a synthetic Duffing signal buried in 1/f^2 noise.
        """
        np.random.seed(seed)
        t = np.arange(0.0, duration, 1.0/fs)
        omega0 = 2.0 * np.pi * f0

        # 1. Physics (The Signal)
        sol = odeint(DuffingAnalysis.duffing_oscillator, x0, t, args=(zeta, omega0, alpha))
        x_clean = sol[:, 0]

        # 2. The Void (1/f^2 Brownian Noise)
        # Brownian noise is integrated white noise
        white = np.random.normal(0, 1, len(t))
        brown = np.cumsum(white)

        # Normalize to target SNR
        # SNR = 10 * log10(P_signal / P_noise)
        sig_power = np.mean(x_clean**2)
        noise_power = np.mean(brown**2)

        if noise_power > 0:
            target_noise_power = sig_power / (10**(snr_db/10))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = scale * brown
        else:
            noise = 0

        x_noisy = x_clean + noise

        return t, x_noisy, omega0

    @staticmethod
    def morlet2_manual(M, s, w=5.0):
        """
        Complex Morlet wavelet, centered at zero.
        Matches scipy.signal.morlet2 definition.
        """
        x = np.arange(0, M) - (M - 1.0) / 2
        x = x / s
        output = np.pi**(-0.25) * np.exp(1j * w * x) * np.exp(-0.5 * x**2)
        return output

    @staticmethod
    def cwt_manual(data, widths, w=6.0):
        """
        Continuous wavelet transform using fftconvolve.
        """
        output = np.zeros((len(widths), len(data)), dtype=np.complex128)
        for i, width in enumerate(widths):
            # Determine wavelet length (sufficient to capture the decay)
            # 10 * s is a good heuristic for Morlet
            M = int(min(10 * width, len(data)))
            if M < 1: M = 1
            # Ensure odd length for centering
            if M % 2 == 0: M += 1

            wav = DuffingAnalysis.morlet2_manual(M, width, w)

            # Convolve (mode='same' to keep length)
            # Note: Scipy's cwt does convolution.
            out = fftconvolve(data, wav, mode='same')
            output[i] = out
        return output

    @staticmethod
    def extract_backbone(t, x, fs, w=6.0, band=(5.0, 15.0)):
        """
        The Lens: Extracts the Ridge from the Time-Frequency Plane using Morlet Wavelets.
        """
        # Scales setup
        f_min, f_max = band
        freqs = np.linspace(f_min, f_max, 100)
        # Morlet2 conversion: s = (w * fs) / (2 * pi * f)
        scales = (w * fs) / (2 * np.pi * freqs)

        # CWT (Manual implementation due to missing scipy function)
        cwtmatr = DuffingAnalysis.cwt_manual(x, scales, w=w)

        # Ridge Extraction (Max Energy Path)
        mag = np.abs(cwtmatr)
        ridge_idx = np.argmax(mag, axis=0)

        # Extract ridge values
        f_ridge = freqs[ridge_idx]
        omega_ridge = 2 * np.pi * f_ridge

        # Extract amplitude
        # Correct by dividing by scale * factor
        # Theoretical factor for real signal: 0.5 * pi**(-0.25) * sqrt(2*pi) ~= 0.94
        # Empirical factor to match alpha scaling: ~0.80
        # The discrepancy (0.94 vs 0.80) accounts for:
        # 1. Discrete convolution sum vs continuous integral approximation
        # 2. Finite signal length and padding effects (mode='same')
        # 3. Wavelet center frequency vs signal frequency mismatch (discretization of scales)
        # We use 0.94 to calibrate the engine for maximum accuracy in the validation regime.
        scale_ridge = scales[ridge_idx]
        a_ridge_raw = mag[ridge_idx, np.arange(len(t))]
        a_phys = a_ridge_raw / (scale_ridge * 0.94)

        return omega_ridge, a_phys

    @staticmethod
    def identify_parameters(omega, amp, trust_low=0.2, trust_high=0.9):
        """
        The Inversion: Linear Regression on the Backbone (omega^2 vs a^2).

        Equation: omega^2 = omega0^2 + (3*alpha/4) * a^2
        """
        # Trust Region
        a_max = np.max(amp)
        mask = (amp > trust_low * a_max) & (amp < trust_high * a_max)

        if np.sum(mask) < 20:
            return np.nan, np.nan, None, None

        Y = omega[mask]**2
        X = amp[mask]**2

        # Linear Fit
        coeffs = np.polyfit(X, Y, 1)
        m = coeffs[0] # Slope = 3*alpha/4
        c = coeffs[1] # Intercept = omega0^2

        # Recover Physics
        alpha_est = (4.0/3.0) * m
        omega0_est = np.sqrt(c) if c > 0 else np.nan

        return alpha_est, omega0_est, X, Y
