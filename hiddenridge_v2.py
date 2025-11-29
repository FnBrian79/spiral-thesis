#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hidden‑Ridge v2 – Blind Duffing α recovery with configurable CLI/YAML,
Monte‑Carlo robustness reporting, and backbone export.

Features
--------
* Duffing free‑decay simulation (hardening/softening α)
* Brownian (1/f²) noise injection at arbitrary SNR
* Continuous‑Wavelet Transform ridge extraction (Morlet2)
* Optional synchrosqueezed transform (requires ssqueezepy)
* Linear inversion ω² = ω₀² + (3α/4)·a²
* CLI arguments + YAML config file
* Monte‑Carlo sweep (default 100 seeds)
* CSV/NPZ export of (a, ω) backbone points
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.integrate import odeint
from scipy.signal import fftconvolve
try:
    from scipy.signal import cwt, morlet2
except ImportError:
    # Will fallback to manual implementation
    pass

# ----------------------------------------------------------------------
# Optional synchrosqueezed import – keep it lazy so the script works
# without the extra dependency.
# --------------------------------------------------------------------
try:
    from ssqueezepy import ssq_cwt
    HAVE_SSQUEEZE = True
except ImportError:
    HAVE_SSQUEEZE = False


# ----------------------------------------------------------------------
# Helper dataclasses
# ----------------------------------------------------------------------
@dataclass
class DuffingParams:
    alpha: float          # nonlinearity (hardening >0, softening <0)
    gamma: float          # damping coefficient (2γ appears in ODE)
    omega0: float         # linear natural frequency (rad/s)


@dataclass
class SimConfig:
    duration: float = 4.0          # seconds
    fs: int = 1000                 # sampling frequency (Hz)
    x0: Tuple[float, float] = (1.5, 0.0)  # initial displacement, velocity
    snr_db: float = 3.0            # target SNR against Brownian noise
    seed: int = 42                 # RNG seed for reproducibility


# ----------------------------------------------------------------------
# 1️⃣ Duffing ODE solver
# ----------------------------------------------------------------------
def duffing_ode(state, t, params: DuffingParams):
    """x'' + 2γ x' + ω₀² x + α x³ = 0"""
    x, v = state
    dxdt = v
    dvdt = -2 * params.gamma * v - (params.omega0 ** 2) * x - params.alpha * (x ** 3)
    return [dxdt, dvdt]


def simulate_duffing(params: DuffingParams, cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return time vector and noisy displacement."""
    np.random.seed(cfg.seed)
    t = np.arange(0.0, cfg.duration, 1.0 / cfg.fs)

    # Clean solution
    sol = odeint(duffing_ode, cfg.x0, t, args=(params,))
    x_clean = sol[:, 0]

    # Brownian noise = integrated white noise
    white = np.random.normal(0.0, 1.0, size=t.size)
    brown = np.cumsum(white)

    # Scale noise to desired SNR (linear)
    sig_pow = np.mean(x_clean ** 2)
    noise_pow = np.mean(brown ** 2)

    if noise_pow > 0:
        target_snr_lin = 10 ** (cfg.snr_db / 10.0)
        scale = np.sqrt(sig_pow / (target_snr_lin * noise_pow))
        x_noisy = x_clean + scale * brown
    else:
        x_noisy = x_clean

    return t, x_noisy


# ----------------------------------------------------------------------
# 2️⃣ Ridge extraction (CWT or synchrosqueezed)
# ----------------------------------------------------------------------
def morlet2_manual(M, s, w=5.0):
    """
    Complex Morlet wavelet, centered at zero.
    Matches scipy.signal.morlet2 definition.
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    output = np.pi**(-0.25) * np.exp(1j * w * x) * np.exp(-0.5 * x**2)
    return output

def cwt_manual(data, widths, w=6.0):
    """
    Continuous wavelet transform using fftconvolve.
    Fallback for when scipy.signal.cwt is missing/broken.
    """
    output = np.zeros((len(widths), len(data)), dtype=np.complex128)
    for i, width in enumerate(widths):
        M = int(min(10 * width, len(data)))
        if M < 1: M = 1
        if M % 2 == 0: M += 1

        try:
            # Try using scipy's morlet2 if available
            wav = morlet2(M, width, w)
        except (NameError, ImportError):
            # Fallback to manual
            wav = morlet2_manual(M, width, w)

        out = fftconvolve(data, wav, mode='same')
        output[i] = out
    return output

def morlet_cwt(x, fs, fmin=5.0, fmax=15.0, n_scales=128, w=6.0):
    """Standard Morlet2 CWT with explicit frequency↔scale mapping."""
    fc = w / (2.0 * np.pi)                # Morlet centre frequency (norm. units)
    freqs = np.linspace(fmin, fmax, n_scales)
    scales = (fs * fc) / freqs

    try:
        wavelet = lambda M, s: morlet2(M, s, w=w)
        cwt_mat = cwt(x, wavelet, scales)
    except (NameError, ImportError):
        # Fallback to manual implementation
        cwt_mat = cwt_manual(x, scales, w=w)

    return cwt_mat, freqs, scales


def synchro_ridge(x, fs, fmin=5.0, fmax=15.0, w=6.0):
    """Synchrosqueezed CWT ridge (requires ssqueezepy)."""
    if not HAVE_SSQUEEZE:
        raise RuntimeError("ssqueezepy not installed – cannot use synchrosqueezed mode.")
    # ssq_cwt returns (Tx, Wx, scales, freqs)
    Tx, Wx, scales, freqs = ssq_cwt(
        x,
        wavelet="morlet",
        fs=fs,
        nv=32,                     # number of voices (default)
        fmin=fmin,
        fmax=fmax,
        wavelet_kwargs={"w": w},
    )
    # Energy ridge: max |Tx| per column
    ridge_idx = np.argmax(np.abs(Tx), axis=0)
    amp_ridge = np.abs(Tx[ridge_idx, np.arange(Tx.shape[1])])
    return ridge_idx, amp_ridge, freqs, scales


def extract_backbone(t, x, fs, cfg):
    """Unified interface – returns ω(t), a(t) and the raw ridge indices."""
    if cfg.use_synchro:
        ridge_idx, amp_ridge, freqs, scales = synchro_ridge(
            x, fs, fmin=cfg.fmin, fmax=cfg.fmax, w=cfg.w
        )
    else:
        cwt_mat, freqs, scales = morlet_cwt(
            x, fs, fmin=cfg.fmin, fmax=cfg.fmax, n_scales=cfg.n_scales, w=cfg.w
        )
        mag = np.abs(cwt_mat)
        ridge_idx = np.argmax(mag, axis=0)
        amp_ridge = mag[ridge_idx, np.arange(cwt_mat.shape[1])]

    # Instantaneous frequency & amplitude
    inst_freq = freqs[ridge_idx]               # Hz
    omega = 2.0 * np.pi * inst_freq            # rad/s

    # [COMPENSATION]: Amplitude Normalization
    # The CWT amplitude from discrete convolution scales with 's'.
    # We divide by 's' to normalize.
    # We further divide by the theoretical factor for real-valued signals:
    # Factor = 0.5 (real-to-analytic) * pi**(-0.25) (peak) * sqrt(2*pi) (integral)
    #        ~= 0.94
    # This "illuminates" the true physical amplitude, compensating for the
    # mathematical properties of the Morlet wavelet convolution.
    CALIBRATION_FACTOR = 0.94
    amp = amp_ridge / (scales[ridge_idx] * CALIBRATION_FACTOR)

    return omega, amp, ridge_idx, freqs, scales, mag


# ----------------------------------------------------------------------
# 3️⃣ Linear inversion on the backbone
# ----------------------------------------------------------------------
def invert_alpha(omega, amp, trust_low=0.2, trust_high=0.9):
    """Fit ω² = ω₀² + (3α/4)·a² on a trusted amplitude window."""
    Y = omega ** 2
    X = amp ** 2

    a_max = np.max(amp)
    mask = (amp > a_max * trust_low) & (amp < a_max * trust_high)

    X_fit = X[mask]
    Y_fit = Y[mask]

    if len(X_fit) < 10:
        return np.nan, np.nan, 0, 0, mask

    # Simple least‑squares linear fit
    coeffs = np.polyfit(X_fit, Y_fit, 1)   # coeffs[0]=slope, coeffs[1]=intercept
    slope, intercept = coeffs

    omega0_est = np.sqrt(intercept) if intercept > 0 else np.nan
    alpha_est = (4.0 / 3.0) * slope

    return alpha_est, omega0_est, slope, intercept, mask


# ----------------------------------------------------------------------
# 4️⃣ Monte‑Carlo sweep helper
# ----------------------------------------------------------------------
def monte_carlo_sweep(
    duff_params: DuffingParams,
    sim_cfg: SimConfig,
    ridge_cfg,
    n_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run `n_trials` seeds, return arrays of α errors (%) and ω₀ errors (%)."""
    alpha_err = np.empty(n_trials)
    omega0_err = np.empty(n_trials)

    for i in range(n_trials):
        sim_cfg.seed = i  # deterministic progression
        t, x = simulate_duffing(duff_params, sim_cfg)
        omega, amp, ridge_idx, freqs, scales, mag = extract_backbone(t, x, sim_cfg.fs, ridge_cfg)

        alpha_hat, omega0_hat, *_ = invert_alpha(omega, amp,
                                                trust_low=ridge_cfg.trust_low,
                                                trust_high=ridge_cfg.trust_high)

        if np.isnan(alpha_hat):
            alpha_err[i] = np.nan
            omega0_err[i] = np.nan
        else:
            alpha_err[i] = (
                np.abs(alpha_hat - duff_params.alpha) / np.abs(duff_params.alpha) * 100.0
            )
            omega0_err[i] = (
                np.abs(omega0_hat - duff_params.omega0) / duff_params.omega0 * 100.0
            )

    return alpha_err, omega0_err


# ----------------------------------------------------------------------
# 5️⃣ Configuration parsing (YAML + CLI)
# ----------------------------------------------------------------------
@dataclass
class RidgeConfig:
    fmin: float = 5.0
    fmax: float = 15.0
    w: float = 6.0
    n_scales: int = 128
    use_synchro: bool = False
    trust_low: float = 0.2
    trust_high: float = 0.9


def load_yaml(path: pathlib.Path) -> dict:
    with open(path, "rt", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config_from_namespace(ns) -> Tuple[DuffingParams, SimConfig, RidgeConfig]:
    duff = DuffingParams(alpha=ns.alpha, gamma=ns.gamma, omega0=2.0 * np.pi * ns.f0_hz)
    sim = SimConfig(
        duration=ns.duration,
        fs=ns.fs,
        x0=(ns.x0_disp, ns.x0_vel),
        snr_db=ns.snr_db,
        seed=ns.seed,
    )
    ridge = RidgeConfig(
        fmin=ns.fmin,
        fmax=ns.fmax,
        w=ns.w,
        n_scales=ns.n_scales,
        use_synchro=ns.use_synchro,
        trust_low=ns.trust_low,
        trust_high=ns.trust_high,
    )
    return duff, sim, ridge


def cli_parser():
    parser = argparse.ArgumentParser(
        description="Hidden‑Ridge v2 – blind Duffing α recovery"
    )
    # ----- YAML ---------------------------------------------------------
    parser.add_argument("-c", "--config", type=pathlib.Path,
                        help="Path to YAML config file (overrides CLI defaults)")
    # ----- Duffing parameters -------------------------------------------
    parser.add_argument("--alpha", type=float, default=500.0,
                        help="True nonlinearity α (hardening >0, softening <0)")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Damping coefficient γ")
    parser.add_argument("--f0-hz", type=float, default=10.0,
                        help="Linear natural frequency (Hz)")
    # ----- Simulation ----------------------------------------------------
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=1000)
    parser.add_argument("--x0-disp", type=float, default=1.5,
                        help="Initial displacement")
    parser.add_argument("--x0-vel", type=float, default=0.0,
                        help="Initial velocity")
    parser.add_argument("--snr-db", type=float, default=10.0,
                        help="Target SNR (dB) against Brownian noise")
    parser.add_argument("--seed", type=int, default=42)
    # ----- Ridge extraction ---------------------------------------------
    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=15.0)
    parser.add_argument("--w", type=float, default=10.0,
                        help="Morlet wavelet parameter")
    parser.add_argument("--n-scales", type=int, default=128)
    parser.add_argument("--use-synchro", action="store_true",
                        help="Enable synchrosqueezed CWT (requires ssqueezepy)")
    parser.add_argument("--trust-low", type=float, default=0.2)
    parser.add_argument("--trust-high", type=float, default=0.9)
    # ----- Monte‑Carlo --------------------------------------------------
    parser.add_argument("--mc-trials", type=int, default=0,
                        help="Number of Monte‑Carlo seeds (>0 runs robustness sweep)")
    parser.add_argument("--export-backbone", type=pathlib.Path,
                        help="CSV file to dump (a, ω) backbone points")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display diagnostic plots")
    return parser


# ----------------------------------------------------------------------
# 6️⃣ Main driver
# ----------------------------------------------------------------------
def main():
    args = cli_parser().parse_args()

    # Load YAML if supplied and overlay onto args namespace
    if args.config:
        yaml_cfg = load_yaml(args.config)
        for key, val in yaml_cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)

    duff_params, sim_cfg, ridge_cfg = build_config_from_namespace(args)

    # ------------------------------------------------------------------
    # Single run (or MC loop)
    # ------------------------------------------------------------------
    if args.mc_trials > 0:
        alpha_err, omega0_err = monte_carlo_sweep(
            duff_params, sim_cfg, ridge_cfg, n_trials=args.mc_trials
        )
        # Filter NaNs
        alpha_err = alpha_err[~np.isnan(alpha_err)]
        omega0_err = omega0_err[~np.isnan(omega0_err)]

        median_alpha = np.median(alpha_err)
        mad_alpha = np.median(np.abs(alpha_err - median_alpha))
        median_omega = np.median(omega0_err)
        mad_omega = np.median(np.abs(omega0_err - median_omega))

        print("\n=== Monte‑Carlo Robustness Report ===")
        print(f"Trials          : {args.mc_trials}")
        print(f"α error  (median±MAD)  : {median_alpha:.2f}% ± {mad_alpha:.2f}%")
        print(f"ω₀ error (median±MAD) : {median_omega:.2f}% ± {mad_omega:.2f}%")
        if args.show_plots:
            plt.boxplot([alpha_err, omega0_err],
                        labels=["α error (%)", "ω₀ error (%)"])
            plt.title("Monte‑Carlo Error Distribution")
            plt.grid(True, axis="y")
            plt.show()
        return

    # ------------------------------------------------------------------
    # Single deterministic run
    # ------------------------------------------------------------------
    t, x_noisy = simulate_duffing(duff_params, sim_cfg)
    omega, amp, ridge_idx, freqs, scales, mag = extract_backbone(t, x_noisy,
                                                          sim_cfg.fs,
                                                          ridge_cfg)

    alpha_hat, omega0_hat, slope, intercept, mask = invert_alpha(
        omega, amp, trust_low=ridge_cfg.trust_low, trust_high=ridge_cfg.trust_high
    )

    # ----- Reporting ---------------------------------------------------
    err_alpha = (
        np.abs(alpha_hat - duff_params.alpha) / np.abs(duff_params.alpha) * 100.0
    )
    err_omega0 = (
        np.abs(omega0_hat - duff_params.omega0) / duff_params.omega0 * 100.0
    )

    print("\n" + "="*60)
    print("HIDDEN RIDGE: ILLUMINATION REPORT")
    print("="*60)
    print(f"[*] Truth (Input):  α = {duff_params.alpha:.4f}")
    print(f"[*] Vision (Ridge): α = {alpha_hat:.4f}")
    print(f"[*] Error:          {err_alpha:.2f}%")
    print("-" * 60)
    print(f"[*] Truth (Input):  f₀ = {duff_params.omega0/(2*np.pi):.4f} Hz")
    print(f"[*] Vision (Ridge): f₀ = {omega0_hat/(2*np.pi):.4f} Hz")
    print(f"[*] Error:          {err_omega0:.2f}%")
    print("="*60)

    if err_alpha < 5.0:
        print(">> SUCCESS: The ridge has spoken the truth.")
    else:
        print(">> NOTE: The noise is loud, but the backbone remains.")

    # ----- Export backbone if requested --------------------------------
    if args.export_backbone:
        export_path = args.export_backbone
        np.savetxt(
            export_path,
            np.column_stack((amp[mask] ** 2, omega[mask] ** 2)),
            delimiter=",",
            header="a_squared,omega_squared",
            comments="",
        )
        print(f"Backbone points saved to: {export_path}")

    # ----- Plots -------------------------------------------------------
    if args.show_plots:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # (a) Noisy time series + envelope
        axs[0, 0].plot(t, x_noisy, 'k', alpha=0.3, label='Noisy Signal')
        axs[0, 0].set_title(f"Signal (SNR {sim_cfg.snr_db}dB)")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].legend()

        # (b) Wavelet Ridge
        axs[0, 1].imshow(mag, aspect='auto', extent=[t[0], t[-1], freqs[0], freqs[-1]], origin='lower')
        axs[0, 1].plot(t, freqs[ridge_idx], 'r', linewidth=1, label='Ridge')
        axs[0, 1].set_title("Time-Frequency Ridge")
        axs[0, 1].set_ylabel("Frequency (Hz)")

        # (c) Backbone Curve (The Money Plot)
        axs[1, 0].scatter(amp[mask]**2, omega[mask]**2, s=1, alpha=0.5, c='b')
        # Fit line
        if len(amp[mask]) > 0:
            x_fit = np.linspace(min(amp[mask]**2), max(amp[mask]**2), 100)
            y_fit = slope * x_fit + intercept
            axs[1, 0].plot(x_fit, y_fit, 'r--', label=f'Fit (Err: {err_alpha:.1f}%)')

        axs[1, 0].set_title(r"Backbone: $\omega^2$ vs $a^2$")
        axs[1, 0].set_xlabel(r"$a^2$")
        axs[1, 0].set_ylabel(r"$\omega^2$")
        axs[1, 0].legend()

        # (d) Instantaneous Frequency
        axs[1, 1].plot(t[mask], omega[mask]/(2*np.pi), 'g', label='Inst. Freq')
        axs[1, 1].set_title("Frequency Decay")
        axs[1, 1].set_ylabel("Frequency (Hz)")
        axs[1, 1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
