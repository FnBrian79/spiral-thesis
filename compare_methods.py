import numpy as np
from scipy.integrate import odeint
from scipy.signal import hilbert, savgol_filter
from duffing_analysis import DuffingAnalysis

# --- CONFIGURATION ---
ALPHA_TRUE = 50.0
GAMMA = 0.2
OMEGA_0_HZ = 10.0
OMEGA_0 = 2 * np.pi * OMEGA_0_HZ
FS = 1000.0
DURATION = 4.0  # Increased duration to see more decay
X0 = [1.0, 0.0]

# --- GENERATION ---
# User's GAMMA is coefficient c/m.
# DuffingAnalysis expects zeta.
# 2*gamma_user = 2*zeta*omega0
# gamma_user = zeta*omega0 => zeta = gamma_user / omega0
zeta_true = GAMMA / OMEGA_0

t = np.arange(0, DURATION, 1/FS)
sol = odeint(DuffingAnalysis.duffing_oscillator, X0, t, args=(zeta_true, OMEGA_0, ALPHA_TRUE))
x_signal = sol[:, 0]

print(f"Simulation Params: Alpha={ALPHA_TRUE}, Gamma={GAMMA} (Zeta={zeta_true:.4f}), F0={OMEGA_0_HZ}Hz")

# --- METHOD 1: USER HILBERT ---
analytic_signal = hilbert(x_signal)
amp_env = np.abs(analytic_signal)
inst_phase = np.unwrap(np.angle(analytic_signal))
inst_freq = np.diff(inst_phase) / (1/FS)

# Align
amp_hilb = amp_env[:-1]
freq_hilb = inst_freq

# Smooth
freq_hilb_smooth = savgol_filter(freq_hilb, window_length=51, polyorder=3)

# Trim
mask_hilb = (t[:-1] > 0.5) & (t[:-1] < DURATION - 0.5)
a2_hilb = amp_hilb[mask_hilb]**2
w2_hilb = freq_hilb_smooth[mask_hilb]**2

# Fit
if len(a2_hilb) > 10:
    slope_h, intercept_h = np.polyfit(a2_hilb, w2_hilb, 1)
    alpha_hilb = slope_h * (4/3)
    print(f"HILBERT Alpha: {alpha_hilb:.2f} (Error: {abs(alpha_hilb - ALPHA_TRUE)/ALPHA_TRUE*100:.2f}%)")
else:
    print("HILBERT: Not enough points")

# --- METHOD 2: CWT ENGINE ---
omega_cwt, amp_cwt = DuffingAnalysis.extract_backbone(t, x_signal, FS, w=10.0, band=(5.0, 15.0))
alpha_cwt, w0_cwt, _, _ = DuffingAnalysis.identify_parameters(omega_cwt, amp_cwt, trust_low=0.2, trust_high=0.9)

print(f"CWT Alpha:     {alpha_cwt:.2f} (Error: {abs(alpha_cwt - ALPHA_TRUE)/ALPHA_TRUE*100:.2f}%)")

# --- EXPERIMENT: HIGHER ALPHA ---
ALPHA_HIGH = 1000.0
sol_high = odeint(DuffingAnalysis.duffing_oscillator, X0, t, args=(zeta_true, OMEGA_0, ALPHA_HIGH))
x_high = sol_high[:, 0]

print(f"\n--- Test with Alpha={ALPHA_HIGH} ---")

# Hilbert
analytic_high = hilbert(x_high)
amp_env_high = np.abs(analytic_high)
inst_phase_high = np.unwrap(np.angle(analytic_high))
inst_freq_high = np.diff(inst_phase_high) / (1/FS)
freq_hilb_high = savgol_filter(inst_freq_high, window_length=51, polyorder=3)
a2_high = amp_env_high[:-1][mask_hilb]**2
w2_high = freq_hilb_high[mask_hilb]**2
slope_h_high, _ = np.polyfit(a2_high, w2_high, 1)
alpha_hilb_high = slope_h_high * (4/3)
print(f"HILBERT Alpha: {alpha_hilb_high:.2f} (Error: {abs(alpha_hilb_high - ALPHA_HIGH)/ALPHA_HIGH*100:.2f}%)")

# CWT
omega_cwt_high, amp_cwt_high = DuffingAnalysis.extract_backbone(t, x_high, FS, w=10.0, band=(5.0, 15.0))
alpha_cwt_high, _, _, _ = DuffingAnalysis.identify_parameters(omega_cwt_high, amp_cwt_high)
print(f"CWT Alpha:     {alpha_cwt_high:.2f} (Error: {abs(alpha_cwt_high - ALPHA_HIGH)/ALPHA_HIGH*100:.2f}%)")
