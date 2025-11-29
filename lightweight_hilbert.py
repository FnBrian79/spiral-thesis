import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import hilbert, savgol_filter

# --- 1. CONFIGURATION (The "Knobs") ---
# We set a known nonlinearity (alpha) to see if we can recover it.
ALPHA_TRUE = 50.0     # The nonlinearity strength (Hardening)
GAMMA = 0.2           # Damping (Light damping is best)
OMEGA_0 = 2 * np.pi * 10  # Linear natural frequency (10 Hz)
FS = 1000             # Sampling frequency
DURATION = 4.0        # Seconds to simulate

# --- 2. SIMULATION (Generating the "Reality") ---
def duffing_deriv(state, t):
    x, v = state
    # The equation: x'' + 2*gamma*x' + w0^2*x + alpha*x^3 = 0
    dxdt = v
    dvdt = -2 * GAMMA * v - (OMEGA_0**2) * x - ALPHA_TRUE * x**3
    return [dxdt, dvdt]

t = np.arange(0, DURATION, 1/FS)
x0 = [1.0, 0.0] # Initial pluck: Amplitude 1.0, Velocity 0
solution = odeint(duffing_deriv, x0, t)
x_signal = solution[:, 0]

# --- 3. EXTRACTION (The "Backbone" Logic) ---
# using Hilbert Transform (standard method for single-mode decay)
analytic_signal = hilbert(x_signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_freq = np.diff(instantaneous_phase) / (1/FS)

# Align arrays (diff reduces length by 1)
amp_fit = amplitude_envelope[:-1]
freq_fit = instantaneous_freq

# --- 4. CLEANUP & FITTING ---
# Smooth the frequency derivative noise
freq_fit = savgol_filter(freq_fit, window_length=51, polyorder=3)

# Trim edges (Hilbert transform has "edge effects" at start/end)
# Increased start trim to 0.5s to avoid SavGol/Hilbert startup transients
mask = (t[:-1] > 0.5) & (t[:-1] < DURATION - 0.5)
a2 = amp_fit[mask]**2
w2 = freq_fit[mask]**2

# Linear Regression: w^2 = w0^2 + (3*alpha/4) * a^2
slope, intercept = np.polyfit(a2, w2, 1)
alpha_recovered = slope * (4/3)

# --- 5. VISUALIZATION ---
# plt.figure(figsize=(10, 6))

# # Plot 1: The raw decay
# plt.subplot(2, 1, 1)
# plt.title(f"Simulated Decay (True Alpha = {ALPHA_TRUE})")
# plt.plot(t, x_signal, 'k', alpha=0.5, label='Raw Signal')
# plt.plot(t, amplitude_envelope, 'r--', label='Envelope')
# plt.grid(True)
# plt.legend()

# # Plot 2: The Backbone Curve (The "Money Plot")
# plt.subplot(2, 1, 2)
# plt.scatter(a2, w2, c='blue', s=1, alpha=0.5, label='Extracted Data')

# # Plot the fit line
# fit_line = slope * a2 + intercept
# plt.plot(a2, fit_line, 'r', linewidth=2, label=f'Fit (Recovered Alpha = {alpha_recovered:.2f})')

# plt.xlabel(r'Squared Amplitude ($a^2$)')
# plt.ylabel(r'Squared Frequency ($\omega^2$)')
# plt.title("Backbone Curve Extraction")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

print(f"---------------------------------------")
print(f"INPUT ALPHA:     {ALPHA_TRUE}")
print(f"RECOVERED ALPHA: {alpha_recovered:.2f}")
print(f"ERROR:           {abs(ALPHA_TRUE - alpha_recovered)/ALPHA_TRUE * 100:.2f}%")
print(f"---------------------------------------")
