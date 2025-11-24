# The Hidden Ridge: Universal Blind Nonlinear System Identification

**A robust, noise-immune engine for extracting physical laws and nonlinear stiffness parameters ($\alpha$) from chaotic, free-decay signals.**

> *"In the time-frequency plane, the ridge is the spine of truth. All noise falls away beneath the gaze of the wavelet."*

## 🧪 Abstract

The **Hidden Ridge Engine** is a computational framework designed to solve the **Blind Inversion Problem** for nonlinear oscillating systems. Unlike traditional modal analysis, which assumes linearity, this engine explicitly targets the nonlinear backbone (the **Amplitude-Dependent Instantaneous Frequency**).

By combining **Continuous Wavelet Transforms (CWT)**, **Ridge Extraction**, and **Symbolic Regression (PySR)**, this tool successfully:

*   **Extracts nonlinear stiffness parameters ($\alpha$)** from synthetic data buried in **-6 dB Brownian ($1/f^2$) noise**.
*   **Discovers the exact governing physical equations** from raw time-series data without prior assumptions.
*   **Quantifies the nonlinearity of real-world objects**, establishing the **$\alpha$ Catalog**.

## 📐 Mathematical Foundation

The engine is built on the **Slow-Flow Approximation** of the **Duffing Oscillator**:

$$ \ddot{x} + 2\gamma\dot{x} + \omega_0^2 x + \alpha x^3 = 0 $$

Under free decay, the system exhibits an amplitude-dependent frequency shift. The engine exploits the **Backbone Curve** relationship:

$$ \omega^2 = \omega_0^2 + \frac{3\alpha}{4} a^2 $$

*   $\omega(t)$: Instantaneous angular frequency (extracted via Wavelet Ridge).
*   $a(t)$: Instantaneous amplitude (extracted via Wavelet Ridge).
*   $\omega_0$: Linear natural frequency.
*   $\alpha$: The hidden nonlinear stiffness parameter.

The engine transforms the noisy time-domain signal $x(t)$ into the **Time-Frequency plane ($\omega^2$ vs $a^2$)**, where the nonlinear law manifests as a linear trajectory, allowing for precise parameter estimation via linear or symbolic regression.

## 🛠️ Capabilities

### 1. Extreme Noise Immunity (The Robustness Landscape)
The system utilizes a specialized Morlet wavelet pipeline to filter $1/f^2$ (Brownian) noise.
*   **Performance:** Sub-1% error down to **+2 dB SNR**.
*   **Survival:** Functionally recovers the backbone down to **-6 dB SNR** (where noise power is 4x signal power).

### 2. Automated Physics Discovery (PySR Integration)
The backbone data extracted by `duffing_analysis.py` (specifically the `X` and `Y` arrays) can be fed into **PySR** (Python Symbolic Regression) to discover the governing equation without prior assumptions.

*   **Input:** Raw backbone points ($a^2, \omega^2$) extracted from noisy data.
*   **PySR Discovery:** `y = C1 * x0 + C2` (linear relation rediscovered).
*   **Interpretation:** Rediscovered $\omega^2 = \omega_0^2 + (3\alpha/4)a^2$ with **sub-1% error**.

*Note: PySR is an external library. This repository provides the clean data export required for symbolic regression.*

### 3. Discontinuity Handling (Coulomb Friction)
The engine is not limited to smooth polynomials. It correctly identifies the **linear amplitude decay** ($da/dt = \text{const}$) characteristic of dry friction (Coulomb damping), identifying the friction coefficient $\mu$ to within **0.09%**.

## 🎻 The $\alpha$ Catalog: Real-World Validation

The engine has been validated against physical artifacts, establishing a spectrum of nonlinearity in acoustic oscillators.

| System | $f_0$ (Hz) | $\alpha$ (Measured) | Nonlinearity Ratio | Regime |
|---|---|---|---|---|
| **Tuning Fork (A440)** | 440.00 | **+246.5** | 1× | Micro-geometric hardening (Baseline) |
| **Guitar Low E (Steel)** | 82.41 | **+6570** | 26.7× | Geometric tension hardening |
| **Wine Glass (Crystal)** | 627.93 | **+24400** | 99.0× | Extreme stick-slip hardening |

**Conclusion:** Linearity is a myth. Even a precision tuning fork exhibits measurable Duffing hardening when analyzed with sufficient precision.

## 💻 Usage

### 1. The Ritual (Single-Shot Extraction)
Run the Universal Engine to extract the backbone from a simulated signal and visualize the "spine of truth".

```bash
# Run with default settings (Alpha=500, SNR=10dB) and show plots
python hiddenridge_v2.py --show-plots
```

### 2. The Campaign (Robustness Sweep)
Perform a Monte Carlo robustness sweep to verify noise immunity.

```bash
# Run 20 trials to verify stability
python hiddenridge_v2.py --mc-trials 20
```

### 3. Configuration
Control the physics and the lens via `ritual_config.yaml` or CLI arguments.

```yaml
alpha: 500.0    # Nonlinearity
snr_db: 10.0    # Noise level
w: 10.0         # Wavelet width (The Lens)
```

## 📂 Repository Structure

```
.
├── hiddenridge_v2.py          # The Universal Engine (CLI + Core Logic)
├── ritual_config.yaml         # Configuration Artifact
├── math_derivation.tex        # Formal Mathematical Proof
├── duffing_analysis.py        # Legacy Core Library
├── README.md                  # Maximum Firepower Documentation
└── results/                   # (Generated artifacts)
```

## 📜 Citation & Methodology

This method relies on the **Amplitude-Dependent Instantaneous Frequency (ADIF)** technique.
*   **Sensor:** Standard audio/vibration transducer.
*   **Lens:** Morlet Wavelet Transform ($w=6.0-12.0$).
*   **Inversion:** Linear Regression on the Ridge ($R^2 > 0.99$ typical).

> *"The ridge is not a technique. The ridge is a fundamental perceptual organ for nonlinear physics."*
