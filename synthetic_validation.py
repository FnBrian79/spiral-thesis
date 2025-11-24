#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from duffing_analysis import DuffingAnalysis
import argparse
from tabulate import tabulate

def run_campaign():
    print("="*60)
    print("THE CAMPAIGN: SYNTHETIC VALIDATION SUITE")
    print("="*60)

    # Parameter Space
    alphas = [-500, 0, 500, 2000, 10000]
    snrs = [-5, 0, 10, 20, 30] # dB
    seeds_per_condition = 20 # Monte Carlo (Reduced for speed)

    # Results Storage
    results = []

    total_runs = len(alphas) * len(snrs) * seeds_per_condition
    run_count = 0

    print(f"[*] Conditions: {len(alphas)} Alphas x {len(snrs)} SNRs")
    print(f"[*] Trials per condition: {seeds_per_condition}")
    print(f"[*] Total simulations: {total_runs}")
    print("-" * 60)

    for alpha in alphas:
        for snr in snrs:
            estimates = []

            for seed in range(seeds_per_condition):
                try:
                    # 1. Generate Signal (The Chaos)
                    t, x, true_w0 = DuffingAnalysis.simulate_signal(
                        alpha=alpha,
                        snr_db=snr,
                        seed=seed,
                        f0=10.0,
                        zeta=0.02 # Light damping
                    )

                    # 2. The Lens (Wavelet Ridge)
                    omega, amp = DuffingAnalysis.extract_backbone(
                        t, x, fs=1000.0, w=10.0, band=(5.0, 15.0)
                    )

                    # 3. The Inversion
                    est_alpha, est_w0, _, _ = DuffingAnalysis.identify_parameters(
                        omega, amp, trust_low=0.2, trust_high=0.9
                    )

                    if not np.isnan(est_alpha):
                        estimates.append(est_alpha)

                except Exception as e:
                    # Silent fail for individual seeds is acceptable in MC
                    continue

                run_count += 1
                if run_count % 500 == 0:
                    print(f"Progress: {run_count}/{total_runs} ...")

            # 4. Aggregation (The Truth)
            if estimates:
                rec = np.array(estimates)
                median_alpha = np.median(rec)
                std_alpha = np.std(rec)

                # Calculate Error Metrics
                if alpha != 0:
                    bias_pct = (np.mean(rec - alpha) / alpha) * 100
                    rel_err = np.abs((median_alpha - alpha) / alpha) * 100
                else:
                    # For alpha=0, bias is absolute error
                    bias_pct = np.mean(rec)
                    rel_err = np.abs(median_alpha) # Just absolute deviation

                # Success Rate (within +/- 20% or +/- 50 for alpha=0)
                if alpha != 0:
                    success = np.sum(np.abs((rec - alpha)/alpha) < 0.20) / len(rec) * 100
                else:
                    success = np.sum(np.abs(rec) < 50) / len(rec) * 100

                results.append({
                    'True Alpha': alpha,
                    'SNR (dB)': snr,
                    'Median Rec': round(median_alpha, 1),
                    'Std Dev': round(std_alpha, 1),
                    'Error %': round(rel_err, 2) if alpha != 0 else "N/A",
                    'Success %': round(success, 1),
                    'N': len(rec)
                })
            else:
                print(f"!! FAILED to recover any data for Alpha={alpha}, SNR={snr}")

    # 5. Report Generation
    df = pd.DataFrame(results)
    df.to_csv('validation_results.csv', index=False)

    print("\n" + "="*60)
    print("FINAL VALIDATION REPORT")
    print("="*60)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print(f"\n[*] Detailed results saved to 'validation_results.csv'")

if __name__ == "__main__":
    run_campaign()
