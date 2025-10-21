"""
Performance comparison between encrypted and plaintext operations for selective homomorphic encryption.
This script compares the performance and accuracy of CKKS-based selective encryption vs plaintext operations.
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
except ImportError:
    print("Warning: Pyfhel not found, using mock implementation for testing...")
    from pyfhel_mock import Pyfhel, PyCtxt, PyPtxt

DATA_DIR = Path("data/covid_ct_cxr")
CONFIG_DIR = Path("config")
MULTIMODAL_PATH = DATA_DIR/"multimodal.csv"
POLICY_PATH = CONFIG_DIR/"selective_he_policy.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_policy(path: Path):
    """Load selective HE policy configuration."""
    with open(path, 'r', encoding='utf-8') as f:
        pol = json.load(f)
    cols = pol["encrypt_columns"]
    weights_map = pol["weights"]
    weights = [weights_map[c] for c in cols]
    bias = pol.get("bias", 0.0)
    return cols, weights, bias, pol


def run_plaintext_operations(df: pd.DataFrame, cols, weights, bias, iterations=10):
    """Run plaintext operations for performance comparison."""
    results = []
    latencies = []
    
    for iteration in range(iterations):
        iter_latencies = []
        for _, row in df.iterrows():
            xvals = [float(row[c]) for c in cols]
            
            # Measure plaintext computation time
            t0 = time.time()
            plain_score = sum(w*x for w, x in zip(weights, xvals)) + bias
            latency = (time.time() - t0) * 1000.0  # Convert to milliseconds
            iter_latencies.append(latency)
            
            if iteration == 0:  # Only store results once
                results.append({
                    "patient_id": row.get("patient_id", "unknown"),
                    "plaintext_score": plain_score,
                    "iteration": iteration
                })
        
        latencies.extend(iter_latencies)
    
    return results, latencies


def run_encrypted_operations(df: pd.DataFrame, cols, weights, bias, iterations=10):
    """Run encrypted operations using CKKS for performance comparison."""
    # Initialize Pyfhel with CKKS
    HE = Pyfhel()
    HE.contextGen(scheme='CKKS', n=2**13, scale=2**30, qi_sizes=[60, 40, 40, 60])
    HE.keyGen()
    HE.relinKeyGen()
    
    results = []
    latencies = []
    
    for iteration in range(iterations):
        iter_latencies = []
        for _, row in df.iterrows():
            xvals = [float(row[c]) for c in cols]
            
            # Measure encrypted computation time
            t0 = time.time()
            
            # Encrypt each value and perform weighted sum
            ct_sum = None
            for w, x in zip(weights, xvals):
                ptxt_x = HE.encodeFrac([x])
                ct_x = HE.encryptPtxt(ptxt_x)
                ptxt_w = HE.encodeFrac([w])
                ct_xw = ct_x * ptxt_w
                ct_sum = ct_xw if ct_sum is None else (ct_sum + ct_xw)
            
            # Add bias
            ptxt_b = HE.encodeFrac([bias])
            ct_sum = ct_sum + ptxt_b
            
            # Decrypt result
            dec = HE.decryptFrac(ct_sum)
            encrypted_score = float(dec[0])
            
            latency = (time.time() - t0) * 1000.0  # Convert to milliseconds
            iter_latencies.append(latency)
            
            if iteration == 0:  # Only store results once
                results.append({
                    "patient_id": row.get("patient_id", "unknown"),
                    "encrypted_score": encrypted_score,
                    "iteration": iteration
                })
        
        latencies.extend(iter_latencies)
    
    return results, latencies


def analyze_performance(plain_latencies, encrypted_latencies, plain_results, encrypted_results):
    """Analyze performance metrics and accuracy."""
    
    # Performance metrics
    plain_mean = np.mean(plain_latencies)
    plain_std = np.std(plain_latencies)
    encrypted_mean = np.mean(encrypted_latencies)
    encrypted_std = np.std(encrypted_latencies)
    
    # Calculate speedup/overhead
    overhead_factor = encrypted_mean / plain_mean
    overhead_percentage = ((encrypted_mean - plain_mean) / plain_mean) * 100
    
    # Accuracy analysis
    plain_scores = [r["plaintext_score"] for r in plain_results]
    encrypted_scores = [r["encrypted_score"] for r in encrypted_results]
    
    # Calculate absolute errors
    abs_errors = [abs(p - e) for p, e in zip(plain_scores, encrypted_scores)]
    mean_abs_error = np.mean(abs_errors)
    max_abs_error = np.max(abs_errors)
    
    # Calculate relative errors
    rel_errors = [abs(p - e) / abs(p) if p != 0 else 0 for p, e in zip(plain_scores, encrypted_scores)]
    mean_rel_error = np.mean(rel_errors) * 100  # Convert to percentage
    
    return {
        "performance": {
            "plaintext_mean_ms": plain_mean,
            "plaintext_std_ms": plain_std,
            "encrypted_mean_ms": encrypted_mean,
            "encrypted_std_ms": encrypted_std,
            "overhead_factor": overhead_factor,
            "overhead_percentage": overhead_percentage
        },
        "accuracy": {
            "mean_absolute_error": mean_abs_error,
            "max_absolute_error": max_abs_error,
            "mean_relative_error_percent": mean_rel_error,
            "sample_count": len(plain_scores)
        }
    }


def create_visualizations(plain_latencies, encrypted_latencies, analysis_results):
    """Create performance comparison visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Latency comparison histogram
    ax1.hist(plain_latencies, bins=30, alpha=0.7, label='Plaintext', color='blue')
    ax1.hist(encrypted_latencies, bins=30, alpha=0.7, label='Encrypted (CKKS)', color='red')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Latency Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    data_for_box = [plain_latencies, encrypted_latencies]
    labels = ['Plaintext', 'Encrypted (CKKS)']
    ax2.boxplot(data_for_box, labels=labels)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics bar chart
    metrics = ['Mean Latency (ms)', 'Std Latency (ms)']
    plain_values = [analysis_results["performance"]["plaintext_mean_ms"], 
                   analysis_results["performance"]["plaintext_std_ms"]]
    encrypted_values = [analysis_results["performance"]["encrypted_mean_ms"], 
                       analysis_results["performance"]["encrypted_std_ms"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, plain_values, width, label='Plaintext', color='blue', alpha=0.7)
    ax3.bar(x + width/2, encrypted_values, width, label='Encrypted (CKKS)', color='red', alpha=0.7)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Overhead analysis
    overhead_data = {
        'Overhead Factor': analysis_results["performance"]["overhead_factor"],
        'Overhead %': analysis_results["performance"]["overhead_percentage"]
    }
    
    ax4.bar(overhead_data.keys(), overhead_data.values(), color=['orange', 'purple'], alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Encryption Overhead Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations for overhead
    for i, (key, value) in enumerate(overhead_data.items()):
        ax4.text(i, value + max(overhead_data.values()) * 0.01, f'{value:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance visualization saved to: {RESULTS_DIR / 'performance_comparison.png'}")


def main():
    """Main function to run performance comparison."""
    
    if not MULTIMODAL_PATH.exists():
        raise SystemExit(f"Multimodal file not found: {MULTIMODAL_PATH}")
    if not POLICY_PATH.exists():
        raise SystemExit(f"Policy file not found: {POLICY_PATH}")
    
    # Load data and configuration
    df = pd.read_csv(MULTIMODAL_PATH)
    cols, weights, bias, pol = load_policy(POLICY_PATH)
    
    print("Starting performance comparison...")
    print(f"Dataset size: {len(df)} samples")
    print(f"Encrypted columns: {cols}")
    
    # Run performance tests
    iterations = 5  # Number of iterations for statistical significance
    
    print("\nRunning plaintext operations...")
    plain_results, plain_latencies = run_plaintext_operations(df, cols, weights, bias, iterations)
    
    print("Running encrypted operations...")
    encrypted_results, encrypted_latencies = run_encrypted_operations(df, cols, weights, bias, iterations)
    
    # Analyze results
    print("\nAnalyzing performance...")
    analysis = analyze_performance(plain_latencies, encrypted_latencies, plain_results, encrypted_results)
    
    # Create detailed results
    detailed_results = {
        "experiment_info": {
            "dataset_size": len(df),
            "iterations": iterations,
            "encrypted_columns": cols,
            "policy": pol
        },
        "analysis": analysis,
        "raw_data": {
            "plaintext_latencies_ms": plain_latencies,
            "encrypted_latencies_ms": encrypted_latencies,
            "plaintext_results": plain_results,
            "encrypted_results": encrypted_results
        }
    }
    
    # Save results
    results_file = RESULTS_DIR / "performance_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Create visualizations
    create_visualizations(plain_latencies, encrypted_latencies, analysis)
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(f"Plaintext mean latency: {analysis['performance']['plaintext_mean_ms']:.4f} ms")
    print(f"Encrypted mean latency: {analysis['performance']['encrypted_mean_ms']:.4f} ms")
    print(f"Overhead factor: {analysis['performance']['overhead_factor']:.2f}x")
    print(f"Overhead percentage: {analysis['performance']['overhead_percentage']:.2f}%")
    print(f"\nAccuracy Analysis:")
    print(f"Mean absolute error: {analysis['accuracy']['mean_absolute_error']:.6f}")
    print(f"Max absolute error: {analysis['accuracy']['max_absolute_error']:.6f}")
    print(f"Mean relative error: {analysis['accuracy']['mean_relative_error_percent']:.4f}%")
    print(f"\nResults saved to: {results_file}")
    print("="*60)


if __name__ == '__main__':
    main()