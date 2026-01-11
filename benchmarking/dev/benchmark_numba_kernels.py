#!/usr/bin/env python
"""Benchmark Phase 2 Numba kernels vs pure NumPy implementation."""

import numpy as np
import time
from crispyx._kernels import (
    irls_iteration_fused_parallel,
    irls_iteration_with_reduction,
    compute_weighted_sums_parallel,
)


def numpy_irls_iteration(
    Y: np.ndarray,
    beta_intercept: np.ndarray,
    beta_pert_coef: np.ndarray,
    offset: np.ndarray,
    alpha: np.ndarray,
    log_min_mu: float,
    min_mu: float,
):
    """Pure NumPy implementation for comparison."""
    # eta = intercept + pert_coef + offset
    eta = beta_intercept[None, :] + beta_pert_coef[None, :] + offset
    np.clip(eta, log_min_mu, 20.0, out=eta)
    
    # mu = exp(eta)
    mu = np.exp(eta)
    np.maximum(mu, min_mu, out=mu)
    
    # W = mu^2 / variance
    var = mu + alpha[None, :] * mu * mu
    W = mu * mu / np.maximum(var, min_mu)
    
    # z = eta + (Y - mu) / mu
    z = eta + (Y - mu) / np.maximum(mu, min_mu)
    
    # Reductions
    W_sum = np.sum(W, axis=0)
    Wz_sum = np.sum(W * z, axis=0)
    
    return eta, mu, W_sum, Wz_sum


def benchmark_kernels(n_cells: int, n_genes: int, n_iters: int = 10):
    """Compare Numba vs NumPy for IRLS iteration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {n_cells} cells × {n_genes} genes")
    print(f"{'='*60}")
    
    rng = np.random.default_rng(42)
    
    # Create test data
    Y = rng.poisson(10, size=(n_cells, n_genes)).astype(np.float64)
    beta_intercept = rng.normal(2.0, 0.5, size=n_genes)
    beta_pert_coef = rng.normal(0.0, 0.5, size=n_genes)
    offset = rng.normal(-0.5, 0.2, size=(n_cells, 1)) * np.ones((1, n_genes))
    alpha = np.abs(rng.normal(0.1, 0.02, size=n_genes))
    log_min_mu = np.log(1e-6)
    min_mu = 1e-6
    
    # Pre-allocate outputs for Numba
    eta_out = np.empty((n_cells, n_genes), dtype=np.float64)
    mu_out = np.empty((n_cells, n_genes), dtype=np.float64)
    W_sum_out = np.empty(n_genes, dtype=np.float64)
    Wz_sum_out = np.empty(n_genes, dtype=np.float64)
    
    # Warmup JIT compilation
    print("Warming up JIT compilation...")
    irls_iteration_with_reduction(
        Y, beta_intercept, beta_pert_coef, offset, alpha,
        log_min_mu, min_mu, eta_out, mu_out, W_sum_out, Wz_sum_out
    )
    
    # Benchmark NumPy
    print(f"Running {n_iters} iterations with NumPy...")
    start = time.perf_counter()
    for _ in range(n_iters):
        eta, mu, W_sum, Wz_sum = numpy_irls_iteration(
            Y, beta_intercept, beta_pert_coef, offset, alpha, log_min_mu, min_mu
        )
    numpy_time = (time.perf_counter() - start) / n_iters
    print(f"  NumPy: {numpy_time*1000:.3f} ms per iteration")
    
    # Benchmark Numba (fused with reduction)
    print(f"Running {n_iters} iterations with Numba (fused + reduction)...")
    start = time.perf_counter()
    for _ in range(n_iters):
        irls_iteration_with_reduction(
            Y, beta_intercept, beta_pert_coef, offset, alpha,
            log_min_mu, min_mu, eta_out, mu_out, W_sum_out, Wz_sum_out
        )
    numba_time = (time.perf_counter() - start) / n_iters
    print(f"  Numba: {numba_time*1000:.3f} ms per iteration")
    
    # Verify correctness
    eta_np, mu_np, W_sum_np, Wz_sum_np = numpy_irls_iteration(
        Y, beta_intercept, beta_pert_coef, offset, alpha, log_min_mu, min_mu
    )
    irls_iteration_with_reduction(
        Y, beta_intercept, beta_pert_coef, offset, alpha,
        log_min_mu, min_mu, eta_out, mu_out, W_sum_out, Wz_sum_out
    )
    
    eta_match = np.allclose(eta_np, eta_out, rtol=1e-10)
    mu_match = np.allclose(mu_np, mu_out, rtol=1e-10)
    W_sum_match = np.allclose(W_sum_np, W_sum_out, rtol=1e-10)
    Wz_sum_match = np.allclose(Wz_sum_np, Wz_sum_out, rtol=1e-10)
    
    print(f"\nCorrectness check:")
    print(f"  eta match: {eta_match}")
    print(f"  mu match: {mu_match}")
    print(f"  W_sum match: {W_sum_match}")
    print(f"  Wz_sum match: {Wz_sum_match}")
    
    speedup = numpy_time / numba_time
    print(f"\n>>> Speedup: {speedup:.2f}× <<<")
    
    return {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'numpy_ms': numpy_time * 1000,
        'numba_ms': numba_time * 1000,
        'speedup': speedup,
        'correct': eta_match and mu_match and W_sum_match and Wz_sum_match,
    }


if __name__ == "__main__":
    print("Phase 2 Numba Kernel Benchmark")
    print("=" * 60)
    
    # Test various sizes
    results = []
    
    # Small (like Adamson perturbation cells)
    results.append(benchmark_kernels(n_cells=300, n_genes=8000))
    
    # Medium
    results.append(benchmark_kernels(n_cells=500, n_genes=8000))
    
    # Large (like Replogle control cells if we needed to process them)
    results.append(benchmark_kernels(n_cells=1000, n_genes=8000))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Cells':<10} {'Genes':<10} {'NumPy (ms)':<15} {'Numba (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_cells']:<10} {r['n_genes']:<10} {r['numpy_ms']:<15.3f} {r['numba_ms']:<15.3f} {r['speedup']:<10.2f}×")
    
    all_correct = all(r['correct'] for r in results)
    print(f"\nAll results correct: {all_correct}")
