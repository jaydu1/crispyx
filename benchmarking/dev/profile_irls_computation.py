#!/usr/bin/env python3
"""Profile IRLS computation bottlenecks in NB-GLM.

This script provides detailed profiling of the fit_batch_with_control_cache function
to identify which operations are slow on large datasets like Replogle-GW-k562.

Usage:
    python -m benchmarking.tools.profile_irls_computation --dataset Replogle-GW-k562 --n-perturbations 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    """Detailed timing breakdown for IRLS operations."""
    # Data loading
    load_data: float = 0.0
    densify: float = 0.0
    
    # Per-iteration breakdown
    eta_computation: float = 0.0
    mu_computation: float = 0.0
    weights_computation: float = 0.0
    working_response: float = 0.0
    xtwx_xtwz: float = 0.0
    solve_system: float = 0.0
    convergence_check: float = 0.0
    dispersion_update: float = 0.0
    
    # Post-IRLS
    standard_errors: float = 0.0
    
    # Counts
    n_iterations: int = 0
    n_genes: int = 0
    n_cells_control: int = 0
    n_cells_pert: int = 0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    def total_irls(self) -> float:
        return (
            self.eta_computation + self.mu_computation + 
            self.weights_computation + self.working_response +
            self.xtwx_xtwz + self.solve_system + 
            self.convergence_check + self.dispersion_update
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "load_data": self.load_data,
            "densify": self.densify,
            "eta_computation": self.eta_computation,
            "mu_computation": self.mu_computation,
            "weights_computation": self.weights_computation,
            "working_response": self.working_response,
            "xtwx_xtwz": self.xtwx_xtwz,
            "solve_system": self.solve_system,
            "convergence_check": self.convergence_check,
            "dispersion_update": self.dispersion_update,
            "standard_errors": self.standard_errors,
            "total_irls": self.total_irls(),
            "n_iterations": self.n_iterations,
            "n_genes": self.n_genes,
            "n_cells_control": self.n_cells_control,
            "n_cells_pert": self.n_cells_pert,
            "peak_memory_mb": self.peak_memory_mb,
        }


def profile_irls_iteration(
    Y_control: np.ndarray,
    Y_pert: np.ndarray,
    offset_control: np.ndarray,
    offset_pert: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    ridge_penalty: float = 1e-6,
) -> TimingBreakdown:
    """Profile a single IRLS fitting with detailed timing breakdown."""
    import psutil
    
    timing = TimingBreakdown()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    n_control, n_valid = Y_control.shape
    n_pert = Y_pert.shape[0]
    n_total = n_control + n_pert
    n_features = 2
    
    timing.n_genes = n_valid
    timing.n_cells_control = n_control
    timing.n_cells_pert = n_pert
    
    # Work arrays
    mu_control = np.empty((n_control, n_valid), dtype=np.float64)
    mu_pert = np.empty((n_pert, n_valid), dtype=np.float64)
    W_control = np.empty_like(mu_control)
    W_pert = np.empty_like(mu_pert)
    
    log_min_mu = np.log(min_mu)
    gene_converged = np.zeros(n_valid, dtype=bool)
    
    for iteration in range(1, max_iter + 1):
        timing.n_iterations = iteration
        
        beta_intercept = beta[0, :]
        beta_pert_coef = beta[1, :]
        
        # === ETA COMPUTATION ===
        t0 = time.perf_counter()
        eta_control = beta_intercept[None, :] + offset_control
        np.clip(eta_control, log_min_mu, 20.0, out=eta_control)
        eta_pert = beta_intercept[None, :] + beta_pert_coef[None, :] + offset_pert
        np.clip(eta_pert, log_min_mu, 20.0, out=eta_pert)
        timing.eta_computation += time.perf_counter() - t0
        
        # === MU COMPUTATION ===
        t0 = time.perf_counter()
        np.exp(eta_control, out=mu_control)
        np.maximum(mu_control, min_mu, out=mu_control)
        np.exp(eta_pert, out=mu_pert)
        np.maximum(mu_pert, min_mu, out=mu_pert)
        timing.mu_computation += time.perf_counter() - t0
        
        # === WEIGHTS COMPUTATION ===
        t0 = time.perf_counter()
        var_control = mu_control + alpha[None, :] * mu_control * mu_control
        np.divide(mu_control * mu_control, np.maximum(var_control, min_mu), out=W_control)
        var_pert = mu_pert + alpha[None, :] * mu_pert * mu_pert
        np.divide(mu_pert * mu_pert, np.maximum(var_pert, min_mu), out=W_pert)
        timing.weights_computation += time.perf_counter() - t0
        
        # === WORKING RESPONSE ===
        t0 = time.perf_counter()
        z_control = eta_control + (Y_control - mu_control) / np.maximum(mu_control, min_mu)
        z_pert = eta_pert + (Y_pert - mu_pert) / np.maximum(mu_pert, min_mu)
        z_control_centered = z_control - offset_control
        z_pert_centered = z_pert - offset_pert
        timing.working_response += time.perf_counter() - t0
        
        # === XᵀWX AND XᵀWz ===
        t0 = time.perf_counter()
        W_control_sum = np.sum(W_control, axis=0)
        W_pert_sum = np.sum(W_pert, axis=0)
        Wz_control_sum = np.sum(W_control * z_control_centered, axis=0)
        Wz_pert_sum = np.sum(W_pert * z_pert_centered, axis=0)
        xtwx_00 = W_control_sum + W_pert_sum
        xtwx_01 = W_pert_sum
        xtwx_11 = W_pert_sum
        xtwz_0 = Wz_control_sum + Wz_pert_sum
        xtwz_1 = Wz_pert_sum
        timing.xtwx_xtwz += time.perf_counter() - t0
        
        # === SOLVE SYSTEM ===
        t0 = time.perf_counter()
        xtwx_00 = xtwx_00 + ridge_penalty
        xtwx_11 = xtwx_11 + ridge_penalty
        det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        beta_new_0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
        beta_new_1 = (xtwx_00 * xtwz_1 - xtwx_01 * xtwz_0) / det
        beta_new = np.vstack([beta_new_0, beta_new_1])
        timing.solve_system += time.perf_counter() - t0
        
        # === CONVERGENCE CHECK ===
        t0 = time.perf_counter()
        beta_diff = np.max(np.abs(beta_new - beta), axis=0)
        newly_converged = (beta_diff < tol) & ~gene_converged
        gene_converged |= newly_converged
        beta = beta_new
        timing.convergence_check += time.perf_counter() - t0
        
        # === DISPERSION UPDATE ===
        t0 = time.perf_counter()
        resid_control = Y_control - mu_control
        resid_pert = Y_pert - mu_pert
        numerator = (
            np.sum((resid_control ** 2 - Y_control) / np.maximum(mu_control ** 2, min_mu), axis=0)
            + np.sum((resid_pert ** 2 - Y_pert) / np.maximum(mu_pert ** 2, min_mu), axis=0)
        )
        dof = max(n_total - n_features, 1)
        alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
        alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
        timing.dispersion_update += time.perf_counter() - t0
        
        if np.all(gene_converged):
            break
    
    # === STANDARD ERRORS ===
    t0 = time.perf_counter()
    W_control_sum = np.sum(W_control, axis=0)
    W_pert_sum = np.sum(W_pert, axis=0)
    M00 = W_control_sum + W_pert_sum
    M01 = W_pert_sum
    M11 = W_pert_sum
    Mr00 = M00 + ridge_penalty
    Mr01 = M01
    Mr11 = M11 + ridge_penalty
    det_r = Mr00 * Mr11 - Mr01 * Mr01
    det_r = np.where(np.abs(det_r) < 1e-12, 1e-12, det_r)
    H00 = Mr11 / det_r
    H01 = -Mr01 / det_r
    H11 = Mr00 / det_r
    Hc0 = H01
    Hc1 = H11
    var_pert_coef = Hc0**2 * M00 + 2 * Hc0 * Hc1 * M01 + Hc1**2 * M11
    se_pert = np.sqrt(np.maximum(var_pert_coef, 1e-12))
    timing.standard_errors = time.perf_counter() - t0
    
    # Memory tracking
    current_memory = process.memory_info().rss / 1024 / 1024
    timing.peak_memory_mb = current_memory - start_memory
    
    return timing


def get_dataset_path(dataset_name: str) -> Path:
    """Resolve dataset name to path."""
    base_dir = Path(__file__).parent.parent.parent
    
    candidates = [
        base_dir / "data" / f"{dataset_name}.h5ad",
        base_dir / ".cache" / f"{dataset_name}.h5ad",
        base_dir / "benchmarking" / ".cache" / f"{dataset_name}.h5ad",
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"standardized_{dataset_name}.h5ad",
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"{dataset_name}.h5ad",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    if Path(dataset_name).exists():
        return Path(dataset_name)
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Checked:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def profile_dataset(
    dataset_path: Path,
    n_perturbations: int = 2,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
) -> list[dict[str, Any]]:
    """Profile IRLS computation on a dataset."""
    import anndata as ad
    from crispyx.data import read_backed
    
    logger.info(f"Profiling {dataset_path.name}...")
    
    # Load dataset info
    backed = read_backed(dataset_path)
    n_cells, n_genes = backed.shape
    
    # Get perturbation labels
    labels = backed.obs[perturbation_column].values
    unique_labels = np.unique(labels)
    
    # Infer control label
    if control_label is None:
        for candidate in ["control", "Control", "CONTROL", "ctrl", "NT", "non-targeting"]:
            if candidate in unique_labels:
                control_label = candidate
                logger.info(f"Inferred control label '{control_label}'")
                break
    
    if control_label is None:
        raise ValueError("Could not infer control label")
    
    # Get perturbation-only labels
    pert_labels = [l for l in unique_labels if l != control_label]
    pert_labels = sorted(pert_labels)[:n_perturbations]
    
    logger.info(f"  {n_cells:,} cells, {n_genes:,} genes, {len(unique_labels):,} perturbations")
    logger.info(f"  Testing {n_perturbations} perturbations: {pert_labels}")
    
    # Load control cells
    control_mask = labels == control_label
    n_control = int(control_mask.sum())
    logger.info(f"  Loading {n_control:,} control cells...")
    
    t0 = time.perf_counter()
    control_matrix = backed.X[control_mask, :]
    load_time = time.perf_counter() - t0
    logger.info(f"  Control load time: {load_time:.3f}s")
    
    # Densify control
    t0 = time.perf_counter()
    if sp.issparse(control_matrix):
        Y_control = np.asarray(control_matrix.toarray(), dtype=np.float64)
    else:
        Y_control = np.asarray(control_matrix, dtype=np.float64)
    densify_time = time.perf_counter() - t0
    logger.info(f"  Control densify time: {densify_time:.3f}s")
    logger.info(f"  Control matrix size: {Y_control.nbytes / 1024 / 1024:.1f} MB")
    
    # Compute control offset (log size factors)
    control_counts = Y_control.sum(axis=1)
    control_sf = control_counts / np.median(control_counts)
    offset_control = np.log(np.clip(control_sf, 1e-8, None))[:, None]
    
    # Initial estimates
    normalized_control = Y_control / np.exp(offset_control)
    mean_expr = normalized_control.mean(axis=0)
    beta_intercept = np.log(np.maximum(mean_expr * np.exp(offset_control.mean()), 1e-10))
    alpha_initial = np.full(n_genes, 0.1, dtype=np.float64)
    
    results = []
    
    for label in pert_labels:
        logger.info(f"\n  === Profiling perturbation: {label} ===")
        
        # Load perturbation cells
        pert_mask = labels == label
        n_pert = int(pert_mask.sum())
        logger.info(f"  Loading {n_pert:,} perturbation cells...")
        
        timing = TimingBreakdown()
        
        t0 = time.perf_counter()
        pert_matrix = backed.X[pert_mask, :]
        timing.load_data = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if sp.issparse(pert_matrix):
            Y_pert = np.asarray(pert_matrix.toarray(), dtype=np.float64)
        else:
            Y_pert = np.asarray(pert_matrix, dtype=np.float64)
        timing.densify = time.perf_counter() - t0
        
        logger.info(f"    Load: {timing.load_data:.3f}s, Densify: {timing.densify:.3f}s")
        logger.info(f"    Pert matrix size: {Y_pert.nbytes / 1024 / 1024:.1f} MB")
        
        # Compute pert offset
        pert_counts = Y_pert.sum(axis=1)
        pert_sf = pert_counts / np.median(pert_counts)
        offset_pert = np.log(np.clip(pert_sf, 1e-8, None))[:, None]
        
        # Initialize beta
        beta = np.zeros((2, n_genes), dtype=np.float64)
        beta[0, :] = beta_intercept
        
        # Profile IRLS
        logger.info(f"    Running IRLS profiling...")
        t0 = time.perf_counter()
        irls_timing = profile_irls_iteration(
            Y_control, Y_pert, offset_control, offset_pert,
            alpha_initial.copy(), beta,
            max_iter=25, tol=1e-6,
        )
        total_irls = time.perf_counter() - t0
        
        # Merge timings
        timing.eta_computation = irls_timing.eta_computation
        timing.mu_computation = irls_timing.mu_computation
        timing.weights_computation = irls_timing.weights_computation
        timing.working_response = irls_timing.working_response
        timing.xtwx_xtwz = irls_timing.xtwx_xtwz
        timing.solve_system = irls_timing.solve_system
        timing.convergence_check = irls_timing.convergence_check
        timing.dispersion_update = irls_timing.dispersion_update
        timing.standard_errors = irls_timing.standard_errors
        timing.n_iterations = irls_timing.n_iterations
        timing.n_genes = n_genes
        timing.n_cells_control = n_control
        timing.n_cells_pert = n_pert
        timing.peak_memory_mb = irls_timing.peak_memory_mb
        
        # Print breakdown
        logger.info(f"\n    IRLS TIMING BREAKDOWN ({timing.n_iterations} iterations):")
        logger.info(f"    {'Operation':<25} {'Time (s)':<12} {'% Total':<10} {'Per Iter (ms)':<15}")
        logger.info(f"    {'-'*62}")
        
        ops = [
            ("eta_computation", timing.eta_computation),
            ("mu_computation", timing.mu_computation),
            ("weights_computation", timing.weights_computation),
            ("working_response", timing.working_response),
            ("xtwx_xtwz", timing.xtwx_xtwz),
            ("solve_system", timing.solve_system),
            ("convergence_check", timing.convergence_check),
            ("dispersion_update", timing.dispersion_update),
        ]
        
        total_irls_sum = sum(t for _, t in ops)
        for name, t in ops:
            pct = 100 * t / total_irls_sum if total_irls_sum > 0 else 0
            per_iter_ms = 1000 * t / timing.n_iterations if timing.n_iterations > 0 else 0
            logger.info(f"    {name:<25} {t:<12.4f} {pct:<10.1f} {per_iter_ms:<15.2f}")
        
        logger.info(f"    {'-'*62}")
        logger.info(f"    {'TOTAL IRLS':<25} {total_irls_sum:<12.4f} {'100.0':<10}")
        logger.info(f"    {'Standard Errors':<25} {timing.standard_errors:<12.4f}")
        logger.info(f"    {'Data Load':<25} {timing.load_data:<12.4f}")
        logger.info(f"    {'Densify':<25} {timing.densify:<12.4f}")
        logger.info(f"    {'GRAND TOTAL':<25} {total_irls + timing.load_data + timing.densify:<12.4f}")
        
        # Compute FLOPs estimate
        n_total = n_control + n_pert
        flops_per_iter = n_total * n_genes * 50  # Rough estimate: 50 ops per cell-gene
        total_flops = flops_per_iter * timing.n_iterations
        gflops = total_flops / 1e9
        gflops_per_sec = gflops / total_irls_sum if total_irls_sum > 0 else 0
        
        logger.info(f"\n    PERFORMANCE METRICS:")
        logger.info(f"    Cells: {n_total:,} ({n_control:,} control + {n_pert:,} pert)")
        logger.info(f"    Genes: {n_genes:,}")
        logger.info(f"    Matrix elements: {n_total * n_genes:,} ({n_total * n_genes * 8 / 1024 / 1024:.1f} MB)")
        logger.info(f"    Estimated GFLOP/s: {gflops_per_sec:.2f}")
        logger.info(f"    Time per 1M cell-genes: {1e6 * total_irls_sum / (n_total * n_genes):.3f}s")
        
        result = {
            "perturbation": label,
            "n_cells_pert": n_pert,
            "n_cells_control": n_control,
            "n_genes": n_genes,
            **timing.to_dict(),
            "total_time": total_irls + timing.load_data + timing.densify,
        }
        results.append(result)
    
    backed.file.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile IRLS computation bottlenecks")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--n-perturbations", type=int, default=2, help="Number of perturbations to test")
    parser.add_argument("--perturbation-column", default="perturbation", help="Column name for perturbation labels")
    parser.add_argument("--control-label", default=None, help="Control label (auto-detected if not specified)")
    parser.add_argument("--output-dir", default="benchmarking/results/irls_profiling", help="Output directory")
    args = parser.parse_args()
    
    dataset_path = get_dataset_path(args.dataset)
    logger.info(f"Using dataset: {dataset_path}")
    
    results = profile_dataset(
        dataset_path,
        n_perturbations=args.n_perturbations,
        perturbation_column=args.perturbation_column,
        control_label=args.control_label,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"irls_profiling_{timestamp}.json"
    
    summary = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "timestamp": timestamp,
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Perturbation':<15} {'Cells':<10} {'Iterations':<12} {'IRLS (s)':<12} {'Total (s)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['perturbation']:<15} {r['n_cells_pert']:<10} {r['n_iterations']:<12} {r['total_irls']:<12.3f} {r['total_time']:<12.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
