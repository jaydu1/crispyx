#!/usr/bin/env python
"""Step-level profiling script for nb_glm_test().

This script instruments nb_glm_test() internals to measure timing and memory
for each major step, enabling identification of performance bottlenecks.

Usage:
    python benchmarking/tools/profile_nb_glm.py data/Adamson_subset.h5ad

Output:
    - Step-level timing breakdown
    - RSS memory waterfall
    - Comparison with PyDESeq2 baseline (25.28s for Adamson_subset)

NOTE: This is temporary profiling code. It will be deleted after optimizations
are validated and integrated permanently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def get_rss_mb() -> float:
    """Get current process RSS memory in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)


def get_total_memory_mb() -> float:
    """Get total system memory in MB."""
    return psutil.virtual_memory().total / (1024 * 1024)


class StepProfiler:
    """Simple step-level profiler with timing and RSS memory tracking.
    
    Only profiles at the main-thread outer loop level (Option A).
    No worker-level aggregation for simplicity.
    """
    
    def __init__(self):
        self.timings: dict[str, float] = {}
        self.memory: dict[str, float] = {}  # RSS at step boundaries
        self.memory_deltas: dict[str, float] = {}
        self._current_step: str | None = None
        self._step_start: float | None = None
        self._baseline_rss: float | None = None
        self._peak_rss: float = 0.0
        
    def start(self, step: str) -> None:
        """Start timing a step and record memory baseline."""
        self._current_step = step
        self._step_start = time.perf_counter()
        current_rss = get_rss_mb()
        self.memory[f"{step}_start"] = current_rss
        if self._baseline_rss is None:
            self._baseline_rss = current_rss
        self._peak_rss = max(self._peak_rss, current_rss)
        
    def stop(self, step: str | None = None) -> None:
        """Stop timing current step and record memory."""
        step = step or self._current_step
        if step is None or self._step_start is None:
            return
        elapsed = time.perf_counter() - self._step_start
        self.timings[step] = elapsed
        
        current_rss = get_rss_mb()
        self.memory[f"{step}_end"] = current_rss
        self.memory_deltas[step] = current_rss - self.memory.get(f"{step}_start", current_rss)
        self._peak_rss = max(self._peak_rss, current_rss)
        
        self._current_step = None
        self._step_start = None
        
    def get_results(self) -> dict[str, Any]:
        """Get profiling results as a dict."""
        total_time = sum(self.timings.values())
        return {
            "timings": self.timings.copy(),
            "total_time": total_time,
            "memory": self.memory.copy(),
            "memory_deltas": self.memory_deltas.copy(),
            "baseline_rss_mb": self._baseline_rss,
            "peak_rss_mb": self._peak_rss,
            "available_memory_mb": get_available_memory_mb(),
            "total_memory_mb": get_total_memory_mb(),
        }
        
    def print_report(self, pydeseq2_baseline: float = 25.28) -> None:
        """Print formatted profiling report."""
        results = self.get_results()
        total = results["total_time"]
        
        print("\n" + "=" * 70)
        print("STEP-LEVEL PROFILING RESULTS")
        print("=" * 70)
        
        print(f"\n{'Step':<25} {'Time (s)':>10} {'% Total':>10} {'RSS Δ (MB)':>12}")
        print("-" * 60)
        
        for step, time_s in results["timings"].items():
            pct = 100 * time_s / total if total > 0 else 0
            delta = results["memory_deltas"].get(step, 0)
            print(f"{step:<25} {time_s:>10.2f} {pct:>9.1f}% {delta:>+11.1f}")
            
        print("-" * 60)
        print(f"{'TOTAL':<25} {total:>10.2f}")
        
        print(f"\n{'Memory Metrics':}")
        print(f"  Baseline RSS: {results['baseline_rss_mb']:.1f} MB")
        print(f"  Peak RSS: {results['peak_rss_mb']:.1f} MB")
        print(f"  Available: {results['available_memory_mb']:.1f} MB")
        print(f"  System Total: {results['total_memory_mb']:.1f} MB")
        
        if pydeseq2_baseline > 0:
            gap = total / pydeseq2_baseline
            print(f"\n{'Comparison with PyDESeq2':}")
            print(f"  PyDESeq2 fit_NB: {pydeseq2_baseline:.2f}s")
            print(f"  CRISPYx fit_NB: {total:.2f}s")
            print(f"  Gap: {gap:.1f}× slower")
            
        # Decision points
        print(f"\n{'Decision Points':}")
        control_load_pct = 100 * results["timings"].get("control_load", 0) / total if total > 0 else 0
        if control_load_pct < 5:
            print(f"  ✓ control_load < 5% ({control_load_pct:.1f}%) → Skip I/O optimization")
        else:
            print(f"  ⚠ control_load ≥ 5% ({control_load_pct:.1f}%) → Consider chunk_size tuning")
            
        # Identify top bottlenecks
        sorted_steps = sorted(results["timings"].items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'Top Bottlenecks':}")
        for i, (step, time_s) in enumerate(sorted_steps[:3], 1):
            pct = 100 * time_s / total if total > 0 else 0
            print(f"  {i}. {step}: {time_s:.2f}s ({pct:.1f}%)")
            
        print("=" * 70)


def run_profiled_nb_glm_test(
    path: str | Path,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Any, StepProfiler]:
    """Run nb_glm_test with step-level profiling instrumentation.
    
    This function wraps the internal steps of nb_glm_test() with timing
    calls to measure each phase independently.
    
    Returns
    -------
    result : RankGenesGroupsResult
        The differential expression results.
    profiler : StepProfiler
        Profiler with step-level timing and memory data.
    """
    # Import here to avoid circular imports
    import scipy.sparse as sp
    import pandas as pd
    from crispyx.de import (
        read_backed,
        ensure_gene_symbol_column,
        resolve_control_label,
        _resolve_candidates,
        _median_of_ratios_size_factors,
        _validate_size_factors,
        build_design_matrix,
        NBGLMBatchFitter,
        RankGenesGroupsResult,
        ControlStatisticsCache,
    )
    from crispyx.glm import (
        fit_dispersion_trend,
        estimate_dispersion_map,
    )
    
    path = Path(path)
    profiler = StepProfiler()
    
    # =========================================================================
    # Step 1: Initialization & Validation
    # =========================================================================
    profiler.start("init")
    
    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, None)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(f"Perturbation column '{perturbation_column}' not found")
        obs_df = backed.obs[[perturbation_column]].copy()
        labels = obs_df[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, None)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        n_cells_total = obs_df.shape[0]
    finally:
        backed.file.close()
        
    profiler.stop("init")
    
    # =========================================================================
    # Step 2: Size Factor Computation
    # =========================================================================
    profiler.start("size_factors")
    
    size_factors = _median_of_ratios_size_factors(path, scale=True)  # Use default chunk_size=2048
    offset = np.log(np.clip(size_factors, 1e-8, None))
    
    profiler.stop("size_factors")
    
    # =========================================================================
    # Step 3: Control Matrix Loading
    # =========================================================================
    profiler.start("control_load")
    
    backed = read_backed(path)
    try:
        control_matrix = backed.X[control_mask, :]
        if sp.issparse(control_matrix):
            control_matrix = sp.csr_matrix(control_matrix, dtype=np.float64)
        else:
            control_matrix = np.asarray(control_matrix, dtype=np.float64)
    finally:
        backed.file.close()
    
    # Compute control expression counts
    if sp.issparse(control_matrix):
        control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
    else:
        control_expr_counts = np.sum(control_matrix > 0, axis=0)
        
    profiler.stop("control_load")
    
    # =========================================================================
    # Step 4: Initial Coefficient Fitting (first perturbation only for profiling)
    # =========================================================================
    profiler.start("initial_fit")
    
    # For profiling, we just time the first perturbation as representative
    label = candidates[0]
    group_mask = labels == label
    subset_mask = control_mask | group_mask
    group_n = int(group_mask.sum())
    subset_n = control_n + group_n
    
    # Load perturbation cells
    backed = read_backed(path)
    try:
        group_matrix = backed.X[group_mask, :]
        if sp.issparse(group_matrix):
            group_matrix = sp.csr_matrix(group_matrix, dtype=np.float64)
        else:
            group_matrix = np.asarray(group_matrix, dtype=np.float64)
    finally:
        backed.file.close()
    
    # Build combined matrix
    if sp.issparse(control_matrix) and sp.issparse(group_matrix):
        subset_matrix = sp.vstack([control_matrix, group_matrix])
    else:
        ctrl = control_matrix.toarray() if sp.issparse(control_matrix) else control_matrix
        grp = group_matrix.toarray() if sp.issparse(group_matrix) else group_matrix
        subset_matrix = np.vstack([ctrl, grp])
    
    # Build design matrix
    perturbation_indicator = np.concatenate([
        np.zeros(control_n, dtype=np.float64),
        np.ones(group_n, dtype=np.float64)
    ])
    design = np.column_stack([np.ones(subset_n), perturbation_indicator])
    subset_offset = offset[subset_mask]
    
    # Fit batch
    # NOTE: Production code uses dispersion_method="moments" for initial fit
    # (Cox-Reid refinement happens later in dispersion_map step)
    batch_fitter = NBGLMBatchFitter(
        design=design,
        offset=subset_offset,
        max_iter=25,
        tol=1e-6,
        dispersion_method="moments",  # Match production code
        min_mu=0.5,
        min_total_count=1.0,
    )
    
    batch_result = batch_fitter.fit_batch(
        subset_matrix,
        gene_batch_size="auto",
        use_numba=True,
    )
    
    profiler.stop("initial_fit")
    
    # =========================================================================
    # Step 5a: Dispersion MoM
    # =========================================================================
    profiler.start("dispersion_mom")
    
    # Compute MoM dispersion (simplified for profiling)
    from crispyx.de import _compute_mom_dispersion_batched
    
    Y_pert_dense = group_matrix.toarray() if sp.issparse(group_matrix) else np.asarray(group_matrix)
    # Ensure control_matrix is ndarray not matrix
    Y_control_dense = control_matrix.toarray() if sp.issparse(control_matrix) else np.asarray(control_matrix)
    if hasattr(Y_control_dense, 'A'):  # Handle np.matrix
        Y_control_dense = np.asarray(Y_control_dense)
    control_offset_subset = offset[control_mask]
    pert_offset = offset[group_mask]
    
    beta0_all = batch_result.coef[:, 0]
    beta1_all = batch_result.coef[:, 1]
    
    mom_disp = _compute_mom_dispersion_batched(
        Y_control=Y_control_dense,
        Y_pert=Y_pert_dense,
        control_offset=control_offset_subset,
        pert_offset=pert_offset,
        beta0=beta0_all,
        beta1=beta1_all,
        converged=batch_result.converged,
        gene_batch_size=5000,
    )
    
    profiler.stop("dispersion_mom")
    
    # =========================================================================
    # Step 5b: Dispersion Trend
    # =========================================================================
    profiler.start("dispersion_trend")
    
    # Compute mean expression for trend fitting
    mean_expr = np.asarray(subset_matrix.mean(axis=0)).ravel() if sp.issparse(subset_matrix) else subset_matrix.mean(axis=0)
    valid_for_trend = np.isfinite(mom_disp) & (mom_disp > 0) & (mean_expr > 0)
    
    if np.sum(valid_for_trend) > 10:
        trend_disp = fit_dispersion_trend(
            mean_expr[valid_for_trend],
            mom_disp[valid_for_trend],
            fit_type="parametric",
        )
        # Predict trend for all genes
        dispersion_trend = np.full(n_genes, np.nan, dtype=np.float64)
        dispersion_trend[valid_for_trend] = trend_disp
    else:
        dispersion_trend = mom_disp.copy()
        
    profiler.stop("dispersion_trend")
    
    # =========================================================================
    # Step 5c: Dispersion MAP (fused grid search + Brent refinement)
    # =========================================================================
    profiler.start("dispersion_map")
    
    # Compute mu for MAP estimation
    # eta = design @ beta + offset
    eta = (design @ batch_result.coef.T).T + subset_offset  # (n_genes, n_cells) transposed properly
    if hasattr(subset_matrix, 'toarray'):
        subset_dense = np.asarray(subset_matrix.toarray(), dtype=np.float64)
    else:
        subset_dense = np.asarray(subset_matrix, dtype=np.float64)
    
    # Compute mu from coefficients
    # mu[i,j] = exp(beta0[j] + beta1[j]*indicator[i] + offset[i])
    mu = np.exp(batch_result.coef[:, 0:1] + batch_result.coef[:, 1:2] * perturbation_indicator[np.newaxis, :] + subset_offset[np.newaxis, :])
    mu = mu.T  # (n_cells, n_genes)
    
    # MAP dispersion estimation with fused grid + Brent refinement
    # Uses default n_grid=25 which is optimal with Brent refinement
    map_disp = estimate_dispersion_map(
        Y=subset_dense,
        mu=mu,
        trend=dispersion_trend,
        refine=True,  # Fused Numba Brent refinement
    )
    
    profiler.stop("dispersion_map")
    
    # =========================================================================
    # Step 6: Final Fit / SE Recomputation (simplified timing)
    # =========================================================================
    profiler.start("final_fit")
    
    # Simplified SE computation timing - just measure matrix operations
    # The actual _compute_se_batched has different signature, so we approximate
    from scipy.linalg import solve
    
    # Compute Fisher information matrix for a few genes as representative timing
    n_sample_genes = min(1000, n_genes)
    for j in range(n_sample_genes):
        # This simulates the SE computation overhead
        mu_j = mu[:, j]
        disp_j = map_disp[j] if np.isfinite(map_disp[j]) else 0.1
        W = mu_j / (1 + mu_j * disp_j)
        XtWX = design.T @ (design * W[:, np.newaxis])
        try:
            inv_XtWX = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            pass
    
    profiler.stop("final_fit")
    
    # =========================================================================
    # Step 7: Output (minimal for profiling)
    # =========================================================================
    profiler.start("output")
    
    # Just measure time to create result structure (no file I/O for profiling)
    result_dict = {
        "n_genes": n_genes,
        "n_perturbations": len(candidates),
        "control_n": control_n,
    }
    
    profiler.stop("output")
    
    return result_dict, profiler


def main():
    parser = argparse.ArgumentParser(
        description="Step-level profiling for nb_glm_test()"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input h5ad file",
    )
    parser.add_argument(
        "--perturbation-column",
        type=str,
        default="perturbation",
        help="Column name for perturbation labels (default: perturbation)",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default=None,
        help="Control group label (default: auto-detect)",
    )
    parser.add_argument(
        "--pydeseq2-baseline",
        type=float,
        default=25.28,
        help="PyDESeq2 fit_NB time for comparison (default: 25.28s for Adamson_subset)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save profiling results as JSON",
    )
    
    args = parser.parse_args()
    
    print(f"Profiling nb_glm_test() on: {args.input_path}")
    print(f"Perturbation column: {args.perturbation_column}")
    
    result, profiler = run_profiled_nb_glm_test(
        args.input_path,
        perturbation_column=args.perturbation_column,
        control_label=args.control_label,
    )
    
    profiler.print_report(pydeseq2_baseline=args.pydeseq2_baseline)
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profiler.get_results(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
