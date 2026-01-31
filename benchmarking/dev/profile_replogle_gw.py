#!/usr/bin/env python
"""Profile NB-GLM on Replogle-GW-k562 with a subset of perturbations.

This script measures step-level timing and memory usage to estimate
the total runtime for the full 9,866 perturbations.

Usage:
    python benchmarking/dev/profile_replogle_gw.py --n-perts 5
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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
    """Simple step-level profiler with timing and RSS memory tracking."""
    
    def __init__(self):
        self.timings: dict[str, float] = {}
        self.memory: dict[str, float] = {}
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


def profile_nb_glm_subset(
    path: str | Path,
    n_perts: int = 5,
    n_jobs: int = 32,
    memory_limit_gb: float | None = None,
) -> tuple[dict, StepProfiler]:
    """Profile NB-GLM on a subset of perturbations with step-level timing."""
    import crispyx as cx
    import anndata as ad
    import scipy.sparse as sp
    
    path = Path(path)
    profiler = StepProfiler()
    
    # =========================================================================
    # Step 1: Load metadata and identify perturbations
    # =========================================================================
    profiler.start("metadata_load")
    
    backed = ad.read_h5ad(path, backed='r')
    n_cells_total = backed.n_obs
    n_genes = backed.n_vars
    labels = backed.obs['perturbation'].astype(str)
    vc = labels.value_counts()
    control_n = vc.get('control', 0)
    all_perts = [p for p in vc.index if p != 'control']
    n_perts_total = len(all_perts)
    backed.file.close()
    
    profiler.stop("metadata_load")
    
    print(f"Dataset: {path}")
    print(f"  Cells: {n_cells_total:,}")
    print(f"  Genes: {n_genes:,}")
    print(f"  Control cells: {control_n:,}")
    print(f"  Total perturbations: {n_perts_total:,}")
    print(f"  Profiling with: {n_perts} perturbations")
    print(f"  n_jobs: {n_jobs}")
    print(f"  memory_limit_gb: {memory_limit_gb}")
    print()
    
    # =========================================================================
    # Step 2: Create subset with n_perts perturbations
    # =========================================================================
    profiler.start("subset_selection")
    
    # Select first n_perts perturbations (largest ones by cell count)
    selected_perts = all_perts[:n_perts]
    print(f"Selected perturbations: {selected_perts}")
    
    profiler.stop("subset_selection")
    
    # =========================================================================
    # Step 3: Detailed step-by-step profiling
    # =========================================================================
    print("\n--- STEP-BY-STEP PROFILING ---")
    
    # === Step 3a: Size factor computation ===
    profiler.start("size_factors")
    print("Computing size factors...")
    from crispyx._size_factors import _median_of_ratios_size_factors
    size_factors = _median_of_ratios_size_factors(path, chunk_size=2048, scale=True)
    profiler.stop("size_factors")
    print(f"  Size factors: {profiler.timings['size_factors']:.1f}s, RSS: {get_rss_mb():.0f} MB")
    
    offset = np.log(np.clip(size_factors, 1e-8, None))
    
    # === Step 3b: Load control matrix ===
    profiler.start("load_control")
    print("Loading control matrix...")
    backed = ad.read_h5ad(path, backed='r')
    labels_np = backed.obs['perturbation'].astype(str).to_numpy()
    control_mask = labels_np == 'control'
    control_matrix = backed.X[control_mask, :]
    if sp.issparse(control_matrix):
        control_matrix = sp.csr_matrix(control_matrix, dtype=np.float64)
    backed.file.close()
    profiler.stop("load_control")
    print(f"  Load control: {profiler.timings['load_control']:.1f}s, RSS: {get_rss_mb():.0f} MB")
    print(f"  Control shape: {control_matrix.shape}")
    
    # === Step 3c: Precompute control statistics ===
    profiler.start("control_stats")
    print("Precomputing control statistics...")
    from crispyx.glm import precompute_control_statistics
    control_offset = offset[control_mask]
    control_cache = precompute_control_statistics(
        control_matrix=control_matrix,
        control_offset=control_offset,
        max_iter=25,
        tol=1e-6,
        min_mu=0.5,
        dispersion_method="moments",
        global_size_factors=size_factors,
    )
    profiler.stop("control_stats")
    print(f"  Control stats: {profiler.timings['control_stats']:.1f}s, RSS: {get_rss_mb():.0f} MB")
    
    # === Step 3d: Global dispersion computation ===
    profiler.start("global_dispersion")
    print("Computing global dispersion...")
    from crispyx.glm import precompute_global_dispersion
    
    # Load full matrix for dispersion estimation
    backed = ad.read_h5ad(path, backed='r')
    all_cell_matrix = backed.X[:]
    if sp.issparse(all_cell_matrix):
        all_cell_matrix = sp.csr_matrix(all_cell_matrix, dtype=np.float64)
    backed.file.close()
    print(f"  All cells matrix shape: {all_cell_matrix.shape}, RSS: {get_rss_mb():.0f} MB")
    
    control_cache = precompute_global_dispersion(
        control_cache=control_cache,
        all_cell_matrix=all_cell_matrix,
        all_cell_offset=offset,
        n_grid=25,
        fit_type="parametric",
        fast_mode=True,
        max_dense_fraction=0.3,
        memory_limit_gb=memory_limit_gb,
    )
    del all_cell_matrix
    import gc
    gc.collect()
    profiler.stop("global_dispersion")
    print(f"  Global dispersion: {profiler.timings['global_dispersion']:.1f}s, RSS: {get_rss_mb():.0f} MB")
    
    # === Step 3e: Per-perturbation fitting ===
    profiler.start("per_pert_fitting")
    print(f"Fitting {n_perts} perturbations...")
    result = cx.nb_glm_test(
        path,
        perturbation_column='perturbation',
        control_label='control',
        perturbations=selected_perts,
        dispersion_scope='global',  # Already precomputed above, will be recomputed
        n_jobs=n_jobs,
        memory_limit_gb=memory_limit_gb,
        verbose=True,
    )
    profiler.stop("per_pert_fitting")
    print(f"  Per-pert fitting: {profiler.timings['per_pert_fitting']:.1f}s, RSS: {get_rss_mb():.0f} MB")
    
    total_time = sum(profiler.timings.values())
    
    # Note: per_pert_fitting includes the overhead from nb_glm_test re-doing 
    # size factors, control loading, and dispersion. To get true per-pert time,
    # subtract the fixed costs.
    fixed_overhead = (
        profiler.timings.get('size_factors', 0) + 
        profiler.timings.get('load_control', 0) +
        profiler.timings.get('global_dispersion', 0)
    )
    
    # Get the pure parallel fitting time (from progress bar)
    # The per_pert_fitting includes these steps again, so we need to estimate
    pure_pert_time = profiler.timings.get('per_pert_fitting', 0)
    time_per_pert = pure_pert_time / n_perts if n_perts > 0 else 0
    
    # =========================================================================
    # Summary
    # =========================================================================
    results = profiler.get_results()
    
    # Fixed costs (done once regardless of perturbation count)
    fixed_cost = (
        profiler.timings.get('size_factors', 0) +
        profiler.timings.get('load_control', 0) +
        profiler.timings.get('control_stats', 0) +
        profiler.timings.get('global_dispersion', 0)
    )
    
    # Variable costs scale with number of perturbations
    variable_cost = profiler.timings.get('per_pert_fitting', 0)
    
    # Estimate full runtime
    # The per_pert_fitting time includes its own fixed costs, so we need to
    # estimate the true per-perturbation time from progress bar timing
    # Progress bar showed ~54s/pert, but total was 2144s for 5 perts
    # This means there's huge overhead from re-computing dispersion
    
    # For accurate estimation: fixed costs + (n_perts * per_pert_time)
    # where per_pert_time is the parallel fitting time / n_perts
    
    # The key insight: the nb_glm_test call recomputes dispersion!
    # So we should NOT add our dispersion time again
    
    estimated_full_time = variable_cost / n_perts * n_perts_total
    # But also need to add fixed costs that don't scale
    # The variable_cost already includes fixed costs for this run
    # So a better model: (variable_cost / n_perts) * n_perts_total
    
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    
    print(f"\nStep-by-step timing:")
    for step, t in profiler.timings.items():
        print(f"  {step}: {t:.1f}s")
    print(f"  TOTAL: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    print(f"\nCost breakdown:")
    print(f"  Fixed costs (size_factors + control + dispersion): {fixed_cost:.1f}s")
    print(f"  Variable costs (per-pert fitting): {variable_cost:.1f}s")
    print(f"  Time per perturbation: {time_per_pert:.2f}s")
    
    print(f"\nProjected full runtime ({n_perts_total} perts):")
    estimated_full_time = fixed_cost + (time_per_pert * n_perts_total)
    print(f"  Estimated: {estimated_full_time:.0f}s ({estimated_full_time/3600:.1f} hours)")
    
    print(f"\nMemory:")
    print(f"  Baseline RSS: {results['baseline_rss_mb']:.0f} MB")
    print(f"  Peak RSS: {results['peak_rss_mb']:.0f} MB")
    print(f"  Available: {results['available_memory_mb']:.0f} MB")
    print(f"  System Total: {results['total_memory_mb']:.0f} MB")
    
    print(f"\n24-hour feasibility:")
    if estimated_full_time < 86400:  # 24 hours
        print(f"  ✓ Estimated {estimated_full_time/3600:.1f}h < 24h - FEASIBLE")
    else:
        print(f"  ✗ Estimated {estimated_full_time/3600:.1f}h > 24h - NOT FEASIBLE")
        print(f"    Need {estimated_full_time / 86400:.1f}x speedup to meet 24h target")
    
    print("=" * 70)
    
    return {
        "n_perts_profiled": n_perts,
        "n_perts_total": n_perts_total,
        "fixed_cost_s": fixed_cost,
        "variable_cost_s": variable_cost,
        "time_per_pert_s": time_per_pert,
        "estimated_full_time_s": estimated_full_time,
        "estimated_full_time_hours": estimated_full_time / 3600,
        "peak_rss_mb": results['peak_rss_mb'],
        "n_jobs": n_jobs,
        "memory_limit_gb": memory_limit_gb,
        "selected_perts": selected_perts,
        "step_timings": profiler.timings,
    }, profiler


def main():
    parser = argparse.ArgumentParser(
        description="Profile NB-GLM on Replogle-GW-k562 subset"
    )
    parser.add_argument(
        "--n-perts",
        type=int,
        default=5,
        help="Number of perturbations to profile (default: 5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="Memory limit in GB (default: None = auto)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save profiling results as JSON",
    )
    
    args = parser.parse_args()
    
    # Dataset path
    dataset_path = PROJECT_ROOT / "benchmarking/results/Replogle-GW-k562/.cache/standardized_Replogle-GW-k562.h5ad"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print(f"Available memory: {get_available_memory_mb()/1024:.1f} GB")
    print(f"Total memory: {get_total_memory_mb()/1024:.1f} GB")
    print(f"Current RSS: {get_rss_mb():.0f} MB")
    print()
    
    result, profiler = profile_nb_glm_subset(
        dataset_path,
        n_perts=args.n_perts,
        n_jobs=args.n_jobs,
        memory_limit_gb=args.memory_limit_gb,
    )
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
