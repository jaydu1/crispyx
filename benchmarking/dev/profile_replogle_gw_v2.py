#!/usr/bin/env python
"""Profile NB-GLM on Replogle-GW-k562 with benchmark-matching constraints.

This script profiles NB-GLM under the same constraints as the benchmark:
- 32 cores
- 128 GB memory limit

It measures step-level timing and memory usage to estimate total runtime
for the full 9,326+ perturbations.

Usage:
    python benchmarking/dev/profile_replogle_gw_v2.py --n-perts 10 --n-jobs 32 --memory-limit-gb 128
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
    """Step-level profiler with timing and RSS memory tracking."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.memory: Dict[str, float] = {}
        self.memory_deltas: Dict[str, float] = {}
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
        
    def get_results(self) -> Dict[str, Any]:
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


def check_sorted_file(qc_filtered_path: Path, perturbation_column: str) -> Path:
    """Check if a sorted file exists, and create one if needed."""
    import anndata as ad
    
    sorted_path = qc_filtered_path.parent / f"{qc_filtered_path.stem}_sorted.h5ad"
    
    if sorted_path.exists():
        # Verify it has sorting_metadata
        print(f"Checking existing sorted file: {sorted_path}")
        backed = ad.read_h5ad(sorted_path, backed='r')
        has_metadata = "sorting_metadata" in backed.uns
        backed.file.close()
        if has_metadata:
            print(f"  ✓ Valid sorted file found with sorting_metadata")
            return sorted_path
        else:
            print(f"  ✗ File exists but missing sorting_metadata, will re-sort")
            sorted_path.unlink()
    
    # Need to create sorted file
    print(f"Sorting file by perturbation (one-time operation)...")
    from crispyx.data import sort_by_perturbation
    
    t_start = time.perf_counter()
    result_path = sort_by_perturbation(
        qc_filtered_path,
        perturbation_column=perturbation_column,
        output_path=sorted_path,
    )
    t_end = time.perf_counter()
    print(f"  Sorting took: {t_end - t_start:.1f}s")
    
    return result_path


def profile_nb_glm(
    path: str | Path,
    n_perts: int = 10,
    n_jobs: int = 32,
    memory_limit_gb: float = 128.0,
    perturbation_column: str = "perturbation",
    control_label: str = "control",
    skip_sort_check: bool = False,
) -> tuple[Dict[str, Any], StepProfiler]:
    """Profile NB-GLM on a subset of perturbations with step-level timing."""
    import crispyx as cx
    import anndata as ad
    import scipy.sparse as sp
    
    path = Path(path)
    profiler = StepProfiler()
    
    print("=" * 70)
    print("NB-GLM PROFILING")
    print("=" * 70)
    print(f"Constraints: {n_jobs} cores, {memory_limit_gb} GB memory limit")
    print()
    
    # =========================================================================
    # Step 0: Check/create sorted file (important for large datasets)
    # =========================================================================
    if not skip_sort_check:
        profiler.start("sort_check")
        path = check_sorted_file(path, perturbation_column)
        profiler.stop("sort_check")
        print(f"Using file: {path}")
        print()
    
    # =========================================================================
    # Step 1: Load metadata and identify perturbations
    # =========================================================================
    profiler.start("metadata_load")
    
    backed = ad.read_h5ad(path, backed='r')
    n_cells_total = backed.n_obs
    n_genes = backed.n_vars
    
    # Check for sorting metadata
    has_sorting_metadata = "sorting_metadata" in backed.uns
    if has_sorting_metadata:
        sorting_meta = backed.uns["sorting_metadata"]
        n_pert_groups = len(sorting_meta.get("perturbation_boundaries", {}))
        print(f"Sorting metadata found: {n_pert_groups} perturbation groups")
    
    labels = backed.obs[perturbation_column].astype(str)
    vc = labels.value_counts()
    control_n = vc.get(control_label, 0)
    all_perts = [p for p in vc.index if p != control_label]
    n_perts_total = len(all_perts)
    backed.file.close()
    
    profiler.stop("metadata_load")
    
    print(f"Dataset: {path}")
    print(f"  Cells: {n_cells_total:,}")
    print(f"  Genes: {n_genes:,}")
    print(f"  Control cells: {control_n:,}")
    print(f"  Total perturbations: {n_perts_total:,}")
    print(f"  Memory: metadata_load took {profiler.timings['metadata_load']:.1f}s")
    print()
    
    # =========================================================================
    # Step 2: Select subset of perturbations for profiling
    # =========================================================================
    profiler.start("subset_selection")
    
    # Select a mix of perturbation sizes for realistic estimate
    selected_perts = all_perts[:n_perts]
    print(f"Selected {len(selected_perts)} perturbations for profiling:")
    for p in selected_perts[:5]:
        print(f"  - {p}: {vc[p]:,} cells")
    if len(selected_perts) > 5:
        print(f"  ... and {len(selected_perts) - 5} more")
    
    profiler.stop("subset_selection")
    print()
    
    # =========================================================================
    # Step 3: Run nb_glm_test with profiling
    # =========================================================================
    print("Running nb_glm_test with profiling enabled...")
    profiler.start("nb_glm_test")
    
    result = cx.nb_glm_test(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        perturbations=selected_perts,
        dispersion_scope='global',  # Use global dispersion
        n_jobs=n_jobs,
        memory_limit_gb=memory_limit_gb,
        max_dense_fraction=0.3,
        verbose=True,
        profiling=True,
        resume=False,
    )
    
    profiler.stop("nb_glm_test")
    
    # =========================================================================
    # Step 4: Extract internal profiling data
    # =========================================================================
    internal_profiling = {}
    if hasattr(result, 'result') and result.result is not None:
        result_adata = result.result
        if hasattr(result_adata, 'uns') and 'profiling' in result_adata.uns:
            internal_profiling = result_adata.uns['profiling']
            print("\nInternal profiling data extracted:")
            for key, value in internal_profiling.items():
                if isinstance(value, (int, float)):
                    if "seconds" in key or "time" in key:
                        print(f"  {key}: {value:.2f}s")
                    elif "mb" in key.lower():
                        print(f"  {key}: {value:.0f} MB")
                    else:
                        print(f"  {key}: {value}")
    
    # =========================================================================
    # Step 5: Calculate estimates
    # =========================================================================
    total_profiled_time = profiler.timings.get("nb_glm_test", 0)
    
    # Extract fixed vs per-perturbation costs from internal profiling
    fixed_cost_s = (
        internal_profiling.get("size_factors_seconds", 0) +
        internal_profiling.get("control_matrix_seconds", 0) +
        internal_profiling.get("control_stats_seconds", 0) +
        internal_profiling.get("global_dispersion_seconds", 0)
    )
    
    parallel_fit_seconds = internal_profiling.get("parallel_fit_seconds", 0)
    if parallel_fit_seconds > 0:
        time_per_pert = parallel_fit_seconds / n_perts
    else:
        # Fallback: estimate from total time
        time_per_pert = (total_profiled_time - fixed_cost_s) / n_perts if n_perts > 0 else 0
    
    # Estimate for full dataset
    estimated_full_time = fixed_cost_s + (time_per_pert * n_perts_total)
    
    # =========================================================================
    # Results Summary
    # =========================================================================
    results = profiler.get_results()
    
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    
    print(f"\nStep-by-step timing:")
    for step, t in profiler.timings.items():
        print(f"  {step}: {t:.1f}s ({t/60:.1f} min)")
    
    print(f"\nCost breakdown (from internal profiling):")
    print(f"  Size factors: {internal_profiling.get('size_factors_seconds', 0):.1f}s")
    print(f"  Control matrix: {internal_profiling.get('control_matrix_seconds', 0):.1f}s")
    print(f"  Control stats: {internal_profiling.get('control_stats_seconds', 0):.1f}s")
    print(f"  Global dispersion: {internal_profiling.get('global_dispersion_seconds', 0):.1f}s")
    print(f"  Parallel fitting: {internal_profiling.get('parallel_fit_seconds', 0):.1f}s")
    print(f"  Fixed costs total: {fixed_cost_s:.1f}s")
    print(f"  Time per perturbation: {time_per_pert:.3f}s")
    
    print(f"\nMemory:")
    print(f"  Baseline RSS: {results['baseline_rss_mb']:.0f} MB")
    print(f"  Peak RSS: {results['peak_rss_mb']:.0f} MB")
    print(f"  Internal peak: {internal_profiling.get('fit_peak_memory_mb', 'N/A')} MB")
    print(f"  Workers used: {internal_profiling.get('n_workers_used', 'N/A')}")
    print(f"  Memory per worker: {internal_profiling.get('estimated_per_worker_mb', 'N/A')} MB")
    
    print(f"\nProjected full runtime ({n_perts_total:,} perturbations):")
    print(f"  Fixed costs: {fixed_cost_s:.0f}s ({fixed_cost_s/60:.1f} min)")
    print(f"  Parallel fitting: {time_per_pert * n_perts_total:.0f}s ({time_per_pert * n_perts_total/3600:.1f} hours)")
    print(f"  Total estimated: {estimated_full_time:.0f}s ({estimated_full_time/3600:.1f} hours)")
    
    print(f"\n24-hour feasibility:")
    if estimated_full_time < 86400:
        speedup_available = 86400 / estimated_full_time
        print(f"  ✓ Estimated {estimated_full_time/3600:.1f}h < 24h - FEASIBLE")
        print(f"    {speedup_available:.1f}× margin available")
    else:
        speedup_needed = estimated_full_time / 86400
        print(f"  ✗ Estimated {estimated_full_time/3600:.1f}h > 24h - NOT FEASIBLE")
        print(f"    Need {speedup_needed:.1f}× speedup to meet 24h target")
    
    print("=" * 70)
    
    return {
        "n_perts_profiled": n_perts,
        "n_perts_total": n_perts_total,
        "n_cells_total": n_cells_total,
        "n_genes": n_genes,
        "control_n": control_n,
        "fixed_cost_s": fixed_cost_s,
        "time_per_pert_s": time_per_pert,
        "estimated_full_time_s": estimated_full_time,
        "estimated_full_time_hours": estimated_full_time / 3600,
        "peak_rss_mb": results['peak_rss_mb'],
        "n_jobs": n_jobs,
        "memory_limit_gb": memory_limit_gb,
        "selected_perts": selected_perts,
        "step_timings": profiler.timings,
        "internal_profiling": internal_profiling,
    }, profiler


def main():
    parser = argparse.ArgumentParser(
        description="Profile NB-GLM on Replogle-GW-k562"
    )
    parser.add_argument(
        "--n-perts",
        type=int,
        default=10,
        help="Number of perturbations to profile (default: 10)",
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
        default=128.0,
        help="Memory limit in GB (default: 128)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to h5ad file (default: use qc_filtered from results)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save profiling results as JSON",
    )
    parser.add_argument(
        "--skip-sort-check",
        action="store_true",
        help="Skip the sorted file check (use if already sorted)",
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.input_path:
        dataset_path = Path(args.input_path)
    else:
        # Default: use QC-filtered file
        dataset_path = PROJECT_ROOT / "benchmarking/results/Replogle-GW-k562/preprocessing/crispyx_qc_filtered.h5ad"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print(f"System Info:")
    print(f"  Available memory: {get_available_memory_mb()/1024:.1f} GB")
    print(f"  Total memory: {get_total_memory_mb()/1024:.1f} GB")
    print(f"  Current RSS: {get_rss_mb():.0f} MB")
    print(f"  CPU cores: {os.cpu_count()}")
    print()
    
    result, profiler = profile_nb_glm(
        dataset_path,
        n_perts=args.n_perts,
        n_jobs=args.n_jobs,
        memory_limit_gb=args.memory_limit_gb,
        skip_sort_check=args.skip_sort_check,
    )
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            result_json = {k: convert(v) for k, v in result.items()}
            json.dump(result_json, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
