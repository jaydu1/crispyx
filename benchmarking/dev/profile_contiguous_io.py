#!/usr/bin/env python
"""Profile contiguous cell slicing vs boolean mask access for NB-GLM optimization.

This script measures:
1. Cost of sorting cells by perturbation label
2. I/O speedup from contiguous slicing vs boolean mask
3. End-to-end NB-GLM timing comparison

Usage:
    python -m benchmarking.tools.profile_contiguous_io \
        --dataset Adamson_subset \
        --n-perturbations 2 \
        --output-dir benchmarking/results/io_profiling

    # Test multiple perturbation counts
    python -m benchmarking.tools.profile_contiguous_io \
        --dataset Adamson \
        --n-perturbations 2 4 10 \
        --output-dir benchmarking/results/io_profiling
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import scipy.sparse as sp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from crispyx.data import read_backed, resolve_control_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data structures for profiling results
# =============================================================================

@dataclass
class SortingProfile:
    """Profiling results for cell sorting."""
    n_cells: int
    n_perturbations: int
    sort_indices_time_s: float
    compute_ranges_time_s: float
    total_time_s: float
    memory_overhead_mb: float = 0.0


@dataclass 
class IOProfile:
    """Profiling results for I/O comparison."""
    perturbation: str
    n_cells_group: int
    boolean_mask_time_s: float
    contiguous_slice_time_s: float
    speedup: float
    

@dataclass
class EndToEndProfile:
    """End-to-end timing for NB-GLM with different access patterns."""
    n_perturbations: int
    baseline_total_s: float
    baseline_per_pert_s: float
    contiguous_total_s: float
    contiguous_per_pert_s: float
    sorting_overhead_s: float
    net_speedup: float
    breakeven_perturbations: int


@dataclass
class ProfilingResult:
    """Complete profiling results for a dataset/config."""
    dataset: str
    n_cells: int
    n_genes: int
    n_perturbations_total: int
    n_perturbations_tested: int
    control_label: str
    perturbations_tested: list[str]
    
    sorting: SortingProfile
    io_profiles: list[IOProfile]
    
    # Aggregate I/O stats
    mean_boolean_time_s: float = 0.0
    mean_contiguous_time_s: float = 0.0
    mean_speedup: float = 0.0
    
    # End-to-end (optional, if run)
    end_to_end: EndToEndProfile | None = None
    
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Compute aggregate stats
        if self.io_profiles:
            self.mean_boolean_time_s = np.mean([p.boolean_mask_time_s for p in self.io_profiles])
            self.mean_contiguous_time_s = np.mean([p.contiguous_slice_time_s for p in self.io_profiles])
            self.mean_speedup = np.mean([p.speedup for p in self.io_profiles])


# =============================================================================
# Sorting utilities
# =============================================================================

def compute_sorted_indices_and_ranges(
    labels: np.ndarray,
    control_label: str,
) -> tuple[np.ndarray, dict[str, tuple[int, int]], float, float]:
    """Sort cell indices by perturbation label and compute contiguous ranges.
    
    Parameters
    ----------
    labels
        Array of perturbation labels for each cell.
    control_label
        Label for control cells.
        
    Returns
    -------
    sort_order
        Indices that sort cells by label.
    label_ranges
        Dict mapping each label to (start, end) indices in sorted order.
    sort_time
        Time to compute sort order.
    range_time
        Time to compute label ranges.
    """
    # Time sorting
    t0 = time.perf_counter()
    sort_order = np.argsort(labels, kind="stable")
    sort_time = time.perf_counter() - t0
    
    # Time range computation
    t0 = time.perf_counter()
    sorted_labels = labels[sort_order]
    
    # Find boundaries between labels
    label_ranges = {}
    unique_labels, first_indices, counts = np.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    
    for label, start, count in zip(unique_labels, first_indices, counts):
        label_ranges[str(label)] = (int(start), int(start + count))
    
    range_time = time.perf_counter() - t0
    
    return sort_order, label_ranges, sort_time, range_time


# =============================================================================
# I/O profiling
# =============================================================================

def profile_io_access(
    path: str | Path,
    labels: np.ndarray,
    sort_order: np.ndarray,
    label_ranges: dict[str, tuple[int, int]],
    perturbations: list[str],
    n_repeats: int = 3,
) -> list[IOProfile]:
    """Profile I/O time for boolean mask vs contiguous slice access.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    labels
        Original (unsorted) cell labels.
    sort_order
        Indices that sort cells by label.
    label_ranges
        Dict mapping labels to (start, end) in sorted order.
    perturbations
        List of perturbation labels to test.
    n_repeats
        Number of times to repeat each access for stable timing.
        
    Returns
    -------
    List of IOProfile results for each perturbation.
    """
    profiles = []
    
    for label in perturbations:
        # Boolean mask access (current approach)
        boolean_times = []
        for _ in range(n_repeats):
            gc.collect()
            backed = read_backed(path)
            try:
                mask = labels == label
                t0 = time.perf_counter()
                data = backed.X[mask, :]
                if sp.issparse(data):
                    _ = data.toarray()  # Force load
                else:
                    _ = np.asarray(data)
                boolean_times.append(time.perf_counter() - t0)
            finally:
                backed.file.close()
        
        # Contiguous slice access (proposed approach)
        # Note: This simulates what we'd get with sorted data
        # We use the sort_order to reindex, which is what a sorted h5ad would have
        contiguous_times = []
        start, end = label_ranges[label]
        n_cells_group = end - start
        
        for _ in range(n_repeats):
            gc.collect()
            backed = read_backed(path)
            try:
                # Get the sorted indices for this group
                group_indices = sort_order[start:end]
                
                # For fair comparison, we measure the time to load via sorted indices
                # In a truly sorted file, this would be backed.X[start:end, :]
                # But since the file isn't sorted, we simulate by loading specific rows
                # This is actually SLOWER than contiguous access on sorted data
                # So our speedup estimate is conservative
                t0 = time.perf_counter()
                data = backed.X[group_indices, :]
                if sp.issparse(data):
                    _ = data.toarray()
                else:
                    _ = np.asarray(data)
                contiguous_times.append(time.perf_counter() - t0)
            finally:
                backed.file.close()
        
        mean_boolean = np.mean(boolean_times)
        mean_contiguous = np.mean(contiguous_times)
        
        # Note: The contiguous time here is actually for indexed access, not true contiguous
        # True contiguous on sorted data would be even faster
        # We'll also measure true contiguous access time separately
        
        profiles.append(IOProfile(
            perturbation=label,
            n_cells_group=n_cells_group,
            boolean_mask_time_s=mean_boolean,
            contiguous_slice_time_s=mean_contiguous,
            speedup=mean_boolean / max(mean_contiguous, 1e-6),
        ))
        
        logger.info(
            f"  {label}: {n_cells_group} cells, "
            f"boolean={mean_boolean:.3f}s, indexed={mean_contiguous:.3f}s, "
            f"speedup={mean_boolean/max(mean_contiguous, 1e-6):.1f}×"
        )
    
    return profiles


def profile_true_contiguous_access(
    path: str | Path,
    n_cells_per_test: list[int],
    n_repeats: int = 3,
) -> dict[int, float]:
    """Profile true contiguous slice access time.
    
    This measures what contiguous access would cost on a sorted file.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    n_cells_per_test
        List of cell counts to test.
    n_repeats
        Number of repeats for timing.
        
    Returns
    -------
    Dict mapping n_cells to mean access time.
    """
    results = {}
    
    backed = read_backed(path)
    n_total = backed.n_obs
    backed.file.close()
    
    for n_cells in n_cells_per_test:
        if n_cells > n_total:
            continue
            
        times = []
        for _ in range(n_repeats):
            gc.collect()
            backed = read_backed(path)
            try:
                # True contiguous slice
                t0 = time.perf_counter()
                data = backed.X[:n_cells, :]
                if sp.issparse(data):
                    _ = data.toarray()
                else:
                    _ = np.asarray(data)
                times.append(time.perf_counter() - t0)
            finally:
                backed.file.close()
        
        results[n_cells] = np.mean(times)
        logger.info(f"  Contiguous {n_cells} cells: {results[n_cells]:.3f}s")
    
    return results


# =============================================================================
# Main profiling function
# =============================================================================

def profile_dataset(
    path: str | Path,
    perturbation_column: str,
    control_label: str | None = None,
    n_perturbations: int | None = None,
    n_repeats: int = 3,
) -> ProfilingResult:
    """Profile a single dataset with specified number of perturbations.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    perturbation_column
        Column in obs with perturbation labels.
    control_label
        Label for control cells. If None, auto-detected.
    n_perturbations
        Number of perturbations to test. If None, test all.
    n_repeats
        Number of repeats for I/O timing.
        
    Returns
    -------
    ProfilingResult with all timing data.
    """
    path = Path(path)
    logger.info(f"Profiling {path.name}...")
    
    # Load metadata
    backed = read_backed(path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
    finally:
        backed.file.close()
    
    # Get unique perturbations (excluding control)
    unique_labels = np.unique(labels)
    all_perturbations = sorted([l for l in unique_labels if l != control_label])
    n_perturbations_total = len(all_perturbations)
    
    # Select perturbations to test (first N by sorted label)
    if n_perturbations is None or n_perturbations >= len(all_perturbations):
        perturbations_to_test = all_perturbations
    else:
        perturbations_to_test = all_perturbations[:n_perturbations]
    
    n_tested = len(perturbations_to_test)
    logger.info(f"  {n_cells} cells, {n_genes} genes, {n_perturbations_total} perturbations")
    logger.info(f"  Testing {n_tested} perturbations: {perturbations_to_test[:5]}{'...' if n_tested > 5 else ''}")
    
    # Profile sorting
    logger.info("Profiling sorting...")
    t0 = time.perf_counter()
    sort_order, label_ranges, sort_time, range_time = compute_sorted_indices_and_ranges(
        labels, control_label
    )
    total_sort_time = time.perf_counter() - t0
    
    # Estimate memory overhead (sort_order array)
    memory_overhead_mb = sort_order.nbytes / 1e6
    
    sorting_profile = SortingProfile(
        n_cells=n_cells,
        n_perturbations=n_perturbations_total,
        sort_indices_time_s=sort_time,
        compute_ranges_time_s=range_time,
        total_time_s=total_sort_time,
        memory_overhead_mb=memory_overhead_mb,
    )
    
    logger.info(f"  Sort indices: {sort_time:.3f}s, compute ranges: {range_time:.3f}s")
    logger.info(f"  Memory overhead: {memory_overhead_mb:.1f} MB")
    
    # Profile I/O access patterns
    logger.info("Profiling I/O access patterns...")
    io_profiles = profile_io_access(
        path=path,
        labels=labels,
        sort_order=sort_order,
        label_ranges=label_ranges,
        perturbations=perturbations_to_test,
        n_repeats=n_repeats,
    )
    
    # Profile true contiguous access for reference
    logger.info("Profiling true contiguous access (reference)...")
    group_sizes = [label_ranges[p][1] - label_ranges[p][0] for p in perturbations_to_test]
    unique_sizes = sorted(set(group_sizes))[:5]  # Test up to 5 unique sizes
    contiguous_ref = profile_true_contiguous_access(path, unique_sizes, n_repeats)
    
    # Build result
    result = ProfilingResult(
        dataset=path.stem,
        n_cells=n_cells,
        n_genes=n_genes,
        n_perturbations_total=n_perturbations_total,
        n_perturbations_tested=n_tested,
        control_label=control_label,
        perturbations_tested=perturbations_to_test,
        sorting=sorting_profile,
        io_profiles=io_profiles,
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {path.stem}")
    logger.info(f"  Sorting overhead: {sorting_profile.total_time_s:.2f}s")
    logger.info(f"  Mean I/O time (boolean): {result.mean_boolean_time_s:.3f}s/pert")
    logger.info(f"  Mean I/O time (indexed): {result.mean_contiguous_time_s:.3f}s/pert")
    logger.info(f"  Mean speedup: {result.mean_speedup:.1f}×")
    
    # Estimate breakeven
    if result.mean_boolean_time_s > result.mean_contiguous_time_s:
        time_saved_per_pert = result.mean_boolean_time_s - result.mean_contiguous_time_s
        breakeven = int(np.ceil(sorting_profile.total_time_s / time_saved_per_pert))
        logger.info(f"  Breakeven: {breakeven} perturbations (sorting pays off after {breakeven} perts)")
    else:
        logger.info(f"  Breakeven: N/A (no speedup observed)")
    
    logger.info("=" * 60)
    
    return result


def save_results(
    results: list[ProfilingResult],
    output_dir: Path,
) -> Path:
    """Save profiling results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"io_profiling_{timestamp}.json"
    
    # Convert to serializable format
    data = []
    for r in results:
        d = {
            "dataset": r.dataset,
            "n_cells": r.n_cells,
            "n_genes": r.n_genes,
            "n_perturbations_total": r.n_perturbations_total,
            "n_perturbations_tested": r.n_perturbations_tested,
            "control_label": r.control_label,
            "perturbations_tested": r.perturbations_tested,
            "sorting": asdict(r.sorting),
            "io_profiles": [asdict(p) for p in r.io_profiles],
            "mean_boolean_time_s": r.mean_boolean_time_s,
            "mean_contiguous_time_s": r.mean_contiguous_time_s,
            "mean_speedup": r.mean_speedup,
            "timestamp": r.timestamp,
        }
        data.append(d)
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return output_file


# =============================================================================
# CLI
# =============================================================================

def get_dataset_path(dataset_name: str) -> Path:
    """Resolve dataset name to path."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Check common locations
    candidates = [
        base_dir / "data" / f"{dataset_name}.h5ad",
        base_dir / ".cache" / f"{dataset_name}.h5ad",
        base_dir / "benchmarking" / ".cache" / f"{dataset_name}.h5ad",
        # Cached standardized datasets in benchmarking results
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"standardized_{dataset_name}.h5ad",
        # For datasets with hyphens, try both variations
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"{dataset_name}.h5ad",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    # If it's already a path
    if Path(dataset_name).exists():
        return Path(dataset_name)
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Checked:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Profile contiguous cell slicing for NB-GLM optimization"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Adamson_subset) or path to h5ad file",
    )
    parser.add_argument(
        "--n-perturbations",
        type=int,
        nargs="+",
        default=[2],
        help="Number of perturbations to test (can specify multiple, e.g., 2 4 10)",
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
        help="Control label (default: auto-detect)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of timing repeats for I/O measurement (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarking/results/io_profiling",
        help="Output directory for results (default: benchmarking/results/io_profiling)",
    )
    
    args = parser.parse_args()
    
    # Resolve dataset path
    dataset_path = get_dataset_path(args.dataset)
    logger.info(f"Using dataset: {dataset_path}")
    
    # Run profiling for each n_perturbations value
    results = []
    for n_pert in args.n_perturbations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {n_pert} perturbations")
        logger.info(f"{'='*60}\n")
        
        result = profile_dataset(
            path=dataset_path,
            perturbation_column=args.perturbation_column,
            control_label=args.control_label,
            n_perturbations=n_pert,
            n_repeats=args.n_repeats,
        )
        results.append(result)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20} {'N Perts':<10} {'Sort (s)':<10} {'Bool (s)':<12} {'Idx (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.dataset:<20} {r.n_perturbations_tested:<10} "
            f"{r.sorting.total_time_s:<10.3f} {r.mean_boolean_time_s:<12.3f} "
            f"{r.mean_contiguous_time_s:<12.3f} {r.mean_speedup:<10.1f}×"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
