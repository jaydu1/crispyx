#!/usr/bin/env python3
"""Profile crispyx QC to identify bottlenecks vs Scanpy.

This script profiles the individual substeps of crispyx's quality_control_summary
to understand where time is spent and identify optimization opportunities.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def profile_scanpy_qc(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
) -> dict[str, Any]:
    """Profile Scanpy QC with detailed timing."""
    import scanpy as sc
    import scipy.sparse as sp

    gc.collect()
    result = {
        "method": "scanpy",
        "substeps": {},
        "memory_mb": {},
    }
    
    mem_start = get_memory_mb()
    
    # Step 1: Load data
    t0 = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load = time.perf_counter() - t0
    result["substeps"]["load"] = t_load
    result["memory_mb"]["after_load"] = get_memory_mb() - mem_start
    gc.collect()
    
    # Step 2: Ensure CSR
    t0 = time.perf_counter()
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
    t_csr = time.perf_counter() - t0
    result["substeps"]["ensure_csr"] = t_csr
    gc.collect()
    
    # Step 3: Filter cells
    t0 = time.perf_counter()
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    t_filter_cells = time.perf_counter() - t0
    result["substeps"]["filter_cells"] = t_filter_cells
    result["memory_mb"]["after_filter_cells"] = get_memory_mb() - mem_start
    gc.collect()
    
    # Step 4: Filter perturbations
    t0 = time.perf_counter()
    if min_cells_per_perturbation > 0:
        labels = adata.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata = adata[keep].copy()
    t_filter_pert = time.perf_counter() - t0
    result["substeps"]["filter_perturbations"] = t_filter_pert
    result["memory_mb"]["after_filter_pert"] = get_memory_mb() - mem_start
    gc.collect()
    
    # Step 5: Filter genes
    t0 = time.perf_counter()
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    t_filter_genes = time.perf_counter() - t0
    result["substeps"]["filter_genes"] = t_filter_genes
    result["memory_mb"]["after_filter_genes"] = get_memory_mb() - mem_start
    gc.collect()
    
    # Step 6: Save
    t0 = time.perf_counter()
    output_path = dataset_path.parent / "scanpy_qc_profile_test.h5ad"
    adata.write_h5ad(output_path)
    t_save = time.perf_counter() - t0
    result["substeps"]["save"] = t_save
    result["memory_mb"]["after_save"] = get_memory_mb() - mem_start
    
    # Cleanup
    output_path.unlink(missing_ok=True)
    
    result["total_seconds"] = sum(result["substeps"].values())
    result["peak_memory_mb"] = max(result["memory_mb"].values())
    result["cells_kept"] = adata.n_obs
    result["genes_kept"] = adata.n_vars
    
    return result


def profile_crispyx_qc(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Profile crispyx QC using the high-level quality_control_summary function.
    
    This uses the optimized dispatch logic that selects the best strategy based on
    file size and storage format (in-memory for small data, column-oriented for CSC,
    row-oriented for CSR).
    """
    from crispyx.qc import quality_control_summary
    from crispyx.data import get_matrix_storage_format
    import tempfile
    import shutil

    gc.collect()
    result = {
        "method": "crispyx",
        "substeps": {},
        "memory_mb": {},
    }
    
    mem_start = get_memory_mb()
    
    # Create temp output directory
    tmp_dir = tempfile.mkdtemp(prefix="crispyx_qc_profile_")
    tmp_path = Path(tmp_dir)
    
    try:
        # Detect storage format for reporting
        storage_format = get_matrix_storage_format(dataset_path)
        result["storage_format"] = storage_format
        
        # Run the full QC pipeline with timing
        t0 = time.perf_counter()
        qc_result = quality_control_summary(
            dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=None,
            min_genes=min_genes,
            min_cells_per_perturbation=min_cells_per_perturbation,
            min_cells_per_gene=min_cells_per_gene,
            output_dir=tmp_path,
            data_name="profile_test",
            chunk_size=chunk_size,
            force_streaming=False,  # Allow in-memory for small data
            memory_limit_gb=2.0,
        )
        t_total = time.perf_counter() - t0
        
        result["substeps"]["total_qc"] = t_total
        result["memory_mb"]["after_qc"] = get_memory_mb() - mem_start
        
        # Report totals (detailed substep breakdown is inside the dispatch)
        result["total_seconds"] = t_total
        result["peak_memory_mb"] = get_memory_mb() - mem_start
        result["cells_kept"] = int(qc_result.cell_mask.sum())
        result["genes_kept"] = int(qc_result.gene_mask.sum())
        
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return result


def main():
    """Run profiling on available datasets."""
    # Define datasets to profile
    datasets = [
        {
            "name": "Adamson_subset",
            "path": Path("data/Adamson_subset.h5ad"),
            "perturbation_column": "perturbation",
            "control_label": None,  # auto-detect
            "min_genes": 100,
            "min_cells_per_perturbation": 50,
            "min_cells_per_gene": 100,
        },
        {
            "name": "Tian-crispra",
            "path": Path("data/Tian-crispra.h5ad"),
            "perturbation_column": "perturbation",
            "control_label": None,  # auto-detect
            "min_genes": 100,
            "min_cells_per_perturbation": 50,
            "min_cells_per_gene": 100,
        },
    ]
    
    # Filter to existing datasets
    available_datasets = [d for d in datasets if d["path"].exists()]
    
    if not available_datasets:
        print("No datasets found. Please ensure data files exist.")
        print("Looked for:", [str(d["path"]) for d in datasets])
        sys.exit(1)
    
    print(f"Profiling {len(available_datasets)} datasets...")
    print("=" * 60)
    
    results = []
    
    for dataset in available_datasets:
        print(f"\n{dataset['name']}")
        print("-" * 40)
        
        # Detect control label if needed
        from crispyx.data import read_backed, resolve_control_label
        backed = read_backed(dataset["path"])
        labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
        control = resolve_control_label(labels, dataset["control_label"], verbose=False)
        backed.file.close()
        
        config = {
            "min_genes": dataset["min_genes"],
            "min_cells_per_perturbation": dataset["min_cells_per_perturbation"],
            "min_cells_per_gene": dataset["min_cells_per_gene"],
            "perturbation_column": dataset["perturbation_column"],
            "control_label": control,
        }
        
        # Profile Scanpy
        print("  Profiling Scanpy...")
        gc.collect()
        scanpy_result = profile_scanpy_qc(dataset["path"], **config)
        scanpy_result["dataset"] = dataset["name"]
        results.append(scanpy_result)
        gc.collect()
        
        # Profile crispyx
        print("  Profiling crispyx...")
        gc.collect()
        crispyx_result = profile_crispyx_qc(dataset["path"], **config)
        crispyx_result["dataset"] = dataset["name"]
        results.append(crispyx_result)
        gc.collect()
        
        # Print comparison
        print(f"\n  Scanpy total:  {scanpy_result['total_seconds']:.3f}s")
        print(f"  crispyx total: {crispyx_result['total_seconds']:.3f}s")
        ratio = crispyx_result['total_seconds'] / scanpy_result['total_seconds']
        print(f"  Ratio (crispyx/scanpy): {ratio:.2f}x")
        
        # Show storage format for context
        storage_fmt = crispyx_result.get("storage_format", "unknown")
        print(f"  Storage format: {storage_fmt}")
        
        print(f"\n  Memory (peak):")
        print(f"    Scanpy:  {scanpy_result['peak_memory_mb']:.1f} MB")
        print(f"    crispyx: {crispyx_result['peak_memory_mb']:.1f} MB")
    
    # Save results
    output_path = Path("benchmarking/dev/qc_profiling_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
