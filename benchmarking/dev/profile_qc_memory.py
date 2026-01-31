#!/usr/bin/env python3
"""Deep memory profiling for crispyx QC vs Scanpy.

This script profiles memory usage at each step of the QC pipeline to identify
where crispyx uses more memory than Scanpy and potential optimization opportunities.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))


def get_memory_mb() -> float:
    """Get current process memory usage in MB (RSS)."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_peak_memory_mb() -> float:
    """Get peak RSS from OS (more accurate for peak)."""
    import resource
    # Linux: ru_maxrss is in KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@contextmanager
def memory_tracking(name: str, results: dict):
    """Track memory before and after a block, and peak during."""
    gc.collect()
    mem_before = get_memory_mb()
    tracemalloc.start()
    peak_before = get_peak_memory_mb()
    
    try:
        yield
    finally:
        gc.collect()
        current, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        mem_after = get_memory_mb()
        peak_after = get_peak_memory_mb()
        
        results[name] = {
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "mem_delta_mb": mem_after - mem_before,
            "peak_delta_mb": peak_after - peak_before,
            "tracemalloc_peak_mb": peak_traced / (1024 * 1024),
            "tracemalloc_current_mb": current / (1024 * 1024),
        }
        print(f"  {name}: delta={mem_after - mem_before:.1f}MB, peak={peak_traced / (1024 * 1024):.1f}MB")


def profile_scanpy_qc_detailed(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
) -> dict[str, Any]:
    """Profile Scanpy QC with memory tracking at each step."""
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp

    gc.collect()
    result = {
        "method": "scanpy",
        "steps": {},
    }
    
    mem_start = get_memory_mb()
    peak_start = get_peak_memory_mb()
    
    # Step 1: Load data
    print("\n[Scanpy] Step 1: Load data")
    with memory_tracking("load", result["steps"]):
        adata = sc.read_h5ad(str(dataset_path))
    
    # Step 2: Ensure CSR
    print("[Scanpy] Step 2: Ensure CSR")
    with memory_tracking("ensure_csr", result["steps"]):
        if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
    
    # Step 3: Filter cells
    print("[Scanpy] Step 3: Filter cells by gene count")
    with memory_tracking("filter_cells", result["steps"]):
        if min_genes > 0:
            n_before = adata.n_obs
            sc.pp.filter_cells(adata, min_genes=min_genes)
            n_after = adata.n_obs
            print(f"    Cells: {n_before} -> {n_after}")
    
    # Step 4: Filter perturbations
    print("[Scanpy] Step 4: Filter perturbations")
    with memory_tracking("filter_perturbations", result["steps"]):
        if min_cells_per_perturbation > 0:
            n_before = adata.n_obs
            labels = adata.obs[perturbation_column].astype(str)
            counts = labels.value_counts()
            keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
            adata = adata[keep].copy()
            n_after = adata.n_obs
            print(f"    Cells: {n_before} -> {n_after}")
    
    # Step 5: Filter genes
    print("[Scanpy] Step 5: Filter genes")
    with memory_tracking("filter_genes", result["steps"]):
        if min_cells_per_gene > 0:
            n_before = adata.n_vars
            sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
            n_after = adata.n_vars
            print(f"    Genes: {n_before} -> {n_after}")
    
    # Step 6: Save (to temp file)
    print("[Scanpy] Step 6: Save output")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        output_path = Path(f.name)
    
    with memory_tracking("save", result["steps"]):
        adata.write_h5ad(output_path)
    
    # Cleanup
    output_path.unlink(missing_ok=True)
    
    # Final stats
    result["total_peak_mb"] = get_peak_memory_mb() - peak_start
    result["final_mem_mb"] = get_memory_mb() - mem_start
    result["n_cells"] = adata.n_obs
    result["n_genes"] = adata.n_vars
    
    # Clean up adata
    del adata
    gc.collect()
    
    return result


def profile_crispyx_qc_detailed(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Profile crispyx QC with memory tracking at each step."""
    from crispyx.data import (
        read_backed,
        ensure_gene_symbol_column,
        get_matrix_storage_format,
        is_dense_storage,
        resolve_control_label,
    )
    from crispyx.qc import (
        filter_cells_by_gene_count,
        filter_perturbations_by_cell_count,
        filter_genes_by_cell_count,
        _filter_genes_with_cache,
        _filter_genes_dense_optimized,
        _compute_gene_count_delta,
    )
    from crispyx.data import write_filtered_subset

    gc.collect()
    result = {
        "method": "crispyx",
        "steps": {},
    }
    
    mem_start = get_memory_mb()
    peak_start = get_peak_memory_mb()
    
    # Detect storage format
    storage_format = get_matrix_storage_format(dataset_path)
    result["storage_format"] = storage_format
    print(f"\n[crispyx] Storage format: {storage_format}")
    
    # Step 1: Load metadata only
    print("[crispyx] Step 1: Load metadata (backed)")
    with memory_tracking("load_metadata", result["steps"]):
        backed = read_backed(dataset_path)
        n_obs, n_vars = backed.n_obs, backed.n_vars
        gene_names = ensure_gene_symbol_column(backed, None)
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control = resolve_control_label(labels, control_label, verbose=False)
        backed.file.close()
    
    print(f"    Dataset: {n_obs} cells × {n_vars} genes")
    
    # Step 2: Filter cells (with return_full_result=True to get gene counts)
    print("[crispyx] Step 2: Filter cells by gene count")
    with memory_tracking("filter_cells", result["steps"]):
        cell_filter_result = filter_cells_by_gene_count(
            dataset_path,
            min_genes=min_genes,
            gene_name_column=None,
            chunk_size=chunk_size,
            return_full_result=True,
        )
        cell_mask = cell_filter_result.cell_mask
        gene_counts_per_cell = cell_filter_result.gene_counts_per_cell
        gene_cell_counts_all = cell_filter_result.gene_cell_counts_all
    
    n_cells_after = int(cell_mask.sum())
    print(f"    Cells: {n_obs} -> {n_cells_after}")
    
    # Memory analysis: arrays sizes
    cell_mask_mb = cell_mask.nbytes / (1024 * 1024)
    gene_counts_per_cell_mb = gene_counts_per_cell.nbytes / (1024 * 1024)
    gene_cell_counts_all_mb = gene_cell_counts_all.nbytes / (1024 * 1024)
    result["steps"]["filter_cells"]["array_sizes"] = {
        "cell_mask_mb": cell_mask_mb,
        "gene_counts_per_cell_mb": gene_counts_per_cell_mb,
        "gene_cell_counts_all_mb": gene_cell_counts_all_mb,
    }
    print(f"    Arrays: cell_mask={cell_mask_mb:.2f}MB, gene_counts={gene_counts_per_cell_mb:.2f}MB, gene_cell_counts={gene_cell_counts_all_mb:.2f}MB")
    
    # Step 3: Filter perturbations
    print("[crispyx] Step 3: Filter perturbations")
    with memory_tracking("filter_perturbations", result["steps"]):
        perturbation_mask = filter_perturbations_by_cell_count(
            dataset_path,
            perturbation_column=perturbation_column,
            control_label=control,
            min_cells=min_cells_per_perturbation,
            base_mask=cell_mask,
        )
        combined_cell_mask = cell_mask & perturbation_mask
    
    n_cells_after_pert = int(combined_cell_mask.sum())
    print(f"    Cells after pert filter: {n_cells_after} -> {n_cells_after_pert}")
    
    # Step 4: Compute gene counts for filtered cells (delta adjustment)
    print("[crispyx] Step 4: Adjust gene counts (delta method)")
    removed_cell_mask = cell_mask & ~perturbation_mask
    n_removed = int(removed_cell_mask.sum())
    
    with memory_tracking("adjust_gene_counts", result["steps"]):
        if n_removed == 0:
            gene_cell_counts = gene_cell_counts_all
            print("    No cells removed, using all-cell counts")
        else:
            delta_counts = _compute_gene_count_delta(
                dataset_path,
                removed_cell_mask=removed_cell_mask,
                gene_name_column=None,
                chunk_size=chunk_size,
            )
            gene_cell_counts = gene_cell_counts_all - delta_counts
            print(f"    Removed {n_removed} cells, subtracted delta counts")
    
    # Step 5: Filter genes with cache
    print("[crispyx] Step 5: Filter genes + cache CSR data")
    gene_mask = gene_cell_counts >= min_cells_per_gene
    n_genes_kept = int(gene_mask.sum())
    print(f"    Genes: {n_vars} -> {n_genes_kept}")
    
    is_dense = is_dense_storage(dataset_path)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        output_path = Path(f.name)
    
    with memory_tracking("filter_genes_cache", result["steps"]):
        if is_dense:
            gene_filter_result = _filter_genes_dense_optimized(
                dataset_path,
                min_cells=min_cells_per_gene,
                cell_mask=combined_cell_mask,
                gene_cell_counts=gene_cell_counts,
                gene_mask=gene_mask,
                gene_name_column=None,
                chunk_size=chunk_size,
            )
            chunk_cache = None
        else:
            gene_filter_result = _filter_genes_with_cache(
                dataset_path,
                min_cells=min_cells_per_gene,
                cell_mask=combined_cell_mask,
                gene_cell_counts=gene_cell_counts,
                gene_name_column=None,
                chunk_size=chunk_size,
                output_path=output_path,  # Enables caching
            )
            chunk_cache = gene_filter_result.chunk_cache
    
    gene_mask = gene_filter_result.gene_mask
    
    # Measure cache size
    if chunk_cache is not None:
        cache_size = 0
        # Handle both _ChunkCache (in-memory) and _MemmapChunkCache (disk-backed)
        if hasattr(chunk_cache, '_chunks'):
            # Old in-memory _ChunkCache
            for chunk in chunk_cache._chunks:
                if chunk is not None:
                    data, indices, indptr_diff = chunk
                    cache_size += data.nbytes + indices.nbytes + indptr_diff.nbytes
        elif hasattr(chunk_cache, '_data_mmap'):
            # New _MemmapChunkCache - data is on disk
            # Report the memmap file size instead of RAM usage
            cache_size = chunk_cache._data_mmap.nbytes + chunk_cache._indices_mmap.nbytes
            print("    Cache type: memmap (disk-backed, minimal RAM)")
        result["steps"]["filter_genes_cache"]["cache_size_mb"] = cache_size / (1024 * 1024)
        print(f"    Cache size: {cache_size / (1024 * 1024):.1f}MB")
    
    # Step 6: Write filtered subset
    print("[crispyx] Step 6: Write filtered output")
    with memory_tracking("write_output", result["steps"]):
        write_filtered_subset(
            dataset_path,
            cell_mask=combined_cell_mask,
            gene_mask=gene_mask,
            output_path=output_path,
            chunk_size=chunk_size,
            var_assignments={"gene_symbols": gene_names[gene_mask]},
            row_nnz=gene_filter_result.row_nnz,
            total_nnz=gene_filter_result.total_nnz,
            data_dtype=gene_filter_result.data_dtype,
            chunk_cache=chunk_cache,
        )
    
    # Cleanup cache
    if chunk_cache is not None:
        chunk_cache.cleanup()
    
    output_path.unlink(missing_ok=True)
    
    # Final stats
    result["total_peak_mb"] = get_peak_memory_mb() - peak_start
    result["final_mem_mb"] = get_memory_mb() - mem_start
    result["n_cells"] = n_cells_after_pert
    result["n_genes"] = n_genes_kept
    
    gc.collect()
    
    return result


def analyze_in_memory_qc_memory(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
) -> dict[str, Any]:
    """Profile the in-memory QC path used by crispyx for small datasets."""
    import anndata as ad
    import scipy.sparse as sp
    from crispyx.data import ensure_gene_symbol_column, read_backed, resolve_control_label
    
    gc.collect()
    result = {
        "method": "crispyx_in_memory",
        "steps": {},
    }
    
    mem_start = get_memory_mb()
    peak_start = get_peak_memory_mb()
    
    # Simulate the _qc_in_memory path
    print("\n[crispyx in-memory] Step 1: Load entire dataset")
    with memory_tracking("load", result["steps"]):
        adata = ad.read_h5ad(dataset_path)
    
    print("[crispyx in-memory] Step 2: Ensure CSR")
    with memory_tracking("ensure_csr", result["steps"]):
        if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
    
    print("[crispyx in-memory] Step 3: Get gene names")
    with memory_tracking("gene_names", result["steps"]):
        gene_names = ensure_gene_symbol_column(adata, None)
    
    print("[crispyx in-memory] Step 4: Compute gene counts per cell")
    with memory_tracking("gene_counts_per_cell", result["steps"]):
        if sp.issparse(adata.X):
            gene_counts_per_cell = np.asarray(adata.X.getnnz(axis=1)).ravel()
        else:
            gene_counts_per_cell = np.count_nonzero(adata.X, axis=1)
    
    print("[crispyx in-memory] Step 5: Filter cells by gene count")
    with memory_tracking("filter_cells", result["steps"]):
        cell_mask_step1 = gene_counts_per_cell >= min_genes
        adata_filtered = adata[cell_mask_step1].copy()  # This creates a copy!
    
    print(f"    Cells: {adata.n_obs} -> {adata_filtered.n_obs}")
    
    # Analyze memory: the .copy() creates a full copy of the sparse matrix
    if sp.issparse(adata.X):
        original_nnz = adata.X.nnz
        filtered_nnz = adata_filtered.X.nnz
        result["steps"]["filter_cells"]["original_X_mb"] = (
            adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes
        ) / (1024 * 1024)
        result["steps"]["filter_cells"]["filtered_X_mb"] = (
            adata_filtered.X.data.nbytes + adata_filtered.X.indices.nbytes + adata_filtered.X.indptr.nbytes
        ) / (1024 * 1024)
        print(f"    Original X: {original_nnz:,} nnz, Filtered: {filtered_nnz:,} nnz")
    
    print("[crispyx in-memory] Step 6: Filter perturbations")
    with memory_tracking("filter_perturbations", result["steps"]):
        labels = adata_filtered.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        pert_keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata_filtered = adata_filtered[pert_keep].copy()  # Another copy!
    
    print(f"    Cells after pert: {adata_filtered.n_obs}")
    
    print("[crispyx in-memory] Step 7: Compute gene cell counts")
    with memory_tracking("gene_cell_counts", result["steps"]):
        if sp.issparse(adata_filtered.X):
            gene_cell_counts = np.asarray(adata_filtered.X.getnnz(axis=0)).ravel()
        else:
            gene_cell_counts = np.count_nonzero(adata_filtered.X, axis=0)
    
    print("[crispyx in-memory] Step 8: Filter genes")
    with memory_tracking("filter_genes", result["steps"]):
        gene_mask_local = gene_cell_counts >= min_cells_per_gene
        adata_filtered = adata_filtered[:, gene_mask_local].copy()  # Another copy!
    
    print(f"    Genes: {adata.n_vars} -> {adata_filtered.n_vars}")
    
    print("[crispyx in-memory] Step 9: Add gene_symbols + save")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        output_path = Path(f.name)
    
    with memory_tracking("save", result["steps"]):
        adata_filtered.var["gene_symbols"] = gene_names[gene_mask_local].to_numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata_filtered.write(output_path)
    
    output_path.unlink(missing_ok=True)
    
    # Final stats
    result["total_peak_mb"] = get_peak_memory_mb() - peak_start
    result["final_mem_mb"] = get_memory_mb() - mem_start
    result["n_cells"] = adata_filtered.n_obs
    result["n_genes"] = adata_filtered.n_vars
    
    del adata, adata_filtered
    gc.collect()
    
    return result


def main():
    """Run detailed memory profiling."""
    datasets = [
        {
            "name": "Tian-crispra",
            "path": Path("data/Tian-crispra.h5ad"),
            "perturbation_column": "perturbation",
            "control_label": None,
            "min_genes": 100,
            "min_cells_per_perturbation": 50,
            "min_cells_per_gene": 100,
        },
        {
            "name": "Adamson",
            "path": Path("data/Adamson_subset.h5ad"),
            "perturbation_column": "perturbation",
            "control_label": None,
            "min_genes": 100,
            "min_cells_per_perturbation": 50,
            "min_cells_per_gene": 100,
        },
    ]
    
    # Find existing datasets
    available = [d for d in datasets if d["path"].exists()]
    if not available:
        print("No datasets found!")
        return
    
    print("=" * 70)
    print("QC Memory Profiling")
    print("=" * 70)
    
    results = []
    
    for dataset in available:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset['name']}")
        print(f"Path: {dataset['path']}")
        print(f"{'=' * 70}")
        
        # Get file info
        file_size_mb = dataset["path"].stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        # Resolve control label
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
        print("\n" + "-" * 40)
        print("SCANPY QC")
        print("-" * 40)
        gc.collect()
        scanpy_result = profile_scanpy_qc_detailed(dataset["path"], **config)
        scanpy_result["dataset"] = dataset["name"]
        scanpy_result["file_size_mb"] = file_size_mb
        results.append(scanpy_result)
        gc.collect()
        
        # Profile crispyx streaming
        print("\n" + "-" * 40)
        print("CRISPYX QC (Streaming)")
        print("-" * 40)
        gc.collect()
        crispyx_result = profile_crispyx_qc_detailed(dataset["path"], **config)
        crispyx_result["dataset"] = dataset["name"]
        crispyx_result["file_size_mb"] = file_size_mb
        results.append(crispyx_result)
        gc.collect()
        
        # Profile crispyx in-memory (to understand the in-memory path)
        print("\n" + "-" * 40)
        print("CRISPYX QC (In-Memory Path)")
        print("-" * 40)
        gc.collect()
        inmem_result = analyze_in_memory_qc_memory(dataset["path"], **config)
        inmem_result["dataset"] = dataset["name"]
        inmem_result["file_size_mb"] = file_size_mb
        results.append(inmem_result)
        gc.collect()
        
        # Summary comparison
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"Scanpy peak memory:            {scanpy_result['total_peak_mb']:.1f} MB")
        print(f"crispyx streaming peak memory: {crispyx_result['total_peak_mb']:.1f} MB")
        print(f"crispyx in-memory peak memory: {inmem_result['total_peak_mb']:.1f} MB")
        
        # Identify top memory consumers
        print("\n[Scanpy] Top memory steps:")
        scanpy_steps = sorted(
            scanpy_result["steps"].items(), 
            key=lambda x: x[1].get("tracemalloc_peak_mb", 0), 
            reverse=True
        )[:3]
        for name, data in scanpy_steps:
            print(f"  {name}: {data.get('tracemalloc_peak_mb', 0):.1f} MB peak")
        
        print("\n[crispyx streaming] Top memory steps:")
        crispyx_steps = sorted(
            crispyx_result["steps"].items(), 
            key=lambda x: x[1].get("tracemalloc_peak_mb", 0), 
            reverse=True
        )[:3]
        for name, data in crispyx_steps:
            print(f"  {name}: {data.get('tracemalloc_peak_mb', 0):.1f} MB peak")
            if "cache_size_mb" in data:
                print(f"      (cache: {data['cache_size_mb']:.1f} MB)")
    
    # Save results
    output_path = Path("benchmarking/dev/qc_memory_profiling_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
