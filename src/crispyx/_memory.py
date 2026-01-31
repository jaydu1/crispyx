"""Memory management utilities for adaptive batch processing.

This module provides functions for estimating memory usage and
determining optimal batch sizes for parallel processing.
"""

from __future__ import annotations

import os


def _get_available_memory_mb() -> float:
    """Get available system memory in MB, with fallback."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e6
    except ImportError:
        return 8000.0  # 8 GB default fallback


def _estimate_dense_memory_gb(n_cells: int, n_genes: int, n_copies: int = 3) -> float:
    """Estimate memory required to densify a matrix with work arrays.
    
    Parameters
    ----------
    n_cells
        Number of cells (rows).
    n_genes
        Number of genes (columns).
    n_copies
        Number of dense matrix copies needed (default 3: Y, mu, work arrays).
        
    Returns
    -------
    float
        Estimated memory in GB.
    """
    bytes_per_element = 8  # float64
    return n_cells * n_genes * bytes_per_element * n_copies / 1e9


def _estimate_gene_batch_size_fitter(
    n_samples: int,
    n_genes: int,
    n_work_arrays: int = 4,
    target_mb: float = 100.0,
) -> int:
    """Estimate optimal gene batch size based on memory constraints.
    
    Calculates batch size to keep work array memory usage under target_mb.
    Work arrays are typically (n_samples, batch_size) shaped.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    n_work_arrays
        Number of work arrays allocated per batch (default 4 after optimization).
    target_mb
        Target memory usage in MB for work arrays (default 100 MB).
        
    Returns
    -------
    int
        Recommended gene batch size, clamped between 256 and n_genes.
    """
    bytes_per_gene = n_samples * 8 * n_work_arrays  # float64 = 8 bytes
    target_bytes = target_mb * 1e6
    batch_size = int(target_bytes / bytes_per_gene)
    # Clamp between 256 (minimum for efficiency) and n_genes (maximum)
    return max(256, min(batch_size, n_genes))


def _estimate_max_workers(
    n_samples: int,
    n_genes: int,
    memory_per_worker_mb: float | None = None,
    available_mb: float | None = None,
    memory_limit_mb: float | None = None,
) -> int:
    """Estimate maximum number of parallel workers based on memory constraints.
    
    Limits worker count to prevent OOM from multiple workers each allocating
    large work arrays.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    memory_per_worker_mb
        Estimated memory per worker in MB. If None, calculated from data size.
    available_mb
        Available memory in MB. If None, uses 80% of system memory.
    memory_limit_mb
        Optional explicit memory limit in MB (e.g., from config). If provided,
        the effective memory budget is min(available_mb, memory_limit_mb).
        
    Returns
    -------
    int
        Recommended maximum number of workers.
    """
    if available_mb is None:
        # Try to get system memory, default to 8 GB if unavailable
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / 1e6 * 0.8
        except ImportError:
            available_mb = 8000.0  # 8 GB default
    
    # Apply explicit memory limit if provided
    if memory_limit_mb is not None:
        effective_mb = min(available_mb, memory_limit_mb * 0.8)  # 80% of limit
    else:
        effective_mb = available_mb
    
    if memory_per_worker_mb is None:
        # Estimate: 4 work arrays + Y subset + overhead
        n_work_arrays = 5
        memory_per_worker_mb = n_samples * n_genes * 8 * n_work_arrays / 1e6
    
    max_workers = max(1, int(effective_mb / memory_per_worker_mb))
    cpu_count = os.cpu_count() or 4
    
    return min(max_workers, cpu_count)
