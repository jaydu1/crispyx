"""Quality control utilities for large ``.h5ad`` datasets."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import (
    AnnData,
    _ensure_csr,
    calculate_optimal_chunk_size,
    ensure_gene_symbol_column,
    is_dense_storage,
    iter_matrix_chunks,
    read_backed,
    resolve_control_label,
    resolve_output_path,
    write_filtered_subset,
)

logger = logging.getLogger(__name__)


@dataclass
class _CellFilterResult:
    """Result of cell filtering with both cell and gene statistics."""
    
    cell_mask: np.ndarray
    gene_counts_per_cell: np.ndarray
    gene_cell_counts_all: np.ndarray  # cells per gene for ALL cells (before perturbation filter)


class _ChunkCache:
    """In-memory cache for CSR chunk data during QC.
    
    Stores CSR chunks in memory during the gene filtering pass, then
    streams them to the write phase without re-reading the original matrix.
    
    For very large datasets, consider using use_chunk_cache=False to avoid
    memory overhead.
    
    Parameters
    ----------
    output_path
        Base path for the output file (not used for caching, kept for API compat).
    """
    
    def __init__(self, output_path: Path | str) -> None:
        self.output_path = Path(output_path)
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (data, indices, indptr_diff)
        self._n_cols: int = 0
    
    def write_chunk(
        self,
        chunk_idx: int,
        data: np.ndarray,
        indices: np.ndarray,
        indptr_diff: np.ndarray,
        n_cols: int,
    ) -> None:
        """Store CSR chunk data in memory."""
        # Ensure list is large enough
        while len(self._chunks) <= chunk_idx:
            self._chunks.append(None)
        self._chunks[chunk_idx] = (data, indices, indptr_diff)
        self._n_cols = n_cols
    
    def iter_filtered_chunks(
        self,
        gene_indices: np.ndarray,
        data_dtype: np.dtype,
    ) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
        """Iterate through cached chunks, yielding filtered CSR data.
        
        Uses vectorized operations for efficiency.
        
        Parameters
        ----------
        gene_indices
            Indices of genes to keep.
        data_dtype
            Target dtype for data array.
            
        Yields
        ------
        tuple
            (filtered_data, filtered_indices, n_cells_in_chunk)
        """
        # Build vectorized index remapping
        gene_set = set(gene_indices.tolist())
        # Create a dense lookup array: old_idx -> new_idx (or -1 if not kept)
        remap = np.full(self._n_cols, -1, dtype=np.int32)
        remap[gene_indices] = np.arange(len(gene_indices), dtype=np.int32)
        
        for chunk in self._chunks:
            if chunk is None:
                continue
            data, indices, indptr_diff = chunk
            n_cells = len(indptr_diff)
            
            # Vectorized filtering: find which entries have kept genes
            new_col_indices = remap[indices]  # -1 for dropped genes
            keep_mask = new_col_indices >= 0
            
            # Filter data and indices
            filtered_data = data[keep_mask].astype(data_dtype, copy=False)
            filtered_indices = new_col_indices[keep_mask]
            
            yield filtered_data, filtered_indices, n_cells
    
    def cleanup(self) -> None:
        """Clear cached data from memory."""
        self._chunks.clear()
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks cached."""
        return len(self._chunks)


@dataclass
class QualityControlResult:
    """Result of quality control filtering."""

    cell_mask: np.ndarray
    gene_mask: np.ndarray
    perturbation_keep: Dict[str, bool]
    filtered: AnnData
    cell_gene_counts: np.ndarray
    gene_cell_counts: np.ndarray

    @property
    def filtered_path(self) -> Path:
        """Compatibility accessor exposing the on-disk filename."""

        return self.filtered.path


def filter_cells_by_gene_count(
    path: str | Path,
    *,
    min_genes: int = 100,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
    return_full_result: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | _CellFilterResult:
    """Return a boolean mask selecting cells with at least ``min_genes`` expressed genes.
    
    This function can optionally compute both genes-per-cell (row nnz) AND
    cells-per-gene (column nnz) in a single matrix pass, avoiding a separate
    gene counting pass later.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum number of expressed genes per cell.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    return_counts
        If True, return (mask, counts) tuple instead of just mask.
        Ignored if return_full_result is True.
    return_full_result
        If True, return a _CellFilterResult containing cell_mask,
        gene_counts_per_cell, and gene_cell_counts_all (cells per gene
        for all cells, before any perturbation filtering).
        
    Returns
    -------
    mask or (mask, counts) or _CellFilterResult
        Boolean mask, optionally with counts, or full result dataclass.
    """

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        
        gene_counts_per_cell = np.zeros(n_obs, dtype=np.int64)
        
        # Only compute cells-per-gene if full result requested
        if return_full_result:
            gene_cell_counts_all = np.zeros(n_vars, dtype=np.int64)
        else:
            gene_cell_counts_all = None
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            if sp.issparse(block):
                gene_counts_per_cell[slc] = np.asarray(block.getnnz(axis=1)).ravel()
                if gene_cell_counts_all is not None:
                    gene_cell_counts_all += np.asarray(block.getnnz(axis=0)).ravel()
            else:
                gene_counts_per_cell[slc] = np.count_nonzero(block, axis=1)
                if gene_cell_counts_all is not None:
                    gene_cell_counts_all += np.count_nonzero(block, axis=0)
    finally:
        backed.file.close()
    
    mask = gene_counts_per_cell >= min_genes
    
    if return_full_result:
        return _CellFilterResult(
            cell_mask=mask,
            gene_counts_per_cell=gene_counts_per_cell,
            gene_cell_counts_all=gene_cell_counts_all,
        )
    
    if return_counts:
        return mask, gene_counts_per_cell
    return mask


def filter_perturbations_by_cell_count(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    min_cells: int = 50,
    base_mask: np.ndarray | None = None,
    return_counts: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.Series]:
    """Return a mask keeping cells whose perturbation has sufficient representation.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label identifying control cells. If None, auto-detected.
    min_cells
        Minimum number of cells required per perturbation.
    base_mask
        Optional mask for cells to consider (e.g., from prior filtering).
    return_counts
        If True, return (mask, cell_counts_per_perturbation) tuple.
        
    Returns
    -------
    mask or (mask, counts)
        Boolean mask, optionally with cell counts per perturbation label.
    """
    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
    finally:
        backed.file.close()

    if base_mask is None:
        base_mask = np.ones_like(labels, dtype=bool)

    # Vectorized implementation: count cells per perturbation among base_mask cells
    label_series = pd.Series(labels)
    counts = label_series[base_mask].value_counts()
    
    # Map counts back to each cell (vectorized lookup)
    count_per_cell = label_series.map(counts).fillna(0).to_numpy()
    
    # Keep cell if: (is control) OR (has enough cells AND passes base_mask)
    is_control = labels == control_label
    has_enough_cells = count_per_cell >= min_cells
    mask = (is_control | has_enough_cells) & base_mask
    
    if return_counts:
        return mask, counts
    return mask


def filter_genes_by_cell_count(
    path: str | Path,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray | None = None,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask selecting genes expressed in at least ``min_cells`` cells.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Optional mask for cells to consider.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    return_counts
        If True, return (mask, counts) tuple instead of just mask.
        
    Returns
    -------
    mask or (mask, counts)
        Boolean mask, optionally with the raw cell counts per gene.
    """

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        counts = np.zeros(backed.n_vars, dtype=np.int64)
        if cell_mask is None:
            cell_mask = np.ones(backed.n_obs, dtype=bool)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_mask = cell_mask[slc]
            if not np.any(local_mask):
                continue
            selected = block[local_mask]
            if sp.issparse(selected):
                counts += np.asarray(selected.getnnz(axis=0)).ravel()
            else:
                counts += np.count_nonzero(selected, axis=0)
    finally:
        backed.file.close()
    
    mask = counts >= min_cells
    if return_counts:
        return mask, counts
    return mask


@dataclass
class _GeneFilterResult:
    """Result of fused gene filtering and nnz counting."""
    
    gene_mask: np.ndarray
    gene_cell_counts: np.ndarray
    row_nnz: np.ndarray
    total_nnz: int
    data_dtype: np.dtype
    chunk_cache: _ChunkCache | None = None  # Optional cache for write phase


def _compute_gene_count_delta(
    path: str | Path,
    *,
    removed_cell_mask: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Compute gene counts for removed cells only (for delta adjustment).
    
    When perturbation filtering removes cells that passed the gene-count filter,
    we need to adjust the all-cell gene counts by subtracting counts from
    removed cells. This function iterates only the removed cells.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    removed_cell_mask
        Boolean mask where True indicates cells to count (i.e., cells that
        passed gene filter but were removed by perturbation filter).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
        
    Returns
    -------
    np.ndarray
        Gene counts for the removed cells only. Subtract from all-cell counts
        to get counts for filtered cells.
    """
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_vars = backed.n_vars
        delta_counts = np.zeros(n_vars, dtype=np.int64)
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_mask = removed_cell_mask[slc]
            if not np.any(local_mask):
                continue
            
            selected = block[local_mask]
            if sp.issparse(selected):
                delta_counts += np.asarray(selected.getnnz(axis=0)).ravel()
            else:
                delta_counts += np.count_nonzero(selected, axis=0)
    finally:
        backed.file.close()
    
    return delta_counts


def _filter_genes_with_cache(
    path: str | Path,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray,
    gene_cell_counts: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    output_path: Path | None = None,
) -> _GeneFilterResult:
    """Compute gene mask and cache CSR data in a single matrix pass.
    
    This function does a single matrix pass that:
    1. Caches CSR chunk data (data, indices, indptr_diff) to disk
    2. Uses pre-computed gene_cell_counts to determine gene_mask
    3. Computes row_nnz and total_nnz from cached data
    
    By caching CSR data during this pass, the write phase can read from
    cache instead of re-reading the original matrix, reducing total passes
    from 4 to 2.
    
    Cache files use uncompressed np.savez (~2GB for 300K cells).
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Boolean mask for cells to include (from prior cell filtering).
    gene_cell_counts
        Pre-computed cells per gene for filtered cells.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    output_path
        Path for output file. Cache directory will be created relative
        to this path. If None, no caching is done.
        
    Returns
    -------
    _GeneFilterResult
        Dataclass containing gene_mask, gene_cell_counts, row_nnz, total_nnz,
        data_dtype, and optionally chunk_cache for the write phase.
    """
    # Compute gene mask from pre-computed counts
    gene_mask = gene_cell_counts >= min_cells
    gene_indices = np.flatnonzero(gene_mask)
    
    # Create cache if output_path provided
    chunk_cache: _ChunkCache | None = None
    if output_path is not None:
        chunk_cache = _ChunkCache(output_path)
    
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_vars = backed.n_vars
        n_filtered_cells = int(cell_mask.sum())
        row_nnz = np.zeros(n_filtered_cells, dtype=np.int64)
        total_nnz = 0
        data_dtype: np.dtype | None = None
        row_offset = 0
        chunk_idx = 0
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_cell_mask = cell_mask[slc]
            if not np.any(local_cell_mask):
                continue
            
            selected = block[local_cell_mask]
            csr = _ensure_csr(selected)
            
            # Cache the full CSR data for this chunk (before gene filtering)
            if chunk_cache is not None:
                chunk_cache.write_chunk(
                    chunk_idx,
                    data=csr.data.copy(),
                    indices=csr.indices.copy(),
                    indptr_diff=np.diff(csr.indptr),
                    n_cols=n_vars,
                )
            
            # Apply gene mask and compute nnz
            if gene_indices.size:
                filtered = csr[:, gene_indices]
            else:
                filtered = csr[:, []]
            
            filtered_csr = _ensure_csr(filtered)
            counts = np.diff(filtered_csr.indptr)
            size = counts.size
            row_nnz[row_offset : row_offset + size] = counts
            total_nnz += int(filtered_csr.nnz)
            
            if data_dtype is None and csr.nnz:
                data_dtype = csr.data.dtype
            
            row_offset += size
            chunk_idx += 1
    finally:
        backed.file.close()
    
    if data_dtype is None:
        data_dtype = np.float32
    
    return _GeneFilterResult(
        gene_mask=gene_mask,
        gene_cell_counts=gene_cell_counts,
        row_nnz=row_nnz,
        total_nnz=total_nnz,
        data_dtype=data_dtype,
        chunk_cache=chunk_cache,
    )


def _filter_genes_dense_optimized(
    path: str | Path,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray,
    gene_cell_counts: np.ndarray,
    gene_mask: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
) -> _GeneFilterResult:
    """Optimized gene filtering for dense-stored datasets.
    
    This function avoids expensive CSR conversion by directly computing
    row nnz from dense blocks using vectorized numpy operations. For
    datasets stored as dense arrays (encoding-type='array'), this is
    significantly faster than the cache-based approach.
    
    Unlike _filter_genes_with_cache, this does NOT cache data because:
    1. Dense→CSR conversion is the bottleneck we're avoiding
    2. Re-reading dense data in write phase is faster than conversion
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Boolean mask for cells to include (from prior cell filtering).
    gene_cell_counts
        Pre-computed cells per gene for filtered cells.
    gene_mask
        Pre-computed gene mask (genes with >= min_cells).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
        
    Returns
    -------
    _GeneFilterResult
        Dataclass containing gene_mask, gene_cell_counts, row_nnz, total_nnz,
        data_dtype. chunk_cache is always None for dense path.
    """
    gene_indices = np.flatnonzero(gene_mask)
    
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_filtered_cells = int(cell_mask.sum())
        row_nnz = np.zeros(n_filtered_cells, dtype=np.int64)
        total_nnz = 0
        data_dtype: np.dtype | None = None
        row_offset = 0
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_cell_mask = cell_mask[slc]
            if not np.any(local_cell_mask):
                continue
            
            selected = block[local_cell_mask]
            
            # Apply gene filter
            if gene_indices.size:
                filtered = selected[:, gene_indices]
            else:
                filtered = selected[:, []]
            
            # Vectorized nnz counting - works for both dense and sparse
            if sp.issparse(filtered):
                counts = np.asarray(filtered.getnnz(axis=1)).ravel()
                chunk_nnz = int(filtered.nnz)
                if data_dtype is None and filtered.nnz:
                    data_dtype = filtered.data.dtype
            else:
                # Dense: count non-zeros per row without CSR conversion
                counts = np.count_nonzero(filtered, axis=1)
                chunk_nnz = int(counts.sum())
                if data_dtype is None:
                    data_dtype = filtered.dtype
            
            size = counts.size
            row_nnz[row_offset : row_offset + size] = counts
            total_nnz += chunk_nnz
            row_offset += size
    finally:
        backed.file.close()
    
    if data_dtype is None:
        data_dtype = np.float32
    
    return _GeneFilterResult(
        gene_mask=gene_mask,
        gene_cell_counts=gene_cell_counts,
        row_nnz=row_nnz,
        total_nnz=total_nnz,
        data_dtype=data_dtype,
        chunk_cache=None,  # No caching for dense path
    )


def quality_control_summary(
    path: str | Path,
    *,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    use_chunk_cache: bool = True,
    delta_threshold: float = 0.3,
) -> QualityControlResult:
    """Run the full quality-control pipeline and persist the filtered AnnData object.
    
    This optimized version uses:
    1. Fused cell+gene counting in Pass 1 (computes both genes-per-cell and
       cells-per-gene in a single matrix scan)
    2. Delta adjustment for gene counts after perturbation filtering (only
       re-scans removed cells if they represent < delta_threshold of total)
    3. Disk caching of CSR chunks to eliminate the write-phase matrix re-read
    
    Cache files use uncompressed np.savez (~2GB disk for 300K cells).
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum number of expressed genes per cell.
    min_cells_per_perturbation
        Minimum number of cells required per perturbation.
    min_cells_per_gene
        Minimum number of cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label identifying control cells. If None, auto-detected.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk. If None, automatically
        calculated based on available memory.
    memory_limit_gb
        Optional memory limit in GB for chunk size calculation.
        Only used when chunk_size is None.
    output_dir
        Directory for output files.
    data_name
        Base name for output files.
    use_chunk_cache
        If True, cache CSR chunk data to disk during gene filtering to
        avoid re-reading the matrix during the write phase. Disable for
        low-disk environments. Default True.
    delta_threshold
        Threshold for delta adjustment. If removed cells represent less
        than this fraction of filtered cells, use delta adjustment instead
        of full recompute. Default 0.3 (30%).
        
    Returns
    -------
    QualityControlResult
        Dataclass containing masks, filtered AnnData, and QC statistics.
    """
    backed = read_backed(path)
    try:
        gene_names = ensure_gene_symbol_column(backed, gene_name_column)
        n_obs, n_vars = backed.n_obs, backed.n_vars
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
    finally:
        backed.file.close()

    # Determine chunk size: explicit user choice overrides auto-detection
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(
            n_obs, n_vars, available_memory_gb=memory_limit_gb
        )

    # Resolve output path early for cache directory
    filtered_path = resolve_output_path(path, suffix="filtered", output_dir=output_dir, data_name=data_name)

    # Pass 1: Filter cells by gene count AND compute cells-per-gene for all cells
    # This fuses two separate passes into one
    cell_filter_result = filter_cells_by_gene_count(
        path,
        min_genes=min_genes,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        return_full_result=True,
    )
    cell_mask = cell_filter_result.cell_mask
    gene_counts_per_cell = cell_filter_result.gene_counts_per_cell
    gene_cell_counts_all = cell_filter_result.gene_cell_counts_all
    
    # No matrix pass: Filter perturbations by cell count (metadata only)
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
        base_mask=cell_mask,
    )
    combined_cell_mask = cell_mask & perturbation_mask
    
    # Compute gene counts for filtered cells using delta adjustment
    # Cells removed by perturbation filter: passed gene filter but failed perturbation filter
    removed_cell_mask = cell_mask & ~perturbation_mask
    n_removed = int(removed_cell_mask.sum())
    n_filtered = int(combined_cell_mask.sum())
    
    if n_removed == 0:
        # No cells removed by perturbation filter, use all-cell counts directly
        gene_cell_counts = gene_cell_counts_all
        logger.debug("No cells removed by perturbation filter, using all-cell gene counts")
    elif n_filtered > 0 and n_removed / n_filtered < delta_threshold:
        # Few cells removed, use delta adjustment (faster)
        logger.debug(
            "Using delta adjustment: %d removed cells (%.1f%% of %d filtered)",
            n_removed, 100 * n_removed / n_filtered, n_filtered
        )
        delta_counts = _compute_gene_count_delta(
            path,
            removed_cell_mask=removed_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )
        gene_cell_counts = gene_cell_counts_all - delta_counts
    else:
        # Many cells removed, full recompute is faster
        logger.debug(
            "Using full recompute: %d removed cells (%.1f%% of %d filtered)",
            n_removed, 100 * n_removed / n_filtered if n_filtered > 0 else 0, n_filtered
        )
        _, gene_cell_counts = filter_genes_by_cell_count(
            path,
            min_cells=0,  # We just want the counts
            cell_mask=combined_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            return_counts=True,
        )
    
    # Pass 2: Gene filtering with nnz counting
    # Choose optimized path based on data storage format
    is_dense = is_dense_storage(path)
    gene_mask = gene_cell_counts >= min_cells_per_gene
    
    if is_dense:
        # Dense storage: use optimized path that avoids CSR conversion
        # This is ~5x faster for dense data (skips expensive dense→CSR conversion)
        logger.debug("Using dense-optimized path (source stored as dense array)")
        gene_filter_result = _filter_genes_dense_optimized(
            path,
            min_cells=min_cells_per_gene,
            cell_mask=combined_cell_mask,
            gene_cell_counts=gene_cell_counts,
            gene_mask=gene_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )
        chunk_cache = None  # No caching for dense path
    else:
        # Sparse storage: use CSR caching for write phase optimization
        logger.debug("Using CSR cache path (source stored as sparse)")
        gene_filter_result = _filter_genes_with_cache(
            path,
            min_cells=min_cells_per_gene,
            cell_mask=combined_cell_mask,
            gene_cell_counts=gene_cell_counts,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            output_path=filtered_path if use_chunk_cache else None,
        )
        chunk_cache = gene_filter_result.chunk_cache
    
    gene_mask = gene_filter_result.gene_mask
    
    try:
        # Pass 3: Write filtered subset
        # Uses cache if available, otherwise reads from source
        write_filtered_subset(
            path,
            cell_mask=combined_cell_mask,
            gene_mask=gene_mask,
            output_path=filtered_path,
            chunk_size=chunk_size,
            var_assignments={"gene_symbols": gene_names[gene_mask]},
            row_nnz=gene_filter_result.row_nnz,
            total_nnz=gene_filter_result.total_nnz,
            data_dtype=gene_filter_result.data_dtype,
            chunk_cache=chunk_cache,
        )
    finally:
        # Always cleanup cache
        if chunk_cache is not None:
            chunk_cache.cleanup()

    filtered = AnnData(filtered_path)
    labels = filtered.obs[perturbation_column].astype(str)
    perturbation_keep = {
        label: (label == control_label) or (labels[labels == label].shape[0] >= min_cells_per_perturbation)
        for label in labels.unique()
    }

    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=filtered,
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts,
    )

