"""Quality control utilities for large ``.h5ad`` datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import (
    AnnData,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    read_backed,
    resolve_control_label,
    resolve_output_path,
    write_filtered_subset,
)


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
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask selecting cells with at least ``min_genes`` expressed genes.
    
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
        
    Returns
    -------
    mask or (mask, counts)
        Boolean mask, optionally with the raw gene counts per cell.
    """

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        counts = np.zeros(backed.n_obs, dtype=np.int64)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            if sp.issparse(block):
                counts[slc] = np.asarray(block.getnnz(axis=1)).ravel()
            else:
                counts[slc] = np.count_nonzero(block, axis=1)
    finally:
        backed.file.close()
    
    mask = counts >= min_genes
    if return_counts:
        return mask, counts
    return mask


def filter_perturbations_by_cell_count(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    min_cells: int = 50,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a mask keeping cells whose perturbation has sufficient representation."""

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

    counts = pd.Series(labels[base_mask]).value_counts()
    mask = np.ones_like(labels, dtype=bool)
    for idx, label in enumerate(labels):
        if label == control_label:
            continue
        mask[idx] = counts.get(label, 0) >= min_cells and base_mask[idx]
    mask &= base_mask
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


def quality_control_summary(
    path: str | Path,
    *,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> QualityControlResult:
    """Run the full quality-control pipeline and persist the filtered AnnData object."""

    backed = read_backed(path)
    try:
        gene_names = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
    finally:
        backed.file.close()

    # Get cell mask AND the gene counts per cell (reuse counts, avoid re-iteration)
    cell_mask, gene_counts_per_cell = filter_cells_by_gene_count(
        path,
        min_genes=min_genes,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        return_counts=True,
    )
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
        base_mask=cell_mask,
    )
    combined_cell_mask = cell_mask & perturbation_mask
    
    # Get gene mask AND the cell counts per gene (reuse counts, avoid re-iteration)
    gene_mask, gene_cell_counts = filter_genes_by_cell_count(
        path,
        min_cells=min_cells_per_gene,
        cell_mask=combined_cell_mask,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        return_counts=True,
    )

    filtered_path = resolve_output_path(path, suffix="filtered", output_dir=output_dir, data_name=data_name)
    write_filtered_subset(
        path,
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        output_path=filtered_path,
        chunk_size=chunk_size,
        var_assignments={"gene_symbols": gene_names[gene_mask]},
    )

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

