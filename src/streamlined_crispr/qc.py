"""Quality control utilities for large ``.h5ad`` datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .data import ensure_gene_symbol_column, iter_matrix_chunks, read_backed, resolve_output_path


@dataclass
class QualityControlResult:
    """Result of quality control filtering."""

    cell_mask: np.ndarray
    gene_mask: np.ndarray
    perturbation_keep: Dict[str, bool]
    filtered_path: Path
    cell_gene_counts: np.ndarray
    gene_cell_counts: np.ndarray


def filter_cells_by_gene_count(
    path: str | Path,
    *,
    min_genes: int = 100,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return a boolean mask selecting cells with at least ``min_genes`` expressed genes."""

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        counts = np.zeros(backed.n_obs, dtype=np.int64)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            expressed = block > 0
            counts[slc] = np.asarray(expressed.sum(axis=1)).ravel()
    finally:
        backed.file.close()
    return counts >= min_genes


def filter_perturbations_by_cell_count(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str,
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
) -> np.ndarray:
    """Return a boolean mask selecting genes expressed in at least ``min_cells`` cells."""

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        counts = np.zeros(backed.n_vars, dtype=np.int64)
        if cell_mask is None:
            cell_mask = np.ones(backed.n_obs, dtype=bool)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            local_mask = cell_mask[slc]
            if not np.any(local_mask):
                continue
            expressed = block[local_mask] > 0
            counts += np.asarray(expressed.sum(axis=0)).ravel()
    finally:
        backed.file.close()
    return counts >= min_cells


def quality_control_summary(
    path: str | Path,
    *,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> QualityControlResult:
    """Run the full quality-control pipeline and persist the filtered AnnData object."""

    backed = read_backed(path)
    try:
        gene_names = ensure_gene_symbol_column(backed, gene_name_column)
    finally:
        backed.file.close()

    cell_mask = filter_cells_by_gene_count(
        path,
        min_genes=min_genes,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
    )
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
        base_mask=cell_mask,
    )
    combined_cell_mask = cell_mask & perturbation_mask
    gene_mask = filter_genes_by_cell_count(
        path,
        min_cells=min_cells_per_gene,
        cell_mask=combined_cell_mask,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
    )

    backed = read_backed(path)
    try:
        counts = np.zeros(backed.n_vars, dtype=np.int64)
        gene_counts_per_cell = np.zeros(backed.n_obs, dtype=np.int64)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            expressed = block > 0
            gene_counts_per_cell[slc] = np.asarray(expressed.sum(axis=1)).ravel()
            local_mask = combined_cell_mask[slc]
            if not np.any(local_mask):
                continue
            counts += np.asarray((block[local_mask] > 0).sum(axis=0)).ravel()
    finally:
        backed.file.close()

    backed = read_backed(path)
    try:
        filtered = backed[combined_cell_mask, gene_mask].to_memory()
    finally:
        backed.file.close()

    filtered.var = filtered.var.copy()
    filtered.var["gene_symbols"] = gene_names[gene_mask]
    filtered_path = resolve_output_path(path, suffix="filtered", output_dir=output_dir, data_name=data_name)
    filtered.write(filtered_path)

    labels = filtered.obs[perturbation_column].astype(str)
    perturbation_keep = {
        label: (label == control_label) or (labels[labels == label].shape[0] >= min_cells_per_perturbation)
        for label in labels.unique()
    }

    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered_path=filtered_path,
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=counts,
    )

