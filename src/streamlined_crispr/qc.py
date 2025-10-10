"""Quality control utilities implemented for the lightweight ``.h5ad`` format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import read_h5ad


@dataclass
class QCFilters:
    cell_mask: List[bool]
    perturbation_mask: List[bool]
    gene_mask: List[bool]


def filter_cells_by_gene_count(path: str, *, min_genes: int = 100) -> List[bool]:
    adata = read_h5ad(path)
    mask = []
    for row in adata.iter_rows():
        expressed = sum(1 for value in row if value > 0)
        mask.append(expressed >= min_genes)
    return mask


def filter_genes_by_cell_count(path: str, *, min_cells: int = 100) -> List[bool]:
    adata = read_h5ad(path)
    counts = [0 for _ in range(adata.n_vars)]
    for row in adata.iter_rows():
        for idx, value in enumerate(row):
            if value > 0:
                counts[idx] += 1
    return [count >= min_cells for count in counts]


def filter_perturbations_by_cell_count(
    path: str,
    *,
    column: str = "perturbation",
    control_label: str = "ctrl",
    min_cells: int = 50,
) -> List[bool]:
    adata = read_h5ad(path)
    perturbations = [cell[column] for cell in adata.obs]
    counts = {}
    for label in perturbations:
        counts[label] = counts.get(label, 0) + 1
    mask = []
    for label in perturbations:
        if label == control_label:
            mask.append(True)
        else:
            mask.append(counts.get(label, 0) >= min_cells)
    return mask


def quality_control_summary(
    path: str,
    *,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    perturbation_column: str = "perturbation",
    control_label: str = "ctrl",
) -> QCFilters:
    cell_mask = filter_cells_by_gene_count(path, min_genes=min_genes)
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
    )
    gene_mask = filter_genes_by_cell_count(path, min_cells=min_cells_per_gene)
    combined = [cell and perturbation for cell, perturbation in zip(cell_mask, perturbation_mask)]
    return QCFilters(cell_mask=combined, perturbation_mask=perturbation_mask, gene_mask=gene_mask)
