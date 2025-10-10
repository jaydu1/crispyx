"""Streamlined CRISPR screen analysis toolkit."""

from .data import read_h5ad, write_h5ad
from .qc import (
    filter_cells_by_gene_count,
    filter_genes_by_cell_count,
    filter_perturbations_by_cell_count,
    quality_control_summary,
)
from .pseudobulk import compute_average_log_expression, compute_pseudobulk_expression
from .de import wald_test, wilcoxon_test

__all__ = [
    "filter_cells_by_gene_count",
    "filter_genes_by_cell_count",
    "filter_perturbations_by_cell_count",
    "quality_control_summary",
    "compute_average_log_expression",
    "compute_pseudobulk_expression",
    "wald_test",
    "wilcoxon_test",
    "read_h5ad",
    "write_h5ad",
]
