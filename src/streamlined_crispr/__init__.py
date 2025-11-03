"""Streamlined CRISPR screen analysis toolkit."""

from .data import (
    ensure_gene_symbol_column,
    preview_backed,
    read_backed,
    resolve_control_label,
    resolve_output_path,
)
from .qc import (
    filter_cells_by_gene_count,
    filter_genes_by_cell_count,
    filter_perturbations_by_cell_count,
    quality_control_summary,
)
from .pseudobulk import compute_average_log_expression, compute_pseudobulk_expression
from .de import RankGenesGroupsResult, nb_glm_test, wald_test, wilcoxon_test

__all__ = [
    "filter_cells_by_gene_count",
    "filter_genes_by_cell_count",
    "filter_perturbations_by_cell_count",
    "quality_control_summary",
    "compute_average_log_expression",
    "compute_pseudobulk_expression",
    "RankGenesGroupsResult",
    "wald_test",
    "wilcoxon_test",
    "nb_glm_test",
    "ensure_gene_symbol_column",
    "preview_backed",
    "read_backed",
    "resolve_control_label",
    "resolve_output_path",
]
