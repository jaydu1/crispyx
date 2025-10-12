"""Utilities for comparing the streaming pipeline with an in-memory Scanpy workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import time

import anndata as ad
import numpy as np
import pandas as pd

from .data import iter_matrix_chunks, normalize_total_block, read_backed
from .pseudobulk import compute_average_log_expression, compute_pseudobulk_expression
from .qc import quality_control_summary


@dataclass
class ComparisonResult:
    """Summary of how the streamlined pipeline compares to a Scanpy-style workflow."""

    normalization_max_abs_diff: float
    log1p_max_abs_diff: float
    streamlined_cell_count: int
    reference_cell_count: int
    streamlined_gene_count: int
    reference_gene_count: int
    avg_log_effect_max_abs_diff: float
    pseudobulk_effect_max_abs_diff: float
    streamlined_timings: Mapping[str, float]
    reference_timings: Mapping[str, float]
    streamlined_effects: Mapping[str, pd.DataFrame]
    reference_effects: Mapping[str, pd.DataFrame]


def _normalize_total(matrix: np.ndarray, target_sum: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    library_size = matrix.sum(axis=1)
    scale = np.divide(
        target_sum,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float64),
        where=library_size > 0,
    )
    normalised = matrix * scale[:, None]
    return normalised, library_size


def _log1p(matrix: np.ndarray) -> np.ndarray:
    return np.log1p(matrix)


def _filter_cells(matrix: np.ndarray, min_genes: int) -> np.ndarray:
    expressed = (matrix > 0).sum(axis=1)
    return expressed >= min_genes


def _filter_genes(matrix: np.ndarray, min_cells: int) -> np.ndarray:
    expressed = (matrix > 0).sum(axis=0)
    return expressed >= min_cells


def _load_dense(path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(str(path))
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    adata.X = np.asarray(matrix, dtype=np.float64)
    return adata


def _compute_reference_effects(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
    baseline_count: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = adata.obs[perturbation_column].astype(str)
    log_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    log_means = log_df.groupby(labels).mean()
    control_log_mean = log_means.loc[control_label]
    avg_effects = log_means.drop(index=control_label).subtract(control_log_mean, axis=1)

    norm_df = pd.DataFrame(
        adata.layers["normalized_counts"],
        index=adata.obs_names,
        columns=adata.var_names,
    )
    norm_means = norm_df.groupby(labels).mean()
    control_bulk = np.log1p(baseline_count * norm_means.loc[control_label])
    pert_bulk = np.log1p(baseline_count * norm_means.drop(index=control_label))
    pseudo_effects = pert_bulk.subtract(control_bulk, axis=1)
    return avg_effects, pseudo_effects


def _align_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_aligned = left.sort_index().sort_index(axis=1)
    right_aligned = right.sort_index().sort_index(axis=1)
    common_rows = left_aligned.index.intersection(right_aligned.index)
    common_cols = left_aligned.columns.intersection(right_aligned.columns)
    return (
        left_aligned.loc[common_rows, common_cols],
        right_aligned.loc[common_rows, common_cols],
    )


def compare_with_scanpy(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    gene_name_column: str | None = None,
    perturbations: Optional[Iterable[str]] = None,
    baseline_count: float = 1.0,
    chunk_size: int = 2048,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> ComparisonResult:
    """Run the streamlined pipeline and compare each step to a Scanpy-style workflow."""

    path = Path(path)

    timings_streamlined: Dict[str, float] = {}
    timings_reference: Dict[str, float] = {}

    raw = _load_dense(path)
    reference_norm = raw.copy()

    t0 = time.perf_counter()
    normalised_matrix, _ = _normalize_total(reference_norm.X)
    timings_reference["normalize_total"] = time.perf_counter() - t0
    reference_norm.X = normalised_matrix

    t0 = time.perf_counter()
    log_matrix = _log1p(reference_norm.X)
    timings_reference["log1p"] = time.perf_counter() - t0
    reference_norm.X = log_matrix

    # Streamed preprocessing comparison
    backed = read_backed(path)
    max_norm_diff = 0.0
    max_log_diff = 0.0
    t0 = time.perf_counter()
    try:
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            norm_block, _ = normalize_total_block(block)
            max_norm_diff = max(
                max_norm_diff, float(np.max(np.abs(norm_block - normalised_matrix[slc])))
            )
            log_block = np.log1p(norm_block)
            max_log_diff = max(
                max_log_diff, float(np.max(np.abs(log_block - log_matrix[slc])))
            )
    finally:
        backed.file.close()
    timings_streamlined["normalize_total+log1p"] = time.perf_counter() - t0

    # Streamlined QC and effect estimation
    t0 = time.perf_counter()
    qc_result = quality_control_summary(
        path,
        min_genes=min_genes,
        min_cells_per_perturbation=min_cells_per_perturbation,
        min_cells_per_gene=min_cells_per_gene,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        output_dir=output_dir,
        data_name=data_name,
    )
    timings_streamlined["quality_control"] = time.perf_counter() - t0
    streamlined_filtered = ad.read_h5ad(str(qc_result.filtered_path))

    t0 = time.perf_counter()
    avg_log_effects = compute_average_log_expression(
        qc_result.filtered_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        chunk_size=chunk_size,
        output_dir=output_dir,
        data_name=data_name,
    )
    timings_streamlined["average_log_expression"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pseudobulk_effects = compute_pseudobulk_expression(
        qc_result.filtered_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        baseline_count=baseline_count,
        chunk_size=chunk_size,
        output_dir=output_dir,
        data_name=data_name,
    )
    timings_streamlined["pseudobulk_expression"] = time.perf_counter() - t0

    # Reference QC and effect estimation (Scanpy-style)
    reference_filtered = raw.copy()
    t0 = time.perf_counter()
    cell_mask = _filter_cells(reference_filtered.X, min_genes=min_genes)
    timings_reference["filter_cells"] = time.perf_counter() - t0
    reference_filtered = reference_filtered[cell_mask].copy()

    labels = reference_filtered.obs[perturbation_column].astype(str)
    counts = labels.value_counts()
    keep_mask = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
    reference_filtered = reference_filtered[keep_mask].copy()

    t0 = time.perf_counter()
    gene_mask = _filter_genes(reference_filtered.X, min_cells=min_cells_per_gene)
    timings_reference["filter_genes"] = time.perf_counter() - t0
    reference_filtered = reference_filtered[:, gene_mask].copy()

    t0 = time.perf_counter()
    ref_norm_matrix, _ = _normalize_total(reference_filtered.X)
    timings_reference["filtered_normalize_total"] = time.perf_counter() - t0
    reference_filtered.layers["normalized_counts"] = ref_norm_matrix
    reference_filtered.X = ref_norm_matrix

    t0 = time.perf_counter()
    reference_filtered.X = _log1p(reference_filtered.X)
    timings_reference["filtered_log1p"] = time.perf_counter() - t0

    reference_avg, reference_pseudo = _compute_reference_effects(
        reference_filtered,
        perturbation_column=perturbation_column,
        control_label=control_label,
        baseline_count=baseline_count,
    )

    aligned_avg, aligned_ref_avg = _align_frames(avg_log_effects, reference_avg)
    aligned_pseudo, aligned_ref_pseudo = _align_frames(
        pseudobulk_effects, reference_pseudo
    )

    avg_diff = float(np.max(np.abs(aligned_avg.to_numpy() - aligned_ref_avg.to_numpy())))
    pseudo_diff = float(
        np.max(np.abs(aligned_pseudo.to_numpy() - aligned_ref_pseudo.to_numpy()))
    )

    return ComparisonResult(
        normalization_max_abs_diff=max_norm_diff,
        log1p_max_abs_diff=max_log_diff,
        streamlined_cell_count=streamlined_filtered.n_obs,
        reference_cell_count=reference_filtered.n_obs,
        streamlined_gene_count=streamlined_filtered.n_vars,
        reference_gene_count=reference_filtered.n_vars,
        avg_log_effect_max_abs_diff=avg_diff,
        pseudobulk_effect_max_abs_diff=pseudo_diff,
        streamlined_timings=timings_streamlined,
        reference_timings=timings_reference,
        streamlined_effects={
            "average_log_effects": avg_log_effects,
            "pseudobulk_effects": pseudobulk_effects,
        },
        reference_effects={
            "average_log_effects": reference_avg,
            "pseudobulk_effects": reference_pseudo,
        },
    )
