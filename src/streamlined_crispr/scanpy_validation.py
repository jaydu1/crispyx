"""Utilities for comparing the streaming pipeline with an in-memory Scanpy workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .data import (
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
)
from .de import _tie_correction, wald_test, wilcoxon_test
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
    wald_effect_max_abs_diff: float
    wald_statistic_max_abs_diff: float
    wald_pvalue_max_abs_diff: float
    wilcoxon_effect_max_abs_diff: float
    wilcoxon_statistic_max_abs_diff: float
    wilcoxon_pvalue_max_abs_diff: float
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


def _resolve_reference_candidates(
    labels: np.ndarray,
    control_label: str,
    perturbations: Optional[Iterable[str]],
) -> list[str]:
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
    else:
        unique = [str(p) for p in perturbations]
    candidates = [label for label in unique if label != control_label]
    if not candidates:
        raise ValueError(
            "No perturbation groups available for differential expression testing"
        )
    return candidates


def _compute_reference_wald(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    perturbations: Optional[Iterable[str]],
    min_cells_expressed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    labels = adata.obs[perturbation_column].astype(str).to_numpy()
    candidates = _resolve_reference_candidates(labels, control_label, perturbations)
    gene_symbols = ensure_gene_symbol_column(adata, gene_name_column)

    log_matrix = np.asarray(adata.X, dtype=np.float64)
    norm_matrix = np.asarray(adata.layers["normalized_counts"], dtype=np.float64)

    control_mask = labels == control_label
    control_n = int(control_mask.sum())
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_log = log_matrix[control_mask]
    control_mean = control_log.mean(axis=0)
    control_var = np.zeros_like(control_mean)
    if control_n > 1:
        control_var = control_log.var(axis=0, ddof=1)
    control_var = np.clip(control_var, a_min=0, a_max=None)

    control_expr = np.count_nonzero(norm_matrix[control_mask], axis=0)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for label in candidates:
        mask = labels == label
        n_cells = int(mask.sum())
        if n_cells == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        group_log = log_matrix[mask]
        group_mean = group_log.mean(axis=0)
        group_var = np.zeros_like(group_mean)
        if n_cells > 1:
            group_var = group_log.var(axis=0, ddof=1)
        group_var = np.clip(group_var, a_min=0, a_max=None)
        effect = group_mean - control_mean
        se = np.sqrt(control_var / control_n + group_var / n_cells)
        expr_group = np.count_nonzero(norm_matrix[mask], axis=0)
        total_expr = control_expr + expr_group
        valid = (se > 0) & (total_expr >= min_cells_expressed)
        z = np.zeros_like(effect)
        pvalue = np.ones_like(effect)
        if np.any(valid):
            z[valid] = effect[valid] / se[valid]
            pvalue[valid] = 2 * norm.sf(np.abs(z[valid]))
        results[label] = {
            "genes": gene_symbols,
            "effect": effect,
            "statistic": z,
            "pvalue": pvalue,
        }
    return results


def _compute_reference_wilcoxon(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    perturbations: Optional[Iterable[str]],
    min_cells_expressed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    labels = adata.obs[perturbation_column].astype(str).to_numpy()
    candidates = _resolve_reference_candidates(labels, control_label, perturbations)
    gene_symbols = ensure_gene_symbol_column(adata, gene_name_column)

    log_matrix = np.asarray(adata.X, dtype=np.float64)
    norm_matrix = np.asarray(adata.layers["normalized_counts"], dtype=np.float64)

    control_mask = labels == control_label
    control_n = int(control_mask.sum())
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_log = log_matrix[control_mask]
    control_norm = norm_matrix[control_mask]
    control_expr = np.count_nonzero(control_norm, axis=0)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for label in candidates:
        mask = labels == label
        n_cells = int(mask.sum())
        if n_cells == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")

        pert_log = log_matrix[mask]
        pert_norm = norm_matrix[mask]
        pert_expr = np.count_nonzero(pert_norm, axis=0)
        total_expr = control_expr + pert_expr

        combined = np.vstack((pert_log, control_log))
        ranks = rankdata(combined, axis=0)
        rank_sum = ranks[:n_cells].sum(axis=0)
        tie = _tie_correction(ranks)

        expected = n_cells * (n_cells + control_n + 1.0) / 2.0
        std = np.sqrt(tie * n_cells * control_n * (n_cells + control_n + 1.0) / 12.0)
        u_stat = rank_sum - n_cells * (n_cells + 1.0) / 2.0

        valid = (std > 0) & (total_expr >= min_cells_expressed)

        effect = np.zeros(adata.n_vars, dtype=np.float64)
        z = np.zeros_like(effect)
        pvalue = np.ones_like(effect)

        with np.errstate(divide="ignore", invalid="ignore"):
            z[valid] = (rank_sum[valid] - expected) / std[valid]
        pvalue[valid] = 2.0 * norm.sf(np.abs(z[valid]))
        effect[valid] = u_stat[valid] / (n_cells * control_n) - 0.5

        results[label] = {
            "genes": gene_symbols,
            "effect": effect,
            "statistic": z,
            "pvalue": pvalue,
        }

    return results


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
    de_min_cells_expressed = 0

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

    t0 = time.perf_counter()
    wald_results = wald_test(
        qc_result.filtered_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        min_cells_expressed=de_min_cells_expressed,
        chunk_size=chunk_size,
        output_dir=output_dir,
        data_name=data_name,
    )
    timings_streamlined["wald_test"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    wilcoxon_results = wilcoxon_test(
        qc_result.filtered_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        min_cells_expressed=de_min_cells_expressed,
        chunk_size=chunk_size,
        output_dir=output_dir,
        data_name=data_name,
    )
    timings_streamlined["wilcoxon_test"] = time.perf_counter() - t0

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

    t0 = time.perf_counter()
    reference_wald = _compute_reference_wald(
        reference_filtered,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        min_cells_expressed=de_min_cells_expressed,
    )
    timings_reference["wald_test"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reference_wilcoxon = _compute_reference_wilcoxon(
        reference_filtered,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        min_cells_expressed=de_min_cells_expressed,
    )
    timings_reference["wilcoxon_test"] = time.perf_counter() - t0

    aligned_avg, aligned_ref_avg = _align_frames(avg_log_effects, reference_avg)
    aligned_pseudo, aligned_ref_pseudo = _align_frames(
        pseudobulk_effects, reference_pseudo
    )

    avg_diff = float(np.max(np.abs(aligned_avg.to_numpy() - aligned_ref_avg.to_numpy())))
    pseudo_diff = float(
        np.max(np.abs(aligned_pseudo.to_numpy() - aligned_ref_pseudo.to_numpy()))
    )

    def _max_abs_diff_from_results(
        stream_dict: Mapping[str, object],
        reference_dict: Dict[str, Dict[str, np.ndarray]],
        stream_attr: str,
        reference_key: str,
    ) -> float:
        max_diff = 0.0
        for label, stream_result in stream_dict.items():
            if label not in reference_dict:
                continue
            ref_result = reference_dict[label]
            stream_series = pd.Series(
                getattr(stream_result, stream_attr), index=stream_result.genes
            )
            ref_series = pd.Series(ref_result[reference_key], index=ref_result["genes"])
            aligned_stream, aligned_ref = stream_series.align(ref_series)
            if aligned_stream.empty:
                continue
            diff = float(
                np.max(np.abs(aligned_stream.to_numpy() - aligned_ref.to_numpy()))
            )
            max_diff = max(max_diff, diff)
        return max_diff

    wald_effect_diff = _max_abs_diff_from_results(
        wald_results, reference_wald, "effect_size", "effect"
    )
    wald_stat_diff = _max_abs_diff_from_results(
        wald_results, reference_wald, "statistic", "statistic"
    )
    wald_pvalue_diff = _max_abs_diff_from_results(
        wald_results, reference_wald, "pvalue", "pvalue"
    )
    wilcoxon_effect_diff = _max_abs_diff_from_results(
        wilcoxon_results, reference_wilcoxon, "effect_size", "effect"
    )
    wilcoxon_stat_diff = _max_abs_diff_from_results(
        wilcoxon_results, reference_wilcoxon, "statistic", "statistic"
    )
    wilcoxon_pvalue_diff = _max_abs_diff_from_results(
        wilcoxon_results, reference_wilcoxon, "pvalue", "pvalue"
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
        wald_effect_max_abs_diff=wald_effect_diff,
        wald_statistic_max_abs_diff=wald_stat_diff,
        wald_pvalue_max_abs_diff=wald_pvalue_diff,
        wilcoxon_effect_max_abs_diff=wilcoxon_effect_diff,
        wilcoxon_statistic_max_abs_diff=wilcoxon_stat_diff,
        wilcoxon_pvalue_max_abs_diff=wilcoxon_pvalue_diff,
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
