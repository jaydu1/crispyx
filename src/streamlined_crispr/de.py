"""Differential expression testing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .data import (
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
    resolve_output_path,
)


@dataclass
class DifferentialExpressionResult:
    genes: pd.Index
    effect_size: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    method: str
    perturbation: str
    result_path: Path
    pvalue_adj: np.ndarray | None = None


def _resolve_candidates(
    labels: np.ndarray,
    control_label: str,
    perturbations: Iterable[str] | None,
) -> list[str]:
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
    else:
        unique = [str(p) for p in perturbations]
    candidates = [label for label in unique if label != control_label]
    if not candidates:
        raise ValueError("No perturbation groups available for differential expression testing")
    return candidates


def _tie_correction(ranks: np.ndarray) -> np.ndarray:
    """Compute tie correction factors for each column of ``ranks``."""

    n_genes = ranks.shape[1]
    correction = np.ones(n_genes, dtype=np.float64)
    for idx in range(n_genes):
        column = np.sort(np.ravel(ranks[:, idx]))
        size = float(column.size)
        if size < 2:
            continue
        boundaries = np.concatenate(
            (
                np.array([True]),
                column[1:] != column[:-1],
                np.array([True]),
            )
        )
        indices = np.flatnonzero(boundaries)
        counts = np.diff(indices).astype(np.float64)
        denom = size**3 - size
        if denom <= 0:
            continue
        correction[idx] = 1.0 - float(np.sum(counts**3 - counts)) / denom
    return correction


def _adjust_pvalue_matrix(
    matrix: np.ndarray,
    method: Literal["benjamini-hochberg", "bonferroni"],
) -> np.ndarray:
    """Return a p-value matrix adjusted per row using the requested method."""

    adjusted = np.ones_like(matrix)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        valid = np.isfinite(row)
        if not np.any(valid):
            adjusted[row_idx] = row
            continue
        values = row[valid]
        if method == "bonferroni":
            n_tests = values.size
            scaled = np.minimum(values * n_tests, 1.0)
            result = np.empty_like(values)
            result[:] = scaled
        elif method == "benjamini-hochberg":
            order = np.argsort(values)
            ordered = values[order]
            n_tests = ordered.size
            ranks = np.arange(1, n_tests + 1, dtype=np.float64)
            adj = ordered * n_tests / ranks
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)
            result = np.empty_like(ordered)
            result[order] = adj
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported correction method: {method}")
        new_row = np.array(row, copy=True)
        new_row[valid] = result
        adjusted[row_idx] = new_row
    return adjusted


def wald_test(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    chunk_size: int = 2048,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> Dict[str, DifferentialExpressionResult]:
    """Perform a Wald-style test comparing log-expression means for each perturbation."""

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
        sumsq = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
        counts = {label: 0 for label in groups}
        expr_counts = {label: np.zeros(n_genes, dtype=np.int64) for label in groups}
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            slice_labels = labels[slc]
            normalised_block, _ = normalize_total_block(block)
            raw_block = np.asarray(block)
            log_block = np.log1p(normalised_block)
            for label in groups:
                mask = slice_labels == label
                if not np.any(mask):
                    continue
                selected = log_block[mask]
                counts[label] += int(mask.sum())
                sums[label] += selected.sum(axis=0)
                sumsq[label] += np.square(selected).sum(axis=0)
                expr_counts[label] += np.asarray((raw_block[mask] > 0).sum(axis=0)).ravel()
    finally:
        backed.file.close()

    control_n = counts[control_label]
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_mean = sums[control_label] / control_n
    control_var = np.zeros_like(control_mean)
    if control_n > 1:
        control_var = (sumsq[control_label] - (sums[control_label] ** 2) / control_n) / (control_n - 1)
    control_var = np.clip(control_var, a_min=0, a_max=None)

    effect_matrix = []
    statistic_matrix = []
    pvalue_matrix = []
    results: Dict[str, DifferentialExpressionResult] = {}

    for label in candidates:
        n_cells = counts[label]
        if n_cells == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        mean = sums[label] / n_cells
        var = np.zeros_like(mean)
        if n_cells > 1:
            var = (sumsq[label] - (sums[label] ** 2) / n_cells) / (n_cells - 1)
        var = np.clip(var, a_min=0, a_max=None)
        effect = mean - control_mean
        se = np.sqrt(control_var / control_n + var / n_cells)
        total_expr = expr_counts[label] + expr_counts[control_label]
        valid = (se > 0) & (total_expr >= min_cells_expressed)
        z = np.zeros_like(effect)
        pvalue = np.ones_like(effect)
        z[valid] = effect[valid] / se[valid]
        pvalue[valid] = 2 * norm.sf(np.abs(z[valid]))
        effect_matrix.append(effect)
        statistic_matrix.append(z)
        pvalue_matrix.append(pvalue)
        results[label] = DifferentialExpressionResult(
            genes=gene_symbols,
            effect_size=effect,
            statistic=z,
            pvalue=pvalue,
            method="wald",
            perturbation=label,
            result_path=Path(),  # placeholder updated after writing file
        )

    gene_symbols = pd.Index(gene_symbols).astype(str)

    if effect_matrix:
        effect_arr = np.vstack(effect_matrix)
        stat_arr = np.vstack(statistic_matrix)
        p_arr = np.vstack(pvalue_matrix)
    else:
        effect_arr = np.zeros((0, gene_symbols.size))
        stat_arr = np.zeros_like(effect_arr)
        p_arr = np.zeros_like(effect_arr)

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_arr, obs=obs, var=var)
    adata.layers["z_score"] = stat_arr
    adata.layers["pvalue"] = p_arr
    adata.uns["method"] = "wald"
    adata.uns["control_label"] = control_label
    output_path = resolve_output_path(path, suffix="wald_de", output_dir=output_dir, data_name=data_name)
    adata.write(output_path)

    for label, result in results.items():
        result.result_path = output_path

    return results


def wilcoxon_test(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    chunk_size: int = 256,
    tie_correct: bool = True,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> Dict[str, DifferentialExpressionResult]:
    """Perform a Wilcoxon rank-sum (Mann-Whitney U) test for each gene."""

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        if control_n == 0:
            raise ValueError("Control group contains no cells")
        pert_masks = {label: labels == label for label in candidates}
        for label, mask in pert_masks.items():
            if not np.any(mask):
                raise ValueError(f"Perturbation '{label}' contains no cells")
    finally:
        backed.file.close()

    effect_matrix = np.zeros((len(candidates), n_genes), dtype=float)
    u_matrix = np.zeros_like(effect_matrix)
    pvalue_matrix = np.ones_like(effect_matrix)
    z_matrix = np.zeros_like(effect_matrix)

    backed = read_backed(path)
    try:
        library_sizes = np.zeros(backed.n_obs, dtype=np.float64)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            _, lib = normalize_total_block(block)
            library_sizes[slc] = lib
    finally:
        backed.file.close()

    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_mask = labels == control_label
        pert_masks = {label: labels == label for label in candidates}
        for slc, block in iter_matrix_chunks(backed, axis=1, chunk_size=chunk_size):
            raw_block = np.asarray(block)
            normalised_block, _ = normalize_total_block(block, library_size=library_sizes)
            normalised_block = normalised_block.astype(np.float32, copy=False)
            log_block = np.log1p(normalised_block, dtype=np.float32)
            control_values = log_block[control_mask, :]
            control_expr = np.count_nonzero(raw_block[control_mask], axis=0)
            chunk_gene_indices = np.arange(slc.start, slc.stop)
            for idx, label in enumerate(candidates):
                pert_values = log_block[pert_masks[label], :]
                pert_expr = np.count_nonzero(raw_block[pert_masks[label]], axis=0)
                total_expr = control_expr + pert_expr
                low_expr = (control_expr < min_cells_expressed) & (pert_expr < min_cells_expressed)
                valid_mask = (total_expr >= min_cells_expressed) & ~low_expr
                if not np.any(valid_mask):
                    continue
                valid_cols = np.where(valid_mask)[0]
                selected_control = control_values[:, valid_cols]
                selected_pert = pert_values[:, valid_cols]
                combined = np.vstack((selected_pert, selected_control))
                ranks = rankdata(combined, axis=0)
                rank_sum = ranks[: selected_pert.shape[0]].sum(axis=0)
                if tie_correct:
                    tie = _tie_correction(ranks)
                else:
                    tie = np.ones_like(rank_sum)
                n_active = float(selected_pert.shape[0])
                m_active = float(selected_control.shape[0])
                expected = n_active * (n_active + m_active + 1.0) / 2.0
                std = np.sqrt(tie * n_active * m_active * (n_active + m_active + 1.0) / 12.0)
                u_stat = rank_sum - n_active * (n_active + 1.0) / 2.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = (rank_sum - expected) / std
                z = np.where(np.isfinite(z), z, 0.0)
                pvals = 2.0 * norm.sf(np.abs(z))
                effect = u_stat / (n_active * m_active) - 0.5

                gene_indices = chunk_gene_indices[valid_cols]
                u_matrix[idx, gene_indices] = u_stat
                pvalue_matrix[idx, gene_indices] = pvals
                effect_matrix[idx, gene_indices] = effect
                z_matrix[idx, gene_indices] = z
    finally:
        backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    pvalue_adj_matrix = _adjust_pvalue_matrix(pvalue_matrix, corr_method)
    adata.layers["u_statistic"] = u_matrix
    adata.layers["z_score"] = z_matrix
    adata.layers["pvalue"] = pvalue_matrix
    adata.layers["pvalue_adj"] = pvalue_adj_matrix
    adata.uns["method"] = "wilcoxon"
    adata.uns["control_label"] = control_label
    adata.uns["tie_correct"] = tie_correct
    adata.uns["pvalue_correction"] = corr_method
    output_path = resolve_output_path(path, suffix="wilcoxon_de", output_dir=output_dir, data_name=data_name)
    adata.write(output_path)
    results: Dict[str, DifferentialExpressionResult] = {}
    for idx, label in enumerate(candidates):
        results[label] = DifferentialExpressionResult(
            genes=gene_symbols,
            effect_size=effect_matrix[idx],
            statistic=u_matrix[idx],
            pvalue=pvalue_matrix[idx],
            method="wilcoxon",
            perturbation=label,
            result_path=output_path,
            pvalue_adj=pvalue_adj_matrix[idx],
        )

    return results

