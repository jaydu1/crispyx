"""Differential expression testing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

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
    stat_matrix = np.zeros_like(effect_matrix)
    pvalue_matrix = np.ones_like(effect_matrix)

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
            normalised_block, _ = normalize_total_block(block, library_size=library_sizes)
            control_values = normalised_block[control_mask, :]
            control_expr = np.count_nonzero(control_values, axis=0)
            for idx, label in enumerate(candidates):
                pert_values = normalised_block[pert_masks[label], :]
                pert_expr = np.count_nonzero(pert_values, axis=0)
                total_expr = control_expr + pert_expr
                for local, gene_idx in enumerate(range(slc.start, slc.stop)):
                    if total_expr[local] < min_cells_expressed:
                        continue
                    res = mannwhitneyu(
                        control_values[:, local],
                        pert_values[:, local],
                        alternative="two-sided",
                        method="auto",
                    )
                    stat_matrix[idx, gene_idx] = res.statistic
                    pvalue_matrix[idx, gene_idx] = res.pvalue
                    effect_matrix[idx, gene_idx] = (
                        res.statistic / (control_values.shape[0] * pert_values.shape[0]) - 0.5
                    )
    finally:
        backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    adata.layers["u_statistic"] = stat_matrix
    adata.layers["pvalue"] = pvalue_matrix
    adata.uns["method"] = "wilcoxon"
    adata.uns["control_label"] = control_label
    output_path = resolve_output_path(path, suffix="wilcoxon_de", output_dir=output_dir, data_name=data_name)
    adata.write(output_path)

    results: Dict[str, DifferentialExpressionResult] = {}
    for idx, label in enumerate(candidates):
        results[label] = DifferentialExpressionResult(
            genes=gene_symbols,
            effect_size=effect_matrix[idx],
            statistic=stat_matrix[idx],
            pvalue=pvalue_matrix[idx],
            method="wilcoxon",
            perturbation=label,
            result_path=output_path,
        )

    return results

