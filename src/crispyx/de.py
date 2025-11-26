"""Differential expression testing utilities."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

from .data import (
    AnnData,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
    resolve_control_label,
    resolve_output_path,
)
from .glm import NBGLMFitter, build_design_matrix

logger = logging.getLogger(__name__)


@dataclass
class DifferentialExpressionResult:
    genes: pd.Index
    effect_size: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    method: str
    perturbation: str
    pvalue_adj: np.ndarray | None = None
    result: AnnData | None = field(default=None, repr=False)

    @property
    def result_path(self) -> Path:
        if self.result is None:
            raise AttributeError("Result AnnData has not been initialised.")
        return self.result.path


@dataclass
class RankGenesGroupsResult(Mapping[str, DifferentialExpressionResult]):
    genes: pd.Index
    groups: list[str]
    statistics: np.ndarray
    pvalues: np.ndarray
    pvalues_adj: np.ndarray
    logfoldchanges: np.ndarray
    effect_size: np.ndarray
    u_statistics: np.ndarray
    pts: np.ndarray
    pts_rest: np.ndarray
    order: np.ndarray
    groupby: str
    method: str
    control_label: str
    tie_correct: bool
    pvalue_correction: Literal["benjamini-hochberg", "bonferroni"]
    result: AnnData | None = field(default=None, repr=False)
    _group_cache: Dict[str, DifferentialExpressionResult] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.genes = pd.Index(self.genes).astype(str)
        self.groups = [str(group) for group in self.groups]

    @property
    def result_path(self) -> Path:
        if self.result is None:
            raise AttributeError("Result AnnData has not been initialised.")
        return self.result.path

    def _ensure_cache(self) -> None:
        if self._group_cache:
            return
        for idx, group in enumerate(self.groups):
            self._group_cache[group] = DifferentialExpressionResult(
                genes=self.genes,
                effect_size=self.effect_size[idx],
                statistic=self.statistics[idx],
                pvalue=self.pvalues[idx],
                method=self.method,
                perturbation=group,
                pvalue_adj=self.pvalues_adj[idx],
                result=self.result,
            )

    def __getitem__(self, key: str) -> DifferentialExpressionResult:
        self._ensure_cache()
        return self._group_cache[key]

    def __iter__(self):  # type: ignore[override]
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def items(self):  # type: ignore[override]
        self._ensure_cache()
        return self._group_cache.items()

    def to_rank_genes_groups_dict(self) -> dict:
        gene_array = self.genes.to_numpy()
        sorted_names = gene_array[self.order]
        sorted_scores = np.take_along_axis(self.statistics, self.order, axis=1)
        sorted_lfc = np.take_along_axis(self.logfoldchanges, self.order, axis=1)
        sorted_pvals = np.take_along_axis(self.pvalues, self.order, axis=1)
        sorted_padj = np.take_along_axis(self.pvalues_adj, self.order, axis=1)
        sorted_pts = np.take_along_axis(self.pts, self.order, axis=1)
        sorted_pts_rest = np.take_along_axis(self.pts_rest, self.order, axis=1)
        sorted_effect = np.take_along_axis(self.effect_size, self.order, axis=1)
        sorted_u = np.take_along_axis(self.u_statistics, self.order, axis=1)

        def to_recarray(matrix: np.ndarray) -> np.recarray:
            arrays = [matrix[idx] for idx in range(matrix.shape[0])]
            return np.rec.fromarrays(arrays, names=self.groups)

        rank_genes_groups = {
            "params": {
                "groupby": self.groupby,
                "method": self.method,
                "reference": self.control_label,
                "tie_correct": self.tie_correct,
                "corr_method": self.pvalue_correction,
            },
            "names": to_recarray(sorted_names.astype(object)),
            "scores": to_recarray(sorted_scores),
            "logfoldchanges": to_recarray(sorted_lfc),
            "pvals": to_recarray(sorted_pvals),
            "pvals_adj": to_recarray(sorted_padj),
            "pts": to_recarray(sorted_pts),
            "pts_rest": to_recarray(sorted_pts_rest),
            "auc": to_recarray(sorted_effect),
            "u_stat": to_recarray(sorted_u),
        }
        rank_genes_groups["full"] = self.to_full_order_dict()
        return rank_genes_groups

    def to_full_order_dict(self) -> dict:
        return {
            "scores": self.statistics.copy(),
            "pvals": self.pvalues.copy(),
            "pvals_adj": self.pvalues_adj.copy(),
            "logfoldchanges": self.logfoldchanges.copy(),
            "auc": self.effect_size.copy(),
            "u_stat": self.u_statistics.copy(),
            "pts": self.pts.copy(),
            "pts_rest": self.pts_rest.copy(),
        }

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


def t_test(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    cell_chunk_size: int = 2048,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
) -> RankGenesGroupsResult:
    """Perform a t-test comparing log-expression means for each perturbation.
    
    Returns a RankGenesGroupsResult containing differential expression statistics
    in Scanpy-compatible format. Results are stored in an h5ad file with 
    `uns['rank_genes_groups']` containing logfoldchanges, scores (z-statistics),
    p-values, adjusted p-values, and proportion of expressing cells.
    
    The RankGenesGroupsResult implements the Mapping interface, so it can be used 
    like a dict: `result[perturbation_label]` returns a DifferentialExpressionResult
    for that perturbation.
    
    Parameters
    ----------
    path
        Path to an h5ad file containing expression data.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels.
    control_label
        Label for the control/reference group. If None, infers from common patterns.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for testing.
    cell_chunk_size
        Number of cells to process per chunk (memory vs. speed tradeoff). This
        controls streaming along the cell axis and is distinct from any future
        perturbation_chunk_size option that would batch perturbations.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "t_test" suffix.
    n_jobs
        Number of parallel workers for computing statistics across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
        If -1, uses all available cores.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results in Scanpy-compatible format. Access results
        via dict-like interface: `result[label].effect_size`, `result[label].pvalue`, etc.
        The h5ad file path is available at `result.result_path`.
    """

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        group_index = {label: idx for idx, label in enumerate(groups)}
        label_codes = pd.Categorical(labels, categories=groups).codes

        # Check if data is already log-transformed
        is_logged = False
        # Read a small chunk to check data type and range
        for _, chunk in iter_matrix_chunks(backed, axis=0, chunk_size=100, convert_to_dense=True):
            if np.issubdtype(chunk.dtype, np.integer):
                logger.warning("Input data appears to be raw counts (integers). t-test expects log-transformed data, but will perform normalization and log1p transformation automatically.")
            elif np.issubdtype(chunk.dtype, np.floating):
                # Check if max value is small (heuristic for log-transformed)
                # 20 is a safe upper bound for log1p(counts) usually (e^20 is huge)
                if chunk.max() < 20: 
                    # Check if it looks like integers (e.g. 1.0, 2.0)
                    if np.all(np.mod(chunk[chunk > 0], 1) == 0):
                        logger.warning("Input data is floats but appears to be integer counts. Will perform normalization and log1p.")
                    else:
                        is_logged = True
                        logger.info("Input data appears to be log-transformed. Skipping normalization and log1p.")
                else:
                    logger.warning("Input data is floats with large values. Treating as raw counts.")
            break # Only check the first chunk

        n_groups_total = len(groups)
        # Use float64 for accumulation to maintain numerical precision
        sums = np.zeros((n_groups_total, n_genes), dtype=np.float64)
        sumsq = np.zeros((n_groups_total, n_genes), dtype=np.float64)
        counts = np.zeros(n_groups_total, dtype=np.int64)
        expr_counts = np.zeros((n_groups_total, n_genes), dtype=np.int32)
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=cell_chunk_size, convert_to_dense=False
        ):
            slice_codes = label_codes[slc]
            
            # Optimize memory: sparse-aware expression counting
            if sp.issparse(block):
                # Convert to CSR for efficient row iteration, then get binary indicator per gene
                csr = sp.csr_matrix(block)
                # For each unique group code in this slice, accumulate expression counts
                for code in np.unique(slice_codes):
                    row_mask = slice_codes == code
                    group_block = csr[row_mask, :]
                    # Count non-zeros per column (gene) - getnnz returns count per column
                    expr_counts[code] += np.asarray(group_block.getnnz(axis=0), dtype=np.int32)
            else:
                mask = block > 0
                np.add.at(expr_counts, slice_codes, mask)
                del mask

            if is_logged:
                if sp.issparse(block):
                    log_block = block.toarray()
                else:
                    log_block = block
                # Ensure float32 for consistency
                if log_block.dtype != np.float32:
                    log_block = log_block.astype(np.float32)
            else:
                # Optimize memory: in-place log1p
                normalised_block, _ = normalize_total_block(block, dtype=np.float32)
                np.log1p(normalised_block, out=normalised_block)
                log_block = normalised_block
            
            np.add.at(sums, slice_codes, log_block)
            # Compute squared values in-place to avoid temporary array allocation
            np.square(log_block, out=log_block)
            np.add.at(sumsq, slice_codes, log_block)
            del log_block  # Explicitly free the chunk memory
            counts += np.bincount(slice_codes, minlength=n_groups_total)
    finally:
        backed.file.close()

    control_idx = group_index[control_label]
    control_n = counts[control_idx]
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_mean = sums[control_idx] / control_n
    control_var = np.zeros_like(control_mean)
    if control_n > 1:
        control_var = (sumsq[control_idx] - (sums[control_idx] ** 2) / control_n) / (control_n - 1)
    control_var = np.clip(control_var, a_min=0, a_max=None)

    # Calculate control pts (proportion of cells expressing each gene)
    control_pts = np.divide(
        expr_counts[control_idx],
        control_n,
        out=np.zeros(n_genes, dtype=float),
        where=control_n > 0,
    )

    # Determine worker count for parallelization
    n_groups = len(candidates)
    max_available_workers = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        worker_count = min(n_groups, max_available_workers)
    else:
        worker_count = min(n_groups, abs(n_jobs))
    worker_count = max(worker_count, 1)

    # Pre-allocate result arrays to avoid memory spikes from list collection and vstack
    # Use float32 for effect/lfc/pts to match scanpy's approach; keep float64 for p-values and statistics
    shape = (n_groups, n_genes)
    effect_arr = np.zeros(shape, dtype=np.float32)
    stat_arr = np.zeros(shape, dtype=np.float64)
    p_arr = np.ones(shape, dtype=np.float64)
    lfc_arr = np.zeros(shape, dtype=np.float32)
    pts_arr = np.zeros(shape, dtype=np.float32)
    # Store control_pts as 1D array; broadcast when creating RankGenesGroupsResult
    pts_rest_1d = control_pts.astype(np.float32)
    order_arr = np.zeros(shape, dtype=np.int64)

    def compute_perturbation(label: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute statistics for a single perturbation."""
        idx = group_index[label]
        n_cells = counts[idx]
        if n_cells == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        # Use in-place operations to minimize temporary arrays
        mean = sums[idx] / n_cells
        # Compute variance in-place
        var = sumsq[idx].copy()
        if n_cells > 1:
            np.subtract(var, np.square(sums[idx]) / n_cells, out=var)
            np.divide(var, n_cells - 1, out=var)
        else:
            var.fill(0)
        np.clip(var, a_min=0, a_max=None, out=var)
        
        # effect = mean - control_mean (reuse mean array)
        effect = mean
        np.subtract(effect, control_mean, out=effect)
        
        # Compute standard error in-place using var array
        se = var  # Reuse var array for se
        np.divide(se, n_cells, out=se)
        np.add(se, control_var / control_n, out=se)
        np.sqrt(se, out=se)
        
        total_expr = expr_counts[idx] + expr_counts[control_idx]
        valid = (se > 0) & (total_expr >= min_cells_expressed)
        
        z = np.zeros(n_genes, dtype=np.float64)
        pvalue = np.ones(n_genes, dtype=np.float64)
        z[valid] = effect[valid] / se[valid]
        pvalue[valid] = 2 * norm.sf(np.abs(z[valid]))

        # Calculate pts (proportion of cells expressing each gene)
        pts = np.zeros(n_genes, dtype=np.float32)
        np.divide(expr_counts[idx], n_cells, out=pts, where=n_cells > 0, casting='unsafe')
        
        # Create ranking order (descending by absolute z-score)
        order = np.argsort(-np.abs(z))
        
        # Return effect as float32 for memory efficiency
        return label, effect.astype(np.float32), z, pvalue, effect.astype(np.float32), pts, order

    # Compute in parallel or sequentially based on worker count
    if n_groups > 0:
        # Map candidates to their index in the result arrays
        candidate_indices = {label: i for i, label in enumerate(candidates)}
        
        if n_groups == 1 or worker_count == 1:
            for label in candidates:
                _, effect, z, pvalue, log_fc, pts, order = compute_perturbation(label)
                i = candidate_indices[label]
                effect_arr[i] = effect
                stat_arr[i] = z
                p_arr[i] = pvalue
                lfc_arr[i] = log_fc
                pts_arr[i] = pts
                order_arr[i] = order
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(compute_perturbation, label): label for label in candidates}
                for future in as_completed(futures):
                    label_res, effect, z, pvalue, log_fc, pts, order = future.result()
                    i = candidate_indices[label_res]
                    effect_arr[i] = effect
                    stat_arr[i] = z
                    p_arr[i] = pvalue
                    lfc_arr[i] = log_fc
                    pts_arr[i] = pts
                    order_arr[i] = order
                    # Help GC
                    del effect, z, pvalue, log_fc, pts, order

    # Apply p-value correction
    pvalue_adj_arr = _adjust_pvalue_matrix(p_arr, method="benjamini-hochberg")

    # Create AnnData with rank_genes_groups format in uns
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    
    # Create minimal AnnData - the actual results are in uns["rank_genes_groups"]
    adata = ad.AnnData(np.zeros((len(candidates), 0)), obs=obs, var=pd.DataFrame(index=[]))
    adata.uns["method"] = "t_test"
    adata.uns["control_label"] = control_label
    adata.uns["genes"] = gene_symbols.to_numpy()
    adata.uns["pvalue_correction"] = "benjamini-hochberg"
    
    output_path = resolve_output_path(path, suffix="t_test", output_dir=output_dir, data_name=data_name)
    
    # Broadcast pts_rest from 1D to 2D at final result construction (avoids storing full copy)
    pts_rest_arr = np.broadcast_to(pts_rest_1d, shape).copy()
    
    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=stat_arr,
        pvalues=p_arr,
        pvalues_adj=pvalue_adj_arr,
        logfoldchanges=lfc_arr,
        effect_size=effect_arr,
        u_statistics=np.zeros(shape, dtype=np.float32),  # Not used in t-test
        pts=pts_arr,
        pts_rest=pts_rest_arr,
        order=order_arr,
        groupby=perturbation_column,
        method="t_test",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction="benjamini-hochberg",
        result=None,
    )
    
    # Add rank_genes_groups to uns for Scanpy compatibility
    adata.uns["rank_genes_groups"] = result.to_rank_genes_groups_dict()
    adata.write(output_path)
    
    result.result = AnnData(output_path)
    return result


def nb_glm_test(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    covariates: Iterable[str] | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    dispersion: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    poisson_init_iter: int = 15,
    min_cells_expressed: int = 0,
    min_total_count: float = 1.0,
    chunk_size: int = 256,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
) -> RankGenesGroupsResult:
    """Perform negative binomial GLM differential expression test.
    
    Returns a RankGenesGroupsResult containing differential expression statistics
    in Scanpy-compatible format. Uses a negative binomial GLM framework that can
    incorporate covariates. Results are stored in an h5ad file with 
    `uns['rank_genes_groups']` containing logfoldchanges, scores (Wald statistics),
    p-values, adjusted p-values, and proportion of expressing cells.
    
    The RankGenesGroupsResult implements the Mapping interface, so it can be used 
    like a dict: `result[perturbation_label]` returns a DifferentialExpressionResult
    for that perturbation.
    
    Parameters
    ----------
    path
        Path to an h5ad file containing raw count data.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels.
    control_label
        Label for the control/reference group. If None, infers from common patterns.
    covariates
        Additional columns in `adata.obs` to include as covariates in the GLM.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    dispersion
        Fixed dispersion parameter for negative binomial. If None, estimates per gene.
    max_iter
        Maximum iterations for GLM fitting.
    tol
        Convergence tolerance for GLM fitting.
    poisson_init_iter
        Initial Poisson iterations before switching to negative binomial.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for testing.
    min_total_count
        Minimum total count across all cells for a gene to be tested.
    chunk_size
        Number of genes to process per chunk (memory vs. speed tradeoff).
    corr_method
        Method for p-value correction: "benjamini-hochberg" or "bonferroni".
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "nb_glm" suffix.
    n_jobs
        Number of parallel workers for fitting GLMs across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
        If -1, uses all available cores.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results in Scanpy-compatible format. Access results
        via dict-like interface: `result[label].effect_size`, `result[label].pvalue`, etc.
        The h5ad file path is available at `result.result_path`.
    """

    covariates = list(covariates or [])

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        obs_df = backed.obs[[perturbation_column] + covariates].copy()
        labels = obs_df[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        if control_n == 0:
            raise ValueError("Control group contains no cells")
        for label in candidates:
            if not np.any(labels == label):
                raise ValueError(f"Perturbation '{label}' contains no cells")
    finally:
        backed.file.close()

    library_sizes = np.zeros(obs_df.shape[0], dtype=np.float64)
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            dense = np.asarray(block, dtype=np.float64)
            library_sizes[slc] = dense.sum(axis=1)
    finally:
        backed.file.close()

    offset = np.log(np.clip(library_sizes, 1e-8, None))

    n_groups = len(candidates)
    effect_matrix = np.zeros((n_groups, n_genes), dtype=np.float64)
    statistic_matrix = np.zeros_like(effect_matrix)
    pvalue_matrix = np.ones_like(effect_matrix)
    logfc_matrix = np.zeros_like(effect_matrix)
    se_matrix = np.full_like(effect_matrix, np.inf)
    pts_matrix = np.zeros_like(effect_matrix)
    pts_rest_matrix = np.zeros_like(effect_matrix)
    dispersion_matrix = np.zeros_like(effect_matrix)
    iter_matrix = np.zeros_like(effect_matrix)
    convergence_matrix = np.zeros_like(effect_matrix, dtype=bool)

    # Note: n_jobs parameter accepted but not yet implemented for nb_glm_test
    # due to file handle management complexity with per-perturbation chunking
    # TODO: Implement parallelization using process-based approach with separate file handles
    for group_idx, label in enumerate(candidates):
        group_mask = labels == label
        subset_mask = control_mask | group_mask
        subset_obs = obs_df.iloc[subset_mask]
        indicator = group_mask[subset_mask].astype(np.float64)
        design, design_columns = build_design_matrix(
            subset_obs,
            covariate_columns=covariates,
            perturbation_indicator=indicator,
            intercept=True,
        )
        perturbation_column_index = design_columns.index("perturbation")
        fitter = NBGLMFitter(
            design,
            offset=offset[subset_mask],
            dispersion=dispersion,
            max_iter=max_iter,
            tol=tol,
            poisson_init_iter=poisson_init_iter,
            min_total_count=min_total_count,
        )
        control_subset_mask = control_mask[subset_mask]
        group_subset_mask = group_mask[subset_mask]
        group_n = int(group_subset_mask.sum())

        backed = read_backed(path)
        try:
            for slc, block in iter_matrix_chunks(backed, axis=1, chunk_size=chunk_size):
                raw_block = np.asarray(block, dtype=np.float64)
                subset_block = raw_block[subset_mask, :]
                control_block = subset_block[control_subset_mask, :]
                group_block = subset_block[group_subset_mask, :]
                control_expr = np.asarray(np.count_nonzero(control_block, axis=0))
                group_expr = np.asarray(np.count_nonzero(group_block, axis=0))
                total_expr = control_expr + group_expr
                valid_mask = total_expr >= min_cells_expressed
                chunk_indices = np.arange(slc.start, slc.stop)
                pts = np.divide(
                    group_expr,
                    group_n,
                    out=np.zeros_like(group_expr, dtype=float),
                    where=group_n > 0,
                )
                pts_rest = np.divide(
                    control_expr,
                    control_n,
                    out=np.zeros_like(control_expr, dtype=float),
                    where=control_n > 0,
                )
                pts_matrix[group_idx, chunk_indices] = np.where(valid_mask, pts, 0.0)
                pts_rest_matrix[group_idx, chunk_indices] = np.where(valid_mask, pts_rest, 0.0)

                if not np.any(valid_mask):
                    continue

                fit_block = subset_block[:, valid_mask]
                gene_results = fitter.fit_matrix(fit_block)
                result_iter = iter(gene_results)

                for local_idx, gene_idx in enumerate(chunk_indices):
                    if not valid_mask[local_idx]:
                        continue
                    result = next(result_iter)
                    convergence_matrix[group_idx, gene_idx] = result.converged
                    iter_matrix[group_idx, gene_idx] = result.n_iter
                    dispersion_matrix[group_idx, gene_idx] = result.dispersion
                    coef = float(result.coef[perturbation_column_index])
                    se = float(result.se[perturbation_column_index])
                    if not result.converged or not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
                        continue
                    statistic = coef / se
                    pvalue = float(2.0 * norm.sf(abs(statistic)))
                    effect_matrix[group_idx, gene_idx] = coef
                    statistic_matrix[group_idx, gene_idx] = statistic
                    pvalue_matrix[group_idx, gene_idx] = pvalue
                    logfc_matrix[group_idx, gene_idx] = coef
                    se_matrix[group_idx, gene_idx] = se
        finally:
            backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    order_matrix = np.argsort(-np.abs(statistic_matrix), axis=1, kind="mergesort")
    pvalue_adj_matrix = _adjust_pvalue_matrix(pvalue_matrix, corr_method)

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)

    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    adata.layers["z_score"] = statistic_matrix
    adata.layers["pvalue"] = pvalue_matrix
    adata.layers["pvalue_adj"] = pvalue_adj_matrix
    adata.layers["logfoldchange"] = logfc_matrix
    adata.layers["standard_error"] = se_matrix
    adata.layers["dispersion"] = dispersion_matrix
    adata.layers["converged"] = convergence_matrix.astype(np.float32)
    adata.layers["iterations"] = iter_matrix.astype(np.float32)
    adata.layers["pts"] = pts_matrix
    adata.layers["pts_rest"] = pts_rest_matrix
    adata.uns["method"] = "nb_glm"
    adata.uns["control_label"] = control_label
    adata.uns["covariates"] = covariates

    output_path = resolve_output_path(
        path,
        suffix="nb_glm",
        output_dir=output_dir,
        data_name=data_name,
    )
    adata.write(output_path)

    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=statistic_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj_matrix,
        logfoldchanges=logfc_matrix,
        effect_size=effect_matrix,
        u_statistics=np.zeros_like(effect_matrix),
        pts=pts_matrix,
        pts_rest=pts_rest_matrix,
        order=order_matrix,
        groupby=perturbation_column,
        method="nb_glm",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=AnnData(output_path),
    )
    return result


def wilcoxon_test(
    path: str | Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    chunk_size: int = 256,
    tie_correct: bool = True,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
) -> RankGenesGroupsResult:
    """Perform a Wilcoxon rank-sum (Mann-Whitney U) test for each gene."""

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
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

    n_groups = len(candidates)
    effect_matrix = np.zeros((n_groups, n_genes), dtype=float)
    u_matrix = np.zeros_like(effect_matrix)
    pvalue_matrix = np.ones_like(effect_matrix)
    z_matrix = np.zeros_like(effect_matrix)
    lfc_matrix = np.zeros_like(effect_matrix)
    pts_matrix = np.zeros_like(effect_matrix)
    pts_rest_matrix = np.zeros_like(effect_matrix)
    order_matrix = np.zeros((n_groups, n_genes), dtype=np.int64)

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
        max_available_workers = os.cpu_count() or 1
        if n_jobs is None or n_jobs == 0:
            worker_count = min(n_groups, max_available_workers)
        else:
            worker_count = min(n_groups, abs(n_jobs))
        worker_count = max(worker_count, 1)

        for slc, block in iter_matrix_chunks(backed, axis=1, chunk_size=chunk_size):
            raw_block = np.asarray(block)
            normalised_block, _ = normalize_total_block(block, library_size=library_sizes)
            normalised_block = normalised_block.astype(np.float32, copy=False)
            log_block = np.log1p(normalised_block, dtype=np.float32)
            control_values = log_block[control_mask, :]
            control_expr = np.asarray(np.count_nonzero(raw_block[control_mask], axis=0))
            control_mean = (
                control_values.mean(axis=0, dtype=np.float64)
                if control_values.size
                else np.zeros(control_values.shape[1], dtype=np.float64)
            )
            control_pts = np.divide(
                control_expr,
                control_n,
                out=np.zeros_like(control_expr, dtype=float),
                where=control_n > 0,
            )
            chunk_gene_indices = np.arange(slc.start, slc.stop)

            def compute_group(
                idx: int, label: str
            ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                mask = pert_masks[label]
                group_values = log_block[mask, :]
                group_expr = np.asarray(np.count_nonzero(raw_block[mask], axis=0))
                group_mean = (
                    group_values.mean(axis=0, dtype=np.float64)
                    if group_values.size
                    else np.zeros_like(control_mean)
                )
                total_expr = control_expr + group_expr
                low_expr = (control_expr < min_cells_expressed) & (
                    group_expr < min_cells_expressed
                )
                valid = (total_expr >= min_cells_expressed) & ~low_expr

                full_u = np.zeros(control_values.shape[1], dtype=float)
                full_z = np.zeros_like(full_u)
                full_p = np.ones_like(full_u)
                full_effect = np.zeros_like(full_u)

                if np.any(valid):
                    valid_cols = np.where(valid)[0]
                    selected_control = control_values[:, valid_cols]
                    selected_group = group_values[:, valid_cols]
                    combined = np.vstack((selected_group, selected_control))
                    ranks = rankdata(combined, axis=0)
                    if tie_correct:
                        tie = _tie_correction(ranks)
                    else:
                        tie = np.ones(ranks.shape[1], dtype=np.float64)
                    n_active = float(group_values.shape[0])
                    m_active = float(control_values.shape[0])
                    rank_sum = ranks[: selected_group.shape[0]].sum(axis=0)
                    expected = n_active * (n_active + m_active + 1.0) / 2.0
                    std = np.sqrt(tie * n_active * m_active * (n_active + m_active + 1.0) / 12.0)
                    u_stat = rank_sum - n_active * (n_active + 1.0) / 2.0
                    valid_std = std > 0
                    z = np.zeros_like(rank_sum)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z[valid_std] = (rank_sum[valid_std] - expected) / std[valid_std]
                    z[~np.isfinite(z)] = 0.0
                    pvals = np.ones_like(rank_sum)
                    pvals[valid_std] = 2.0 * norm.sf(np.abs(z[valid_std]))
                    effect = np.zeros_like(rank_sum)
                    if n_active > 0 and m_active > 0:
                        effect = u_stat / (n_active * m_active) - 0.5
                    full_u[valid_cols] = u_stat
                    full_z[valid_cols] = z
                    full_p[valid_cols] = pvals
                    full_effect[valid_cols] = effect

                pts = np.divide(
                    group_expr,
                    float(group_values.shape[0]),
                    out=np.zeros_like(group_expr, dtype=float),
                    where=group_values.shape[0] > 0,
                )
                pts = np.where(valid, pts, 0.0)
                pts_rest = np.where(valid, control_pts, 0.0)
                log_fc = group_mean - control_mean
                log_fc = np.where(valid, log_fc, 0.0)
                return idx, full_u, full_z, full_p, full_effect, log_fc, pts, pts_rest

            tasks = [(idx, label) for idx, label in enumerate(candidates)]
            if n_groups == 0:
                continue
            if n_groups == 1 or worker_count == 1:
                computed = [compute_group(idx, label) for idx, label in tasks]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(compute_group, idx, label) for idx, label in tasks]
                    computed = [future.result() for future in futures]
            for idx, u_stat, z, pvals, effect, log_fc, pts, pts_rest in computed:
                gene_indices = chunk_gene_indices
                u_matrix[idx, gene_indices] = u_stat
                pvalue_matrix[idx, gene_indices] = pvals
                effect_matrix[idx, gene_indices] = effect
                z_matrix[idx, gene_indices] = z
                lfc_matrix[idx, gene_indices] = log_fc
                pts_matrix[idx, gene_indices] = pts
                pts_rest_matrix[idx, gene_indices] = pts_rest
    finally:
        backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    gene_array = gene_symbols.to_numpy()
    order_matrix = np.argsort(-z_matrix, axis=1, kind="mergesort")
    pvalue_adj_matrix = _adjust_pvalue_matrix(pvalue_matrix, corr_method)

    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=z_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj_matrix,
        logfoldchanges=lfc_matrix,
        effect_size=effect_matrix,
        u_statistics=u_matrix,
        pts=pts_matrix,
        pts_rest=pts_rest_matrix,
        order=order_matrix,
        groupby=perturbation_column,
        method="wilcoxon",
        control_label=control_label,
        tie_correct=tie_correct,
        pvalue_correction=corr_method,
    )

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    adata = ad.AnnData(np.zeros((len(candidates), 0)), obs=pd.DataFrame(index=obs_index))
    adata.uns["rank_genes_groups"] = result.to_rank_genes_groups_dict()
    adata.uns["genes"] = gene_array
    adata.uns["method"] = "wilcoxon"
    adata.uns["control_label"] = control_label
    adata.uns["tie_correct"] = tie_correct
    adata.uns["pvalue_correction"] = corr_method
    output_path = resolve_output_path(path, suffix="wilcoxon", output_dir=output_dir, data_name=data_name)
    adata.write(output_path)
    result.result = AnnData(output_path)

    return result

