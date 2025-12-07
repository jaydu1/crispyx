"""Differential expression testing utilities."""

from __future__ import annotations

import logging
import os
import resource
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Tuple

from joblib import Parallel, delayed

from numpy.typing import ArrayLike

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import norm, rankdata

from .data import (
    AnnData,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    read_backed,
    resolve_control_label,
    resolve_output_path,
)
from .glm import (
    NBGLMFitter,
    NBGLMBatchFitter,
    JointModelResult,
    build_design_matrix,
    estimate_covariate_effects_streaming,
    estimate_dispersion_map,
    estimate_global_dispersion_streaming,
    estimate_joint_model_streaming,
    estimate_joint_model_lbfgsb,
    fit_dispersion_trend,
    shrink_dispersions,
    shrink_log_foldchange,
    shrink_lfc_apeglm,
)

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
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return a p-value matrix adjusted per row using the requested method.

    Parameters
    ----------
    matrix
        P-value matrix to correct.
    method
        Correction method to apply.
    out
        Optional array to write results into. If provided, must be the same shape
        as ``matrix``.
    """

    adjusted = out if out is not None else np.ones_like(matrix)
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


def _validate_size_factors(
    size_factors: ArrayLike, n_cells: int, *, scale: bool = True
) -> np.ndarray:
    size_factors_arr = np.asarray(size_factors, dtype=np.float64).reshape(-1)
    if size_factors_arr.shape[0] != n_cells:
        raise ValueError(
            f"Provided size_factors have length {size_factors_arr.shape[0]} but expected {n_cells}"
        )
    mask = np.isfinite(size_factors_arr) & (size_factors_arr > 0)
    if not np.any(mask):
        raise ValueError("Provided size_factors contain no positive finite values")
    size_factors_arr[~mask] = np.nanmedian(size_factors_arr[mask])
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors_arr, 1e-12, None))))
        return size_factors_arr / scale_factor
    return size_factors_arr


def _median_of_ratios_size_factors(
    path: str | Path, *, chunk_size: int = 256, scale: bool = True
) -> np.ndarray:
    backed = read_backed(path)
    n_cells = backed.n_obs
    n_genes = backed.n_vars
    log_sum = np.zeros(n_genes, dtype=np.float64)
    log_count = np.zeros(n_genes, dtype=np.int64)
    try:
        for _, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block)
            csc = csr.tocsc()
            counts_per_gene = np.diff(csc.indptr)
            if not counts_per_gene.any():
                continue
            gene_indices = np.repeat(np.arange(n_genes, dtype=np.int64), counts_per_gene)
            data = np.log(np.clip(csc.data, 1e-12, None))
            np.add.at(log_sum, gene_indices, data)
            np.add.at(log_count, gene_indices, 1)
    finally:
        backed.file.close()
    geo_means = np.zeros(n_genes, dtype=np.float64)
    valid_geo = log_count > 0
    geo_means[valid_geo] = np.exp(log_sum[valid_geo] / log_count[valid_geo])

    size_factors = np.full(n_cells, np.nan, dtype=np.float64)
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block)
            for row_idx in range(csr.shape[0]):
                start = csr.indptr[row_idx]
                end = csr.indptr[row_idx + 1]
                gene_idx = csr.indices[start:end]
                data = csr.data[start:end]
                mask = geo_means[gene_idx] > 0
                if not np.any(mask):
                    continue
                ratios = data[mask] / geo_means[gene_idx[mask]]
                ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                if ratios.size == 0:
                    continue
                size_factors[slc.start + row_idx] = np.median(ratios)
    finally:
        backed.file.close()

    valid_sf = np.isfinite(size_factors) & (size_factors > 0)
    if not np.any(valid_sf):
        return np.ones(n_cells, dtype=np.float64)
    fallback = np.nanmedian(size_factors[valid_sf])
    size_factors[~valid_sf] = fallback
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors, 1e-12, None))))
        return size_factors / scale_factor
    return size_factors


def _deseq2_style_size_factors(
    path: str | Path, *, chunk_size: int = 256, scale: bool = True
) -> np.ndarray:
    """Compute DESeq2-style size factors using only genes expressed in all cells.
    
    This method computes size factors in the same way as DESeq2/PyDESeq2:
    1. Find genes expressed (count > 0) in ALL cells
    2. Compute geometric mean of counts per gene across all cells
    3. For each cell, compute ratios of counts to geometric means
    4. Take median of ratios as the size factor
    
    This approach works well for bulk RNA-seq where most genes are expressed in all
    samples, but may have issues with very sparse single-cell data where few genes
    are expressed in all cells.
    
    Parameters
    ----------
    path
        Path to h5ad file with count data.
    chunk_size
        Number of cells to process per chunk.
    scale
        If True, rescale size factors so their geometric mean equals 1.
        
    Returns
    -------
    np.ndarray
        Size factors for each cell.
    """
    from scipy.stats import gmean
    
    backed = read_backed(path)
    n_cells = backed.n_obs
    n_genes = backed.n_vars
    
    # First pass: find genes expressed in all cells
    all_expressed = np.ones(n_genes, dtype=bool)
    try:
        for _, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block)
            # Check which genes have zero counts in this chunk
            gene_has_nonzero = np.zeros(n_genes, dtype=bool)
            if csr.nnz > 0:
                gene_indices = csr.indices
                gene_has_nonzero[gene_indices] = True
            # Genes that have zeros in any chunk are not "all expressed"
            genes_with_zeros = ~gene_has_nonzero
            # For each row in chunk, check if gene has zero
            for row_idx in range(csr.shape[0]):
                start = csr.indptr[row_idx]
                end = csr.indptr[row_idx + 1]
                row_gene_indices = set(csr.indices[start:end])
                for gene_idx in range(n_genes):
                    if gene_idx not in row_gene_indices:
                        all_expressed[gene_idx] = False
    finally:
        backed.file.close()
    
    n_all_expressed = np.sum(all_expressed)
    if n_all_expressed < 10:
        logger.warning(
            f"Only {n_all_expressed} genes expressed in all cells. "
            "Consider using size_factor_method='sparse' for sparse data."
        )
    if n_all_expressed == 0:
        logger.warning(
            "No genes expressed in all cells. Falling back to sparse method."
        )
        return _median_of_ratios_size_factors(path, chunk_size=chunk_size, scale=scale)
    
    # Collect counts for all-expressed genes
    all_expressed_idx = np.where(all_expressed)[0]
    counts_filtered = np.zeros((n_cells, n_all_expressed), dtype=np.float64)
    
    backed = read_backed(path)
    try:
        cell_offset = 0
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            block_arr = np.asarray(block)
            counts_filtered[slc.start:slc.stop, :] = block_arr[:, all_expressed_idx]
            cell_offset = slc.stop
    finally:
        backed.file.close()
    
    # Compute geometric means per gene
    geo_means = gmean(counts_filtered, axis=0)
    
    # Compute ratios and take median
    ratios = counts_filtered / geo_means
    size_factors = np.median(ratios, axis=1)
    
    # Handle any invalid values
    valid_sf = np.isfinite(size_factors) & (size_factors > 0)
    if not np.all(valid_sf):
        fallback = np.nanmedian(size_factors[valid_sf])
        size_factors[~valid_sf] = fallback
    
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors, 1e-12, None))))
        return size_factors / scale_factor
    return size_factors


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

    Input data **should already be normalized and log-transformed** (for example
    using `scanpy.pp.normalize_total` followed by `scanpy.pp.log1p`). To maintain
    backward compatibility with Scanpy-style workflows, count-like inputs are
    automatically normalized and log-transformed in streaming fashion, with a
    warning to encourage explicit preprocessing upstream.
    
    Parameters
    ----------
    path
        Path to an h5ad file containing log-transformed expression data.
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
        perturbation_chunk_size option that would batch perturbations. Data must
        already be normalized/log-transformed before chunking.
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
        control_label = resolve_control_label(list(labels), control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        group_index = {label: idx for idx, label in enumerate(groups)}
        label_codes = pd.Categorical(labels, categories=groups).codes

        # Only allow sparse matrices; raise error if dense
        for _, chunk in iter_matrix_chunks(backed, axis=0, chunk_size=100, convert_to_dense=False):
            if not sp.issparse(chunk):
                raise ValueError(
                    "t_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                )
            # Optionally warn if data looks like counts
            if np.issubdtype(chunk.dtype, np.integer):
                logger.warning(
                    "Detected integer count data in t_test; input should be normalized/log-transformed. "
                    "For reproducibility, please preprocess explicitly upstream."
                )
            elif np.issubdtype(chunk.dtype, np.floating):
                non_zero = chunk.data[chunk.data > 0]
                is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
                if is_count_like:
                    logger.warning(
                        "Detected count-like floating point values in t_test; input should be normalized/log-transformed. "
                        "Please ensure preprocessing is applied upstream for consistent results."
                    )
            break  # Only check the first chunk

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
            csr = sp.csr_matrix(block)
            for code in np.unique(slice_codes):
                row_mask = slice_codes == code
                group_block = csr[row_mask, :]
                # Expression count: number of nonzero per gene
                expr_counts[code] += np.asarray(group_block.getnnz(axis=0), dtype=np.int32)
                # Sum and sumsq using sparse ops
                sums[code] += group_block.sum(axis=0).A1.astype(np.float64)
                sumsq[code] += group_block.power(2).sum(axis=0).A1.astype(np.float64)
                counts[code] += row_mask.sum()
    finally:
        backed.file.close()

    control_idx = group_index[control_label]
    control_n = counts[control_idx]
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_mean = sums[control_idx] / control_n
    # Precompute control term for LFC calculation once (avoid recomputing per perturbation)
    control_mean_expm1 = np.expm1(control_mean) + 1e-9
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

    # Prepare on-disk buffers for results
    shape = (n_groups, n_genes)
    output_path = resolve_output_path(path, suffix="t_test", output_dir=output_dir, data_name=data_name)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)

    adata = ad.AnnData(np.zeros((len(candidates), 0)), obs=obs, var=pd.DataFrame(index=[]))
    adata.uns["method"] = "t_test"
    adata.uns["control_label"] = control_label
    adata.uns["genes"] = gene_symbols.to_numpy()
    adata.uns["pvalue_correction"] = "benjamini-hochberg"
    adata.write(output_path)

    candidate_indices = {label: i for i, label in enumerate(candidates)}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        stat_memmap = np.memmap(tmp_path / "statistics.dat", mode="w+", dtype=np.float64, shape=shape)
        pval_memmap = np.memmap(tmp_path / "pvalues.dat", mode="w+", dtype=np.float64, shape=shape)
        lfc_memmap = np.memmap(tmp_path / "logfoldchanges.dat", mode="w+", dtype=np.float32, shape=shape)
        effect_memmap = np.memmap(tmp_path / "effect_size.dat", mode="w+", dtype=np.float32, shape=shape)
        pts_memmap = np.memmap(tmp_path / "pts.dat", mode="w+", dtype=np.float32, shape=shape)
        order_memmap = np.memmap(tmp_path / "order.dat", mode="w+", dtype=np.int64, shape=shape)
        pts_rest_memmap = np.memmap(
            tmp_path / "pts_rest.dat", mode="w+", dtype=np.float32, shape=shape
        )
        pts_rest_memmap[:] = control_pts.astype(np.float32)

        with h5py.File(output_path, "r+") as handle:
            uns_group = handle.require_group("uns")
            if "rank_genes_groups" in uns_group:
                del uns_group["rank_genes_groups"]
            rgg = uns_group.create_group("rank_genes_groups")
            full = rgg.create_group("full")
            ds_scores = full.create_dataset("scores", shape=shape, dtype="float64")
            ds_pvals = full.create_dataset("pvals", shape=shape, dtype="float64")
            ds_pvals_adj = full.create_dataset("pvals_adj", shape=shape, dtype="float64")
            ds_lfc = full.create_dataset("logfoldchanges", shape=shape, dtype="float32")
            ds_auc = full.create_dataset("auc", shape=shape, dtype="float32")
            ds_u = full.create_dataset("u_stat", shape=shape, dtype="float32")
            ds_pts = full.create_dataset("pts", shape=shape, dtype="float32")
            ds_pts_rest = full.create_dataset("pts_rest", shape=shape, dtype="float32")
            ds_order = rgg.create_dataset("order", shape=shape, dtype="int64")

            ds_auc[:] = 0.0
            ds_u[:] = 0.0
            ds_pts_rest[:] = pts_rest_memmap

            batch_size = worker_count
            effect_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
            stat_buffer = np.zeros((batch_size, n_genes), dtype=np.float64)
            pval_buffer = np.ones((batch_size, n_genes), dtype=np.float64)
            lfc_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
            pts_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
            order_buffer = np.zeros((batch_size, n_genes), dtype=np.int64)
            mean_buffer = np.zeros(n_genes, dtype=np.float64)
            var_buffer = np.zeros(n_genes, dtype=np.float64)
            se_buffer = np.zeros(n_genes, dtype=np.float64)
            lfc_work_buffer = np.zeros(n_genes, dtype=np.float64)  # Work buffer for in-place LFC

            def compute_perturbation(label: str, slot: int) -> None:
                idx = group_index[label]
                n_cells = counts[idx]
                if n_cells == 0:
                    raise ValueError(f"Perturbation '{label}' contains no cells")

                np.divide(sums[idx], n_cells, out=mean_buffer)
                np.copyto(var_buffer, sumsq[idx])
                if n_cells > 1:
                    np.subtract(var_buffer, np.square(sums[idx]) / n_cells, out=var_buffer)
                    np.divide(var_buffer, n_cells - 1, out=var_buffer)
                else:
                    var_buffer.fill(0)
                np.clip(var_buffer, a_min=0, a_max=None, out=var_buffer)

                np.subtract(mean_buffer, control_mean, out=effect_buffer[slot])
                effect_f32 = effect_buffer[slot].astype(np.float32, copy=False)

                np.divide(var_buffer, n_cells, out=se_buffer)
                np.add(se_buffer, control_var / control_n, out=se_buffer)
                np.sqrt(se_buffer, out=se_buffer)

                total_expr = expr_counts[idx] + expr_counts[control_idx]
                valid = (se_buffer > 0) & (total_expr >= min_cells_expressed)

                stat_buffer[slot].fill(0)
                pval_buffer[slot].fill(1)
                stat_buffer[slot][valid] = effect_f32[valid] / se_buffer[valid]
                pval_buffer[slot][valid] = 2 * norm.sf(np.abs(stat_buffer[slot][valid]))

                np.divide(
                    expr_counts[idx],
                    n_cells,
                    out=pts_buffer[slot],
                    where=n_cells > 0,
                    casting="unsafe",
                )

                order_buffer[slot] = np.argsort(-np.abs(stat_buffer[slot]))
                # Scanpy-compatible log2 fold change: log2((expm1(mean_group) + eps) / (expm1(mean_rest) + eps))
                # Use in-place operations to minimize temporary allocations
                np.expm1(mean_buffer, out=lfc_work_buffer)
                np.add(lfc_work_buffer, 1e-9, out=lfc_work_buffer)
                np.divide(lfc_work_buffer, control_mean_expm1, out=lfc_work_buffer)
                np.log2(lfc_work_buffer, out=lfc_work_buffer)
                lfc_buffer[slot] = lfc_work_buffer.astype(np.float32)

            if n_groups > 0:
                for batch_start in range(0, n_groups, batch_size):
                    batch_labels = candidates[batch_start : batch_start + batch_size]
                    for local_idx, label in enumerate(batch_labels):
                        compute_perturbation(label, local_idx)

                    for local_idx, label in enumerate(batch_labels):
                        global_idx = candidate_indices[label]
                        effect_memmap[global_idx] = effect_buffer[local_idx]
                        stat_memmap[global_idx] = stat_buffer[local_idx]
                        pval_memmap[global_idx] = pval_buffer[local_idx]
                        lfc_memmap[global_idx] = lfc_buffer[local_idx]
                        pts_memmap[global_idx] = pts_buffer[local_idx]
                        order_memmap[global_idx] = order_buffer[local_idx]

                        ds_scores[global_idx] = stat_buffer[local_idx]
                        ds_pvals[global_idx] = pval_buffer[local_idx]
                        ds_lfc[global_idx] = lfc_buffer[local_idx]
                        ds_pts[global_idx] = pts_buffer[local_idx]
                        ds_order[global_idx] = order_buffer[local_idx]

            pvalue_adj_memmap = np.memmap(
                tmp_path / "pvalues_adj.dat", mode="w+", dtype=np.float64, shape=shape
            )
            _adjust_pvalue_matrix(pval_memmap, method="benjamini-hochberg", out=pvalue_adj_memmap)
            ds_pvals_adj[:] = pvalue_adj_memmap

        pts_rest_arr = np.asarray(pts_rest_memmap)
        result = RankGenesGroupsResult(
            genes=gene_symbols,
            groups=candidates,
            statistics=np.asarray(stat_memmap),
            pvalues=np.asarray(pval_memmap),
            pvalues_adj=np.asarray(pvalue_adj_memmap),
            logfoldchanges=np.asarray(lfc_memmap),
            effect_size=np.asarray(effect_memmap),
            u_statistics=np.zeros(shape, dtype=np.float32),
            pts=np.asarray(pts_memmap),
            pts_rest=pts_rest_arr,
            order=np.asarray(order_memmap),
            groupby=perturbation_column,
            method="t_test",
            control_label=control_label,
            tie_correct=False,
            pvalue_correction="benjamini-hochberg",
            result=None,
        )

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
    fit_method: Literal["independent", "joint"] = "independent",
    share_dispersion: bool | None = None,
    dispersion: float | None = None,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    optimization_method: Literal["irls", "lbfgsb"] = "lbfgsb",
    max_iter: int = 25,
    tol: float = 1e-6,
    poisson_init_iter: int = 5,
    min_cells_expressed: int = 0,
    min_total_count: float = 1.0,
    chunk_size: int = 256,
    irls_batch_size: int | None = 128,
    size_factors: ArrayLike | None = None,
    size_factor_method: Literal["sparse", "deseq2"] = "sparse",
    cook_filter: bool = False,
    use_map_dispersion: bool = True,
    shrink_dispersion: bool = True,
    shrink_logfc: bool = False,
    lfc_shrinkage_type: Literal["normal", "apeglm", "none"] = "none",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
    use_sparse: bool = True,
    use_numba: bool = True,
    joint_optimizer: Literal["irls", "lbfgsb"] = "lbfgsb",
    scale_size_factors: bool = True,
    lfc_base: Literal["log2", "ln"] = "log2",
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
    fit_method
        Method for fitting the GLM model:
        - "independent": Estimate intercept, covariates, and dispersion independently 
          for each perturbation comparison (control vs one perturbation). This method
          achieves near-perfect rank correlation (ρ > 0.99) with PyDESeq2 results
          and is recommended for most use cases.
        - "joint": Fit a full model with all perturbations simultaneously using all
          cells. The intercept represents the control baseline, and perturbation
          effects are estimated as log-fold changes from control. This approach
          may produce less stable estimates for genes with sparse expression in
          some groups. For best results with joint mode, enable LFC shrinkage
          (lfc_shrinkage_type="normal") which achieves ρ > 0.97 with PyDESeq2.
    share_dispersion
        If True, estimate dispersion once using all cells (PyDESeq2 style), then 
        use the same dispersion values for all Wald tests. If False, estimate
        dispersion separately for each perturbation comparison. If None (default),
        automatically set to True for fit_method="joint" and False for "independent".
    dispersion
        Fixed dispersion parameter for negative binomial. If None, estimates per gene.
    dispersion_method
        Method for estimating dispersion when ``dispersion`` is None:
        - "moments": Method-of-moments (fast but less accurate)
        - "cox-reid": Cox-Reid adjusted profile likelihood (slower but more
          accurate, similar to DESeq2). This is the default.
    optimization_method
        Method for coefficient optimization:
        - "lbfgsb": L-BFGS-B optimization (PyDESeq2 style, default). Directly
          optimizes the negative binomial log-likelihood.
        - "irls": Iteratively Reweighted Least Squares (Fisher scoring). The
          classic GLM fitting approach.
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
        Number of genes to process per chunk (memory vs. speed tradeoff). Smaller
        values stream more, reducing peak memory at the cost of additional I/O.
    irls_batch_size
        Maximum number of genes to densify per IRLS step. Keep this small to
        limit per-iteration memory when working with large sparse matrices. Set
        to ``None`` to process each chunk without additional batching.
    size_factors
        Optional array of per-cell size factors. If None, computes size factors
        using the method specified by `size_factor_method`.
    size_factor_method
        Method for computing size factors when ``size_factors`` is None:
        - "sparse": Sparse-aware median-of-ratios (default). Computes geometric
          means using only non-zero values, suitable for sparse single-cell data.
        - "deseq2": Classic DESeq2/PyDESeq2 style. Uses only genes expressed in
          ALL cells (typically ~50-100 genes). Provides better numerical alignment
          with PyDESeq2 results but may be less robust for very sparse data.
    cook_filter
        Whether to apply Cook's distance outlier filtering when available.
    shrink_dispersion
        If True, fit a mean-dispersion trend and shrink gene-wise dispersions
        toward the trend using an empirical Bayes prior.
    shrink_logfc
        Deprecated. Use `lfc_shrinkage_type` instead.
        If True and lfc_shrinkage_type is "none", sets lfc_shrinkage_type to "normal".
    lfc_shrinkage_type
        Type of log-fold change shrinkage to apply:
        - "none": No shrinkage (PyDESeq2 default, recommended for most cases)
        - "normal": Normal-normal empirical Bayes shrinkage
        - "apeglm": Adaptive shrinkage that preserves strong signals
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
    use_sparse
        If True, use sparse matrix operations for perturbation accumulation
        in joint fitting mode when no covariates are present. Falls back to
        Numba otherwise. Only applies when fit_method="joint" and joint_optimizer="irls".
    use_numba
        If True, use Numba-accelerated kernels for inner loops in joint fitting
        mode. Provides ~5-10× speedup over Python loops. Only applies when
        fit_method="joint" and joint_optimizer="irls".
    joint_optimizer
        Optimizer for joint fitting mode:
        - "lbfgsb": L-BFGS-B optimization (faster, ~10× speedup, recommended)
        - "irls": Iteratively Reweighted Least Squares (original, slower)
        Only applies when fit_method="joint".
    scale_size_factors
        If True (default), scale size factors so their geometric mean equals 1.
        This is the standard DESeq2/crispyx behavior. If False, use raw 
        median-of-ratios size factors without rescaling, which matches 
        PyDESeq2's default behavior and can improve numerical alignment.
    lfc_base
        Log base for fold change output:
        - "log2" (default): Output log2 fold change, matching PyDESeq2/edgeR.
        - "ln": Output natural log fold change (raw GLM coefficients).
        Standard error is also converted to match the selected log base.
        Wald statistics remain unchanged since both LFC and SE are scaled equally.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results in Scanpy-compatible format. Access results
        via dict-like interface: `result[label].effect_size`, `result[label].pvalue`, etc.
        The h5ad file path is available at `result.result_path`.
    """
    # Handle deprecated shrink_logfc parameter
    if shrink_logfc and lfc_shrinkage_type == "none":
        lfc_shrinkage_type = "normal"
    
    # Set share_dispersion default based on fit_method
    if share_dispersion is None:
        share_dispersion = (fit_method == "joint")

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
        control_label = resolve_control_label(list(labels), control_label)
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

    n_cells_total = obs_df.shape[0]
    if size_factors is None:
        if size_factor_method == "deseq2":
            size_factors = _deseq2_style_size_factors(
                path, chunk_size=chunk_size, scale=scale_size_factors
            )
        else:  # "sparse" (default)
            size_factors = _median_of_ratios_size_factors(
                path, chunk_size=chunk_size, scale=scale_size_factors
            )
    else:
        size_factors = _validate_size_factors(
            size_factors, n_cells_total, scale=scale_size_factors
        )

    offset = np.log(np.clip(size_factors, 1e-8, None))

    # =========================================================================
    # Joint fitting mode: use new estimate_joint_model_streaming
    # =========================================================================
    if fit_method == "joint":
        logger.info(f"Joint fitting: Estimating full model with {joint_optimizer.upper()} optimizer...")
        backed = read_backed(path)
        try:
            if joint_optimizer == "lbfgsb":
                # Fast vectorized IRLS approach (recommended)
                joint_result = estimate_joint_model_lbfgsb(
                    backed,
                    obs_df=obs_df,
                    perturbation_labels=labels,
                    control_label=control_label,
                    covariate_columns=covariates,
                    size_factors=size_factors,
                    chunk_size=chunk_size,
                    max_iter=max_iter,
                    tol=tol,
                    dispersion_method=dispersion_method,
                    shrink_dispersion=shrink_dispersion,
                    per_comparison_dispersion=True,  # Match independent mode accuracy
                    use_map_dispersion=use_map_dispersion,
                    cook_filter=cook_filter,
                    lfc_shrinkage_type=lfc_shrinkage_type,
                    n_jobs=n_jobs,
                )
            else:
                # Original IRLS approach
                joint_result = estimate_joint_model_streaming(
                    backed,
                    obs_df=obs_df,
                    perturbation_labels=labels,
                    control_label=control_label,
                    covariate_columns=covariates,
                    size_factors=size_factors,
                    chunk_size=chunk_size,
                    poisson_iter=poisson_init_iter,
                    nb_iter=max_iter,
                    tol=tol,
                    dispersion_method=dispersion_method,
                    shrink_dispersion=shrink_dispersion,
                    use_sparse=use_sparse,
                    use_numba=use_numba,
                )
        finally:
            backed.file.close()
        
        n_pert = len(joint_result.perturbation_labels)
        logger.info(f"  Estimated effects for {n_pert} perturbations and {n_genes} genes")
        
        # Build label-to-index mapping for joint result
        joint_label_to_idx = {
            str(label): i for i, label in enumerate(joint_result.perturbation_labels)
        }
        
        # Filter candidates to those in joint result and in original candidates list
        n_groups = len(candidates)
        
        # Prepare output arrays
        effect_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        statistic_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        pvalue_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        logfc_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        logfc_raw_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        se_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        pts_matrix = np.zeros((n_groups, n_genes), dtype=np.float32)
        pts_rest_matrix = np.zeros((n_groups, n_genes), dtype=np.float32)
        dispersion_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        dispersion_raw_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        dispersion_trend_matrix = np.full((n_groups, n_genes), np.nan, dtype=np.float64)
        mean_matrix = np.zeros((n_groups, n_genes), dtype=np.float64)
        iter_matrix = np.zeros((n_groups, n_genes), dtype=np.int32)
        convergence_matrix = np.zeros((n_groups, n_genes), dtype=bool)
        
        # Compute pts for each perturbation group
        backed = read_backed(path)
        try:
            control_matrix = backed.X[control_mask, :]
            if sp.issparse(control_matrix):
                control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
            else:
                control_expr_counts = np.sum(np.asarray(control_matrix) > 0, axis=0)
            
            pts_rest_shared = np.divide(
                control_expr_counts,
                control_n,
                out=np.zeros(n_genes, dtype=np.float32),
                where=control_n > 0,
            )
            
            for group_idx, label in enumerate(candidates):
                if label not in joint_label_to_idx:
                    continue
                joint_idx = joint_label_to_idx[label]
                group_mask = labels == label
                group_n = int(group_mask.sum())
                
                group_matrix = backed.X[group_mask, :]
                if sp.issparse(group_matrix):
                    group_expr_counts = np.asarray(group_matrix.getnnz(axis=0)).ravel()
                else:
                    group_expr_counts = np.sum(np.asarray(group_matrix) > 0, axis=0)
                
                pts = np.divide(
                    group_expr_counts,
                    group_n,
                    out=np.zeros(n_genes, dtype=np.float32),
                    where=group_n > 0,
                )
                pts_matrix[group_idx, :] = pts
                pts_rest_matrix[group_idx, :] = pts_rest_shared
                
                # Get coefficients and SEs from joint result
                coef = joint_result.beta_perturbation[joint_idx, :]
                se = joint_result.se_perturbation[joint_idx, :]
                
                # Compute Wald statistics and p-values
                valid = np.isfinite(coef) & np.isfinite(se) & (se > 0)
                statistic = np.zeros(n_genes, dtype=np.float64)
                pvalue = np.ones(n_genes, dtype=np.float64)
                statistic[valid] = coef[valid] / se[valid]
                pvalue[valid] = 2.0 * norm.sf(np.abs(statistic[valid]))
                
                # Store results
                effect_matrix[group_idx, :] = coef
                statistic_matrix[group_idx, :] = statistic
                pvalue_matrix[group_idx, :] = pvalue
                logfc_matrix[group_idx, :] = coef
                logfc_raw_matrix[group_idx, :] = coef
                se_matrix[group_idx, :] = se
                dispersion_matrix[group_idx, :] = joint_result.dispersion
                dispersion_raw_matrix[group_idx, :] = joint_result.dispersion
                dispersion_trend_matrix[group_idx, :] = joint_result.dispersion
                convergence_matrix[group_idx, :] = joint_result.converged
                iter_matrix[group_idx, :] = joint_result.n_iter
        finally:
            backed.file.close()
        
        # Apply LFC shrinkage if requested
        if lfc_shrinkage_type != "none":
            for group_idx in range(n_groups):
                prior_var = None
                finite_se = se_matrix[group_idx][np.isfinite(se_matrix[group_idx])]
                if finite_se.size:
                    prior_var = float(np.nanmedian(finite_se ** 2))
                shrunk_lfc = shrink_log_foldchange(
                    logfc_matrix[group_idx], se_matrix[group_idx], 
                    prior_var=prior_var, shrinkage_type=lfc_shrinkage_type
                )
                logfc_matrix[group_idx] = shrunk_lfc
                effect_matrix[group_idx] = shrunk_lfc
        
        # Adjust p-values
        pvalue_adj_matrix = np.ones_like(pvalue_matrix)
        _adjust_pvalue_matrix(pvalue_matrix, corr_method, out=pvalue_adj_matrix)
        
        # Create order matrix
        statistic_for_order = np.where(
            np.isfinite(statistic_matrix), np.abs(statistic_matrix), -np.inf
        )
        order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")
        
        # Save results
        gene_symbols = pd.Index(gene_symbols).astype(str)
        obs_index = pd.Index(candidates, name="perturbation").astype(str)
        obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
        var = pd.DataFrame(index=gene_symbols)

        # Convert from natural log to log2 if requested (PyDESeq2/edgeR convention)
        if lfc_base == "log2":
            ln2 = np.log(2)
            effect_matrix = effect_matrix / ln2
            logfc_matrix = logfc_matrix / ln2
            logfc_raw_matrix = logfc_raw_matrix / ln2
            se_matrix = se_matrix / ln2

        adata = ad.AnnData(effect_matrix, obs=obs, var=var)
        adata.layers["z_score"] = statistic_matrix
        adata.layers["pvalue"] = pvalue_matrix
        adata.layers["pvalue_adj"] = pvalue_adj_matrix
        adata.layers["logfoldchange"] = logfc_matrix
        adata.layers["logfoldchange_raw"] = logfc_raw_matrix
        adata.layers["standard_error"] = se_matrix
        adata.layers["dispersion"] = dispersion_matrix
        adata.layers["dispersion_raw"] = dispersion_raw_matrix
        adata.layers["dispersion_trend"] = dispersion_trend_matrix
        adata.layers["converged"] = convergence_matrix.astype(np.float32)
        adata.layers["iterations"] = iter_matrix.astype(np.float32)
        adata.layers["pts"] = pts_matrix
        adata.layers["pts_rest"] = pts_rest_matrix
        adata.uns["method"] = "nb_glm"
        adata.uns["fit_method"] = "joint"
        adata.uns["control_label"] = control_label
        adata.uns["covariates"] = covariates
        adata.uns["size_factors"] = size_factors

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
        )
        result.result = AnnData(output_path)
        return result

    # =========================================================================
    # Independent fitting mode: original per-perturbation approach
    # =========================================================================
    # Stage 1: Joint estimation (if fit_method="joint") - LEGACY, kept for reference
    # This code path is no longer used when fit_method="joint"
    beta_cov = None
    beta_intercept = None
    global_dispersion = None

    # -------------------------------------------------------------------------
    # Worker function for parallel fitting of a single perturbation group
    # -------------------------------------------------------------------------
    def _fit_perturbation_worker(
        group_idx: int,
        label: str,
        path: str | Path,
        labels: np.ndarray,
        control_mask: np.ndarray,
        control_matrix: np.ndarray | sp.csr_matrix,
        control_expr_counts: np.ndarray,
        control_n: int,
        obs_df: pd.DataFrame,
        covariates: list[str],
        size_factors: np.ndarray,
        offset: np.ndarray,
        n_genes: int,
        min_cells_expressed: int,
        min_total_count: float,
        max_iter: int,
        tol: float,
        poisson_init_iter: int,
        dispersion_method: str,
        global_dispersion: np.ndarray | None,
        shrink_dispersion: bool,
        use_map_dispersion: bool,
        lfc_shrinkage_type: str,
        pts_rest_shared: np.ndarray,
    ) -> dict:
        """Fit NB-GLM for a single perturbation group and return results."""
        group_mask = labels == label
        subset_mask = control_mask | group_mask
        subset_obs = obs_df.iloc[subset_mask]
        indicator = group_mask[subset_mask].astype(np.float64)
        subset_size_factors = np.asarray(size_factors)[subset_mask]
        
        # Build design matrix
        design, design_columns = build_design_matrix(
            subset_obs,
            covariate_columns=covariates,
            perturbation_indicator=indicator,
            intercept=True,
        )
        perturbation_column_index = design_columns.index("perturbation")
        
        control_subset_mask = control_mask[subset_mask]
        group_subset_mask = group_mask[subset_mask]
        group_n = int(group_subset_mask.sum())
        subset_n = int(subset_mask.sum())
        n_control = int(control_subset_mask.sum())

        # Load perturbation group cells
        backed = read_backed(path)
        try:
            group_matrix = backed.X[group_mask, :]
            if sp.issparse(group_matrix):
                group_matrix = sp.csr_matrix(group_matrix, dtype=np.float64)
            else:
                group_matrix = np.asarray(group_matrix, dtype=np.float64)
        finally:
            backed.file.close()

        # Combine control and group matrices
        if sp.issparse(control_matrix) and sp.issparse(group_matrix):
            stacked = sp.vstack([control_matrix, group_matrix])
            reorder = np.empty(subset_n, dtype=np.int32)
            reorder[np.where(control_subset_mask)[0]] = np.arange(n_control)
            reorder[np.where(group_subset_mask)[0]] = np.arange(n_control, n_control + group_n)
            subset_matrix = sp.csr_matrix(stacked[reorder, :])
        else:
            if sp.issparse(control_matrix):
                ctrl = control_matrix.toarray()
            else:
                ctrl = control_matrix
            if sp.issparse(group_matrix):
                grp = group_matrix.toarray()
            else:
                grp = group_matrix
            stacked = np.vstack([ctrl, grp])
            reorder = np.empty(subset_n, dtype=np.int32)
            reorder[np.where(control_subset_mask)[0]] = np.arange(n_control)
            reorder[np.where(group_subset_mask)[0]] = np.arange(n_control, n_control + group_n)
            subset_matrix = stacked[reorder, :]

        # Compute expression counts
        if sp.issparse(group_matrix):
            group_expr_counts = np.asarray(group_matrix.getnnz(axis=0)).ravel()
        else:
            group_expr_counts = np.sum(group_matrix > 0, axis=0)
        
        total_expr_counts = control_expr_counts + group_expr_counts
        valid_mask = total_expr_counts >= min_cells_expressed
        valid_indices = np.where(valid_mask)[0]

        # Initialize result arrays
        result = {
            "group_idx": group_idx,
            "effect": np.full(n_genes, np.nan, dtype=np.float64),
            "statistic": np.full(n_genes, np.nan, dtype=np.float64),
            "pvalue": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "se": np.full(n_genes, np.nan, dtype=np.float64),
            "pts": np.zeros(n_genes, dtype=np.float32),
            "pts_rest": np.zeros(n_genes, dtype=np.float32),
            "dispersion": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_trend": np.full(n_genes, np.nan, dtype=np.float64),
            "mean": np.zeros(n_genes, dtype=np.float64),
            "iterations": np.zeros(n_genes, dtype=np.int32),
            "converged": np.zeros(n_genes, dtype=bool),
        }

        # Compute pts
        pts = np.divide(
            group_expr_counts,
            group_n,
            out=np.zeros(n_genes, dtype=np.float32),
            where=group_n > 0,
        )
        result["pts"] = np.where(valid_mask, pts, 0.0).astype(np.float32)
        result["pts_rest"] = np.where(valid_mask, pts_rest_shared, 0.0).astype(np.float32)

        # Compute mean expression
        if sp.issparse(subset_matrix):
            normalized = subset_matrix.multiply(1.0 / subset_size_factors[:, None])
            mean_expr = np.asarray(normalized.sum(axis=0)).ravel() / subset_n
        else:
            normalized = subset_matrix / subset_size_factors[:, None]
            mean_expr = normalized.sum(axis=0) / subset_n
        result["mean"] = mean_expr

        if not np.any(valid_mask):
            return result

        # Fit valid genes
        fit_matrix = subset_matrix[:, valid_mask]
        
        batch_fitter = NBGLMBatchFitter(
            design,
            offset=offset[subset_mask],
            max_iter=max_iter,
            tol=tol,
            poisson_init_iter=poisson_init_iter,
            dispersion_method=dispersion_method,
            min_mu=0.5,
            min_total_count=min_total_count,
        )
        
        batch_result = batch_fitter.fit_batch(fit_matrix)

        # Extract results
        result["converged"][valid_indices] = batch_result.converged
        result["iterations"][valid_indices] = batch_result.n_iter
        result["dispersion_raw"][valid_indices] = batch_result.dispersion

        coefs = batch_result.coef[:, perturbation_column_index]
        ses = batch_result.se[:, perturbation_column_index]

        valid_results = (
            batch_result.converged
            & np.isfinite(coefs)
            & np.isfinite(ses)
            & (ses > 0)
        )
        
        for local_idx, gene_idx in enumerate(valid_indices):
            if not valid_results[local_idx]:
                continue
            coef = coefs[local_idx]
            se = ses[local_idx]
            statistic = coef / se
            pvalue = float(2.0 * norm.sf(abs(statistic)))
            result["statistic"][gene_idx] = statistic
            result["pvalue"][gene_idx] = pvalue
            result["logfc"][gene_idx] = coef
            result["se"][gene_idx] = se

        # Handle dispersion
        if global_dispersion is not None:
            result["dispersion_raw"][:] = global_dispersion
            result["dispersion"][:] = global_dispersion
            result["dispersion_trend"][:] = global_dispersion
        elif shrink_dispersion:
            trend = fit_dispersion_trend(result["mean"], result["dispersion_raw"])
            result["dispersion_trend"] = trend
            if use_map_dispersion:
                # Use PyDESeq2-style MAP estimation with proper fitted values
                # First, compute mu from the fitted model coefficients
                if sp.issparse(subset_matrix):
                    Y = np.asarray(subset_matrix.todense(), dtype=np.float64)
                else:
                    Y = np.asarray(subset_matrix, dtype=np.float64)
                
                # Get fitted values from the model: mu = exp(X @ beta + offset)
                # batch_result.coef has shape (n_valid_genes, n_design_cols)
                # design has shape (n_cells, n_design_cols)
                n_subset = subset_n
                mu = np.zeros((n_subset, n_genes), dtype=np.float64)
                
                # For valid genes, compute mu from fitted coefficients
                for local_idx, gene_idx in enumerate(valid_indices):
                    if batch_result.converged[local_idx]:
                        # eta = X @ beta + offset
                        eta = design @ batch_result.coef[local_idx, :] + offset[subset_mask]
                        mu[:, gene_idx] = np.exp(np.clip(eta, -30, 30))
                
                # For invalid genes, use normalized counts as fallback
                invalid_genes = ~np.isin(np.arange(n_genes), valid_indices[batch_result.converged])
                if np.any(invalid_genes):
                    mu[:, invalid_genes] = Y[:, invalid_genes] / subset_size_factors[:, None]
                
                mu = np.maximum(mu, 1e-10)
                # Use PyDESeq2-style bounds: max(10, n_cells)
                max_disp = max(10.0, float(n_subset))
                result["dispersion"] = estimate_dispersion_map(
                    Y, mu, trend, max_disp=max_disp
                )
            else:
                result["dispersion"] = shrink_dispersions(result["dispersion_raw"], trend)
        else:
            result["dispersion_trend"] = result["dispersion_raw"].copy()
            result["dispersion"] = result["dispersion_raw"].copy()

        result["logfc_raw"] = result["logfc"].copy()
        if lfc_shrinkage_type != "none":
            finite_se = result["se"][np.isfinite(result["se"])]
            prior_var = float(np.nanmedian(finite_se ** 2)) if finite_se.size else None
            shrunk_lfc = shrink_log_foldchange(
                result["logfc"], result["se"],
                prior_var=prior_var, shrinkage_type=lfc_shrinkage_type
            )
            result["logfc"] = shrunk_lfc
            result["effect"] = shrunk_lfc
        else:
            result["effect"] = result["logfc"].copy()

        return result
    # -------------------------------------------------------------------------

    n_groups = len(candidates)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        def _create_memmap(
            name: str, dtype: np.dtype, *, fill: float | int | bool | None = np.nan
        ) -> np.memmap:
            mmap = np.memmap(
                tmp_path / f"{name}.dat",
                mode="w+",
                dtype=dtype,
                shape=(n_groups, n_genes),
            )
            if fill is None:
                return mmap
            if isinstance(fill, float) and np.isnan(fill):
                mmap.fill(np.nan)
            else:
                mmap.fill(fill)
            return mmap

        effect_memmap = _create_memmap("effect", np.float64)
        statistic_memmap = _create_memmap("statistic", np.float64)
        pvalue_memmap = _create_memmap("pvalue", np.float64)
        logfc_memmap = _create_memmap("logfoldchange", np.float64)
        logfc_raw_memmap = _create_memmap("logfoldchange_raw", np.float64)
        se_memmap = _create_memmap("standard_error", np.float64)
        pts_memmap = _create_memmap("pts", np.float32, fill=0.0)
        pts_rest_memmap = _create_memmap("pts_rest", np.float32, fill=0.0)
        dispersion_memmap = _create_memmap("dispersion", np.float64)
        dispersion_raw_memmap = _create_memmap("dispersion_raw", np.float64)
        dispersion_trend_memmap = _create_memmap("dispersion_trend", np.float64)
        mean_memmap = _create_memmap("mean", np.float64, fill=0.0)
        iter_memmap = _create_memmap("iterations", np.int32, fill=0)
        convergence_memmap = _create_memmap("converged", np.bool_, fill=False)

        # Load control cells matrix once (all genes) - this is small enough to fit in memory
        # since control is typically a subset of all cells
        backed = read_backed(path)
        try:
            control_matrix = backed.X[control_mask, :]
            if sp.issparse(control_matrix):
                control_matrix = sp.csr_matrix(control_matrix, dtype=np.float64)
            else:
                control_matrix = np.asarray(control_matrix, dtype=np.float64)
        finally:
            backed.file.close()
        
        # Pre-compute control expression counts
        if sp.issparse(control_matrix):
            control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
        else:
            control_expr_counts = np.sum(control_matrix > 0, axis=0)
        
        # Compute pts_rest once (same for all perturbations)
        pts_rest_shared = np.divide(
            control_expr_counts,
            control_n,
            out=np.zeros(n_genes, dtype=np.float32),
            where=control_n > 0,
        )

        # =====================================================================
        # Parallel fitting of perturbation groups
        # =====================================================================
        # Determine number of parallel workers
        max_workers = os.cpu_count() or 1
        if n_jobs is None or n_jobs == 0:
            effective_n_jobs = max_workers
        elif n_jobs == -1:
            effective_n_jobs = max_workers
        elif n_jobs < 0:
            effective_n_jobs = max(1, max_workers + n_jobs + 1)
        else:
            effective_n_jobs = min(n_jobs, max_workers)
        effective_n_jobs = max(1, effective_n_jobs)
        
        # Run parallel fitting
        logger.info(f"Fitting {n_groups} perturbations with {effective_n_jobs} workers...")
        
        results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
            delayed(_fit_perturbation_worker)(
                group_idx=group_idx,
                label=label,
                path=path,
                labels=labels,
                control_mask=control_mask,
                control_matrix=control_matrix,
                control_expr_counts=control_expr_counts,
                control_n=control_n,
                obs_df=obs_df,
                covariates=covariates,
                size_factors=size_factors,
                offset=offset,
                n_genes=n_genes,
                min_cells_expressed=min_cells_expressed,
                min_total_count=min_total_count,
                max_iter=max_iter,
                tol=tol,
                poisson_init_iter=poisson_init_iter,
                dispersion_method=dispersion_method,
                global_dispersion=global_dispersion,
                shrink_dispersion=shrink_dispersion,
                use_map_dispersion=use_map_dispersion,
                lfc_shrinkage_type=lfc_shrinkage_type,
                pts_rest_shared=pts_rest_shared,
            )
            for group_idx, label in enumerate(candidates)
        )
        
        # Write results to memmaps
        for res in results:
            idx = res["group_idx"]
            effect_memmap[idx, :] = res["effect"]
            statistic_memmap[idx, :] = res["statistic"]
            pvalue_memmap[idx, :] = res["pvalue"]
            logfc_memmap[idx, :] = res["logfc"]
            logfc_raw_memmap[idx, :] = res["logfc_raw"]
            se_memmap[idx, :] = res["se"]
            pts_memmap[idx, :] = res["pts"]
            pts_rest_memmap[idx, :] = res["pts_rest"]
            dispersion_memmap[idx, :] = res["dispersion"]
            dispersion_raw_memmap[idx, :] = res["dispersion_raw"]
            dispersion_trend_memmap[idx, :] = res["dispersion_trend"]
            mean_memmap[idx, :] = res["mean"]
            iter_memmap[idx, :] = res["iterations"]
            convergence_memmap[idx, :] = res["converged"]

        pvalue_adj_memmap = np.memmap(
            tmp_path / "pvalue_adj.dat", mode="w+", dtype=np.float64, shape=(n_groups, n_genes)
        )
        _adjust_pvalue_matrix(pvalue_memmap, corr_method, out=pvalue_adj_memmap)

        gene_symbols = pd.Index(gene_symbols).astype(str)
        statistic_for_order = np.where(
            np.isfinite(statistic_memmap), np.abs(statistic_memmap), -np.inf
        )
        order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")

        effect_matrix = np.array(effect_memmap)
        statistic_matrix = np.array(statistic_memmap)
        pvalue_matrix = np.array(pvalue_memmap)
        pvalue_adj_matrix = np.array(pvalue_adj_memmap)
        logfc_matrix = np.array(logfc_memmap)
        logfc_raw_matrix = np.array(logfc_raw_memmap)
        se_matrix = np.array(se_memmap)
        dispersion_matrix = np.array(dispersion_memmap)
        dispersion_raw_matrix = np.array(dispersion_raw_memmap)
        dispersion_trend_matrix = np.array(dispersion_trend_memmap)
        mean_matrix = np.array(mean_memmap)
        iter_matrix = np.array(iter_memmap)
        convergence_matrix = np.array(convergence_memmap)
        pts_matrix = np.array(pts_memmap, dtype=np.float32)
        pts_rest_matrix = np.array(pts_rest_memmap, dtype=np.float32)

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)

    # Convert from natural log to log2 if requested (PyDESeq2/edgeR convention)
    if lfc_base == "log2":
        ln2 = np.log(2)
        effect_matrix = effect_matrix / ln2
        logfc_matrix = logfc_matrix / ln2
        logfc_raw_matrix = logfc_raw_matrix / ln2
        se_matrix = se_matrix / ln2

    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    adata.layers["z_score"] = statistic_matrix
    adata.layers["pvalue"] = pvalue_matrix
    adata.layers["pvalue_adj"] = pvalue_adj_matrix
    adata.layers["logfoldchange"] = logfc_matrix
    adata.layers["logfoldchange_raw"] = logfc_raw_matrix
    adata.layers["standard_error"] = se_matrix
    adata.layers["dispersion"] = dispersion_matrix
    adata.layers["dispersion_raw"] = dispersion_raw_matrix
    adata.layers["dispersion_trend"] = dispersion_trend_matrix
    adata.layers["converged"] = convergence_matrix.astype(np.float32)
    adata.layers["iterations"] = iter_matrix.astype(np.float32)
    adata.layers["pts"] = pts_matrix
    adata.layers["pts_rest"] = pts_rest_matrix
    adata.uns["method"] = "nb_glm"
    adata.uns["control_label"] = control_label
    adata.uns["covariates"] = covariates
    adata.uns["size_factors"] = size_factors

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
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = peak_rss_kb / 1024.0
    logger.info(
        "nb_glm_test peak RSS: %.2f MB (chunk_size=%d, irls_batch_size=%s)",
        peak_rss_mb,
        chunk_size,
        "auto" if irls_batch_size is None else irls_batch_size,
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
    """Perform a Wilcoxon rank-sum (Mann-Whitney U) test for each gene.

    Input data **must already be library-size normalised and log-transformed**.
    The function operates directly on the provided matrix without additional
    preprocessing. As a safeguard, the first sparse chunk is inspected and a
    warning is emitted if the data appear to be raw counts (integer or
    count-like floats), encouraging explicit preprocessing upstream.
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
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        def _create_memmap(name: str, dtype: np.dtype, *, fill: float | int = 0):
            path = tmpdir_path / f"{name}.dat"
            mmap = np.memmap(path, dtype=dtype, mode="w+", shape=(n_groups, n_genes))
            if fill != 0:
                mmap[:] = fill
            else:
                mmap.fill(0)
            return mmap

        effect_matrix = _create_memmap("effect", np.float64)
        u_matrix = _create_memmap("u_stat", np.float64)
        pvalue_matrix = _create_memmap("pvalue", np.float64, fill=1.0)
        z_matrix = _create_memmap("z_score", np.float64)
        lfc_matrix = _create_memmap("logfoldchange", np.float64)
        pts_matrix = _create_memmap("pts", np.float32)
        pts_rest_matrix = _create_memmap("pts_rest", np.float32)
        order_matrix = np.memmap(
            tmpdir_path / "order.dat", dtype=np.int64, mode="w+", shape=(n_groups, n_genes)
        )

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

            dtype_checked = False

            def _warn_if_count_like(chunk: sp.spmatrix) -> bool:
                if np.issubdtype(chunk.dtype, np.integer):
                    logger.warning(
                        "Detected integer count data in wilcoxon_test; input should be normalized/log-transformed. "
                        "For reproducibility, please preprocess explicitly upstream."
                    )
                    return True
                if np.issubdtype(chunk.dtype, np.floating):
                    non_zero = chunk.data[chunk.data > 0]
                    is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
                    if is_count_like:
                        logger.warning(
                            "Detected count-like floating point values in wilcoxon_test; input should be normalized/log-transformed. "
                            "Please ensure preprocessing is applied upstream for consistent results."
                        )
                    return bool(is_count_like)
                return False

            for slc, block in iter_matrix_chunks(
                backed, axis=1, chunk_size=chunk_size, convert_to_dense=False
            ):
                if not dtype_checked:
                    if not sp.issparse(block):
                        raise ValueError(
                            "wilcoxon_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                        )
                    _warn_if_count_like(block)
                    dtype_checked = True

                csr_block = sp.csr_matrix(block, dtype=np.float64)

                control_values = csr_block[control_mask, :]
                control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()
                control_mean = (
                    np.asarray(control_values.mean(axis=0)).ravel()
                    if control_values.nnz
                    else np.zeros(csr_block.shape[1], dtype=np.float64)
                )
                # Precompute control term for LFC calculation once per chunk
                control_mean_expm1 = np.expm1(control_mean) + 1e-9
                # Work buffer for in-place LFC calculation
                lfc_work_buffer = np.empty_like(control_mean)
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
                    group_values = csr_block[mask, :]
                    group_expr = np.asarray(group_values.getnnz(axis=0)).ravel()
                    group_mean = (
                        np.asarray(group_values.mean(axis=0)).ravel()
                        if group_values.nnz
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
                        selected_control = control_values[:, valid_cols].toarray()
                        selected_group = group_values[:, valid_cols].toarray()
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
                    # Scanpy-compatible log2 fold change: log2((expm1(mean_group) + eps) / (expm1(mean_rest) + eps))
                    # Use in-place operations to minimize temporary allocations
                    np.expm1(group_mean, out=lfc_work_buffer)
                    np.add(lfc_work_buffer, 1e-9, out=lfc_work_buffer)
                    np.divide(lfc_work_buffer, control_mean_expm1, out=lfc_work_buffer)
                    np.log2(lfc_work_buffer, out=lfc_work_buffer)
                    log_fc = np.where(valid, lfc_work_buffer, 0.0)
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
        pvalue_adj_matrix = _create_memmap("pvalue_adj", np.float64)
        _adjust_pvalue_matrix(pvalue_matrix, corr_method, out=pvalue_adj_matrix)

        for idx in range(n_groups):
            order_matrix[idx] = np.argsort(-z_matrix[idx], kind="mergesort")

        result = RankGenesGroupsResult(
            genes=gene_symbols,
            groups=candidates,
            statistics=np.array(z_matrix),
            pvalues=np.array(pvalue_matrix),
            pvalues_adj=np.array(pvalue_adj_matrix),
            logfoldchanges=np.array(lfc_matrix),
            effect_size=np.array(effect_matrix),
            u_statistics=np.array(u_matrix),
            pts=np.array(pts_matrix, dtype=np.float32),
            pts_rest=np.array(pts_rest_matrix, dtype=np.float32),
            order=np.array(order_matrix),
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


def shrink_lfc(
    path: str | Path,
    *,
    shrinkage_type: Literal["normal", "apeglm"] = "apeglm",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> RankGenesGroupsResult:
    """Apply log-fold change shrinkage to existing NB-GLM results.
    
    This function applies post-hoc LFC shrinkage to NB-GLM results that were
    computed without shrinkage. It uses the `logfoldchange_raw` and 
    `standard_error` layers stored in the result h5ad file.
    
    This enables separating the base NB-GLM fitting from shrinkage for:
    - Benchmarking: measure base fitting and shrinkage times separately
    - Flexibility: try different shrinkage types on the same base results
    - Memory efficiency: avoid re-running the full GLM
    
    Parameters
    ----------
    path
        Path to an h5ad file containing NB-GLM results from `nb_glm_test`.
        Must have `logfoldchange_raw` and `standard_error` layers.
    shrinkage_type
        Type of LFC shrinkage to apply:
        - "normal": Normal-normal empirical Bayes shrinkage (fast)
        - "apeglm": Adaptive shrinkage that preserves strong signals (recommended)
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, appends "_shrunk" to input name.
    
    Returns
    -------
    RankGenesGroupsResult
        Updated differential expression results with shrunken LFCs.
        The result h5ad has:
        - `logfoldchange`: shrunken LFC values
        - `logfoldchange_raw`: original MLE LFC values (preserved)
        - `X`: updated to shrunken LFC (effect_size)
    
    Examples
    --------
    >>> # First run NB-GLM without shrinkage
    >>> result = crispyx.de.nb_glm_test(
    ...     "data.h5ad",
    ...     perturbation_column="perturbation",
    ...     lfc_shrinkage_type="none",  # No shrinkage during fitting
    ... )
    >>> # Then apply shrinkage as a separate step
    >>> shrunk_result = crispyx.de.shrink_lfc(
    ...     result.result_path,
    ...     shrinkage_type="apeglm",
    ... )
    """
    path = Path(path)
    
    # Load the NB-GLM result
    adata = ad.read_h5ad(path)
    
    # Validate that this is an NB-GLM result with required layers
    if "logfoldchange_raw" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' does not have 'logfoldchange_raw' layer. "
            "This function requires NB-GLM results from nb_glm_test. "
            "Ensure the NB-GLM was run with a version that stores raw LFCs."
        )
    if "standard_error" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' does not have 'standard_error' layer. "
            "This function requires NB-GLM results with standard errors."
        )
    
    # Get raw LFC and SE
    raw_lfc = adata.layers["logfoldchange_raw"]
    se = adata.layers["standard_error"]
    
    n_groups, n_genes = raw_lfc.shape
    shrunk_lfc = np.zeros_like(raw_lfc)
    
    # Apply shrinkage per perturbation group
    for group_idx in range(n_groups):
        prior_var = None
        finite_se = se[group_idx][np.isfinite(se[group_idx])]
        if finite_se.size:
            prior_var = float(np.nanmedian(finite_se ** 2))
        
        shrunk_lfc[group_idx] = shrink_log_foldchange(
            raw_lfc[group_idx], 
            se[group_idx], 
            prior_var=prior_var, 
            shrinkage_type=shrinkage_type,
        )
    
    # Update layers
    adata.layers["logfoldchange"] = shrunk_lfc
    adata.X = shrunk_lfc  # Update effect_size matrix
    
    # Update metadata
    adata.uns["lfc_shrinkage_type"] = shrinkage_type
    
    # Determine output path
    if data_name is None:
        # Append _shrunk to the stem
        stem = path.stem
        # Remove crispyx_ prefix if present for cleaner naming
        if stem.startswith("crispyx_"):
            stem = stem[8:]
        if stem.endswith("_shrunk"):
            # Already shrunk, use as-is
            data_name = stem
        else:
            data_name = f"{stem}_shrunk"
    
    # Use the data_name directly as output filename
    if output_dir is None:
        output_dir = path.parent
    else:
        output_dir = Path(output_dir)
    
    # Ensure crispyx prefix
    if not data_name.startswith("crispyx_"):
        output_filename = f"crispyx_{data_name}.h5ad"
    else:
        output_filename = f"{data_name}.h5ad"
    
    output_path = output_dir / output_filename
    adata.write(output_path)
    
    # Build result object
    gene_symbols = pd.Index(adata.var_names).astype(str)
    candidates = list(adata.obs_names.astype(str))
    perturbation_column = adata.uns.get("perturbation_column", "perturbation")
    control_label = adata.uns.get("control_label", "control")
    corr_method = adata.uns.get("pvalue_correction", "benjamini-hochberg")
    
    # Get matrices from layers
    statistic_matrix = adata.layers.get("z_score", np.zeros_like(shrunk_lfc))
    pvalue_matrix = adata.layers.get("pvalue", np.ones_like(shrunk_lfc))
    pvalue_adj_matrix = adata.layers.get("pvalue_adj", np.ones_like(shrunk_lfc))
    pts_matrix = adata.layers.get("pts", np.zeros((n_groups, n_genes), dtype=np.float32))
    pts_rest_matrix = adata.layers.get("pts_rest", np.zeros((n_groups, n_genes), dtype=np.float32))
    
    # Create order matrix
    statistic_for_order = np.where(
        np.isfinite(statistic_matrix), np.abs(statistic_matrix), -np.inf
    )
    order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")
    
    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=np.asarray(statistic_matrix),
        pvalues=np.asarray(pvalue_matrix),
        pvalues_adj=np.asarray(pvalue_adj_matrix),
        logfoldchanges=shrunk_lfc,
        effect_size=shrunk_lfc,
        u_statistics=np.zeros_like(shrunk_lfc),
        pts=np.asarray(pts_matrix, dtype=np.float32),
        pts_rest=np.asarray(pts_rest_matrix, dtype=np.float32),
        order=order_matrix,
        groupby=perturbation_column,
        method="nb_glm",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
    )
    result.result = AnnData(output_path)
    
    logger.info(
        f"Applied {shrinkage_type} LFC shrinkage to {n_groups} perturbations, "
        f"{n_genes} genes. Output: {output_path}"
    )
    
    return result
