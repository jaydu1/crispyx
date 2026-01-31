"""Differential expression testing utilities."""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Mapping, Tuple

from joblib import Parallel, delayed
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from numpy.typing import ArrayLike

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import norm, rankdata, t as t_dist

from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    ensure_gene_symbol_column,
    get_perturbation_slice,
    iter_matrix_chunks,
    needs_sorting_for_nbglm,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    sort_by_perturbation,
)
from .glm import (
    NBGLMFitter,
    NBGLMBatchFitter,
    ControlStatisticsCache,
    build_design_matrix,
    estimate_covariate_effects_streaming,
    estimate_dispersion_map,
    estimate_global_dispersion_streaming,
    fit_dispersion_trend,
    precompute_control_statistics,
    precompute_global_dispersion,
    precompute_global_dispersion_from_path,
    shrink_dispersions,
    shrink_lfc_apeglm,
    shrink_lfc_apeglm_from_stats,
    _estimate_apeglm_prior_scale,
)
from ._kernels import (
    _rankdata_2d_numba,
    _tie_correction_numba,
    _compute_rank_sums_batch_numba,
    _wilcoxon_sparse_batch_numba,
    _wilcoxon_all_perts_numba,
    _ZERO_PARTITION_THRESHOLD,
)
from ._checkpoint import (
    _write_checkpoint_atomic,
    _read_checkpoint,
    _scan_h5ad_completed,
    _get_resumable_candidates,
    _get_checkpoint_interval,
    _create_progress_context,
    _DummyProgress,
)
from ._size_factors import (
    _validate_size_factors,
    _median_of_ratios_size_factors,
    _deseq2_style_size_factors,
    _compute_subset_size_factors,
)
from ._statistics import (
    _tie_correction,
    _adjust_pvalue_matrix,
    _compute_se_batched,
    _compute_mom_dispersion_batched,
)
from ._memory import (
    _estimate_max_workers,
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


def _load_existing_nb_glm_result(
    output_path: Path,
    candidates: list[str],
    gene_symbols: list[str],
    perturbation_column: str,
    control_label: str,
    corr_method: str,
) -> "RankGenesGroupsResult":
    """Load an existing NB-GLM result from an h5ad file.
    
    Used when resume=True and all perturbations are already completed.
    """
    adata = ad.read_h5ad(output_path)
    
    # NB-GLM stores results in layers, not uns["rank_genes_groups"]
    statistic_matrix = np.array(adata.layers["z_score"])
    pvalue_matrix = np.array(adata.layers["pvalue"])
    pvalue_adj_matrix = np.array(adata.layers["pvalue_adj"])
    logfc_matrix = np.array(adata.layers["logfoldchange"])
    effect_matrix = logfc_matrix.copy()  # effect_size equals logfc for NB-GLM
    pts_matrix = np.array(adata.layers.get("pts", np.zeros_like(effect_matrix, dtype=np.float32)))
    pts_rest_matrix = np.array(adata.layers.get("pts_rest", np.zeros_like(effect_matrix, dtype=np.float32)))
    
    # Reconstruct order from statistics
    statistic_for_order = np.where(
        np.isfinite(statistic_matrix), np.abs(statistic_matrix), -np.inf
    )
    order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")
    
    return RankGenesGroupsResult(
        genes=pd.Index(gene_symbols),
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
        result=adata,
    )


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


def _write_rank_genes_groups_hdf5(
    output_path: Path,
    result: "RankGenesGroupsResult",
) -> None:
    """Write rank_genes_groups to HDF5 for Scanpy compatibility.
    
    Writes arrays in full matrix order (groups × genes) to uns/rank_genes_groups/full.
    This format is compatible with Scanpy's rank_genes_groups output but avoids
    the recarray format which causes HDF5 header size limits for large group counts.
    
    Parameters
    ----------
    output_path
        Path to the h5ad file to modify.
    result
        RankGenesGroupsResult containing the DE statistics.
        
    Notes
    -----
    For datasets with many groups (>1000), this adds ~2-6 seconds of I/O overhead.
    The recarray format (with group names as dtype fields) is avoided because it
    hits HDF5 header size limits at ~2000+ groups.
    """
    with h5py.File(output_path, "r+") as handle:
        uns_group = handle.require_group("uns")
        if "rank_genes_groups" in uns_group:
            del uns_group["rank_genes_groups"]
        rgg = uns_group.create_group("rank_genes_groups")
        
        # Store full-order matrices (groups × genes)
        full = rgg.create_group("full")
        full.create_dataset("scores", data=result.statistics)
        full.create_dataset("pvals", data=result.pvalues)
        full.create_dataset("pvals_adj", data=result.pvalues_adj)
        full.create_dataset("logfoldchanges", data=result.logfoldchanges)
        full.create_dataset("auc", data=result.effect_size)
        full.create_dataset("u_stat", data=result.u_statistics)
        full.create_dataset("pts", data=result.pts)
        full.create_dataset("pts_rest", data=result.pts_rest)
        
        # Store order and metadata
        rgg.create_dataset("order", data=result.order)
        rgg.create_dataset("names", data=np.array(result.groups, dtype="S"))
        
        # Store params for compatibility
        params = rgg.create_group("params")
        params.attrs["groupby"] = result.groupby
        params.attrs["method"] = result.method
        params.attrs["reference"] = result.control_label
        params.attrs["tie_correct"] = result.tie_correct
        params.attrs["corr_method"] = result.pvalue_correction


def t_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    cell_chunk_size: int | None = None,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
    verbose: bool = False,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    scanpy_format: bool = False,
) -> RankGenesGroupsResult:
    """Perform a t-test comparing log-expression means for each perturbation.
    
    Returns a RankGenesGroupsResult containing differential expression statistics.
    Results are stored in an h5ad file with layers containing the statistics.
    
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
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        log-transformed expression data.
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
    verbose
        If True, show a progress bar for perturbation processing. Requires tqdm.
    resume
        If True, attempt to resume from a previous interrupted run using checkpoint.
    checkpoint_interval
        Number of perturbations between checkpoint saves. Auto-determined if None.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.
    """

    path = resolve_data_path(data)
    backed = read_backed(path)
    try:
        # Calculate adaptive chunk_size if not provided
        if cell_chunk_size is None:
            cell_chunk_size = calculate_optimal_chunk_size(backed.n_obs, backed.n_vars)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")
    
    # Handle resume logic
    if resume:
        candidates_to_run, completed_labels, failed_labels = _get_resumable_candidates(
            checkpoint_path, output_path, candidates, retry_failed=True
        )
    else:
        candidates_to_run = candidates
        completed_labels = []
        failed_labels = []
    
    # Determine checkpoint interval
    eff_checkpoint_interval = _get_checkpoint_interval(n_groups, checkpoint_interval)
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}
    
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

                # Compute variance terms for SE and Welch-Satterthwaite df
                var_term_pert = var_buffer / n_cells  # var_pert / n_pert
                var_term_ctrl = control_var / control_n  # var_ctrl / n_ctrl
                
                # SE = sqrt(var_pert/n_pert + var_ctrl/n_ctrl)
                np.add(var_term_pert, var_term_ctrl, out=se_buffer)
                np.sqrt(se_buffer, out=se_buffer)

                total_expr = expr_counts[idx] + expr_counts[control_idx]
                valid = (se_buffer > 0) & (total_expr >= min_cells_expressed)

                stat_buffer[slot].fill(0)
                pval_buffer[slot].fill(1)
                stat_buffer[slot][valid] = effect_f32[valid] / se_buffer[valid]
                
                # Welch-Satterthwaite degrees of freedom for Welch's t-test
                # df = (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
                numerator = (var_term_pert + var_term_ctrl) ** 2
                denominator = np.zeros_like(numerator)
                if n_cells > 1:
                    denominator += (var_term_pert ** 2) / (n_cells - 1)
                if control_n > 1:
                    denominator += (var_term_ctrl ** 2) / (control_n - 1)
                # Avoid division by zero; set df to a large value when denominator is 0
                # Use np.divide with where to prevent NaN from 0/0
                df_welch = np.divide(
                    numerator, denominator,
                    out=np.full_like(numerator, 1e6),
                    where=denominator > 0
                )
                # Clip df to reasonable bounds (minimum 1, no upper limit needed)
                df_welch = np.clip(df_welch, 1.0, None)
                
                # Use t-distribution for p-value calculation (matches scanpy's Welch's t-test)
                pval_buffer[slot][valid] = 2 * t_dist.sf(np.abs(stat_buffer[slot][valid]), df_welch[valid])

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

            # Track completed labels
            newly_completed = list(completed_labels)
            newly_failed = list(failed_labels)
            completed_set = set(completed_labels)
            n_processed = 0
            
            # Helper to save checkpoint
            def _save_t_test_checkpoint() -> None:
                checkpoint_data = {
                    "total": n_groups,
                    "completed": newly_completed,
                    "failed": newly_failed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": "t_test",
                    "control_label": control_label,
                }
                _write_checkpoint_atomic(checkpoint_path, checkpoint_data)

            if n_groups > 0:
                with _create_progress_context(len(candidates_to_run), "t-test DE", verbose) as pbar:
                    for batch_start in range(0, n_groups, batch_size):
                        batch_labels = candidates[batch_start : batch_start + batch_size]
                        # Filter to only labels that need processing
                        batch_to_run = [l for l in batch_labels if l not in completed_set]
                        
                        for local_idx, label in enumerate(batch_labels):
                            if label in completed_set:
                                continue
                            try:
                                compute_perturbation(label, local_idx)
                            except Exception as e:
                                logger.error(f"Failed perturbation {label}: {e}")
                                newly_failed.append(label)
                                continue

                        for local_idx, label in enumerate(batch_labels):
                            if label in completed_set or label in newly_failed:
                                continue
                            global_idx = candidate_to_idx[label]
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
                            
                            newly_completed.append(label)
                            n_processed += 1
                            pbar.update(1)
                            logger.debug(f"Completed perturbation: {label}")
                        
                        # Save checkpoint after each batch
                        if len(batch_to_run) > 0 and n_processed % eff_checkpoint_interval == 0:
                            _save_t_test_checkpoint()
                
                # Final checkpoint
                _save_t_test_checkpoint()
                logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")

            pvalue_adj_memmap = np.memmap(
                tmp_path / "pvalues_adj.dat", mode="w+", dtype=np.float64, shape=shape
            )
            _adjust_pvalue_matrix(pval_memmap, method="benjamini-hochberg", out=pvalue_adj_memmap)
            ds_pvals_adj[:] = pvalue_adj_memmap

        # Convert memmap arrays to regular arrays before tempdir cleanup
        stat_matrix = np.asarray(stat_memmap)
        pval_matrix = np.asarray(pval_memmap)
        pval_adj_matrix = np.asarray(pvalue_adj_memmap)
        lfc_matrix = np.asarray(lfc_memmap)
        effect_matrix = np.asarray(effect_memmap)
        pts_matrix = np.asarray(pts_memmap)
        pts_rest_matrix = np.asarray(pts_rest_memmap)
        order_matrix = np.asarray(order_memmap)
        
        result = RankGenesGroupsResult(
            genes=gene_symbols,
            groups=candidates,
            statistics=stat_matrix,
            pvalues=pval_matrix,
            pvalues_adj=pval_adj_matrix,
            logfoldchanges=lfc_matrix,
            effect_size=effect_matrix,
            u_statistics=np.zeros(shape, dtype=np.float32),
            pts=pts_matrix,
            pts_rest=pts_rest_matrix,
            order=order_matrix,
            groupby=perturbation_column,
            method="t_test",
            control_label=control_label,
            tie_correct=False,
            pvalue_correction="benjamini-hochberg",
            result=None,
        )

    # Create AnnData with layer-based storage (avoid recarray-based rank_genes_groups
    # which fails with HDF5 header size limits for large group counts)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    adata.layers["z_score"] = stat_matrix  # t-statistic (converges to z for large n)
    adata.layers["pvalue"] = pval_matrix
    adata.layers["pvalue_adj"] = pval_adj_matrix
    adata.layers["logfoldchange"] = lfc_matrix
    adata.layers["pts"] = pts_matrix
    adata.layers["pts_rest"] = pts_rest_matrix
    adata.uns["method"] = "t_test"
    adata.uns["control_label"] = control_label
    adata.uns["perturbation_column"] = perturbation_column
    adata.uns["pvalue_correction"] = "benjamini-hochberg"
    adata.write(output_path)
    
    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format:
        _write_rank_genes_groups_hdf5(output_path, result)
    
    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    result.result = AnnData(output_path)
    return result


def nb_glm_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    # ---- Data parameters ----
    perturbation_column: str,
    control_label: str | None = None,
    covariates: Iterable[str] | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    # ---- Size factor parameters ----
    size_factors: ArrayLike | None = None,
    size_factor_method: Literal["sparse", "deseq2"] = "sparse",
    size_factor_scope: Literal["global", "per_comparison"] = "global",
    scale_size_factors: bool = True,
    # ---- Dispersion parameters ----
    dispersion: float | None = None,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    dispersion_scope: Literal["global", "per_comparison"] = "global",
    share_dispersion: bool = False,
    use_map_dispersion: bool = True,
    shrink_dispersion: bool = True,
    # ---- Optimization parameters ----
    optimization_method: Literal["irls", "lbfgsb"] = "lbfgsb",
    max_iter: int = 25,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    poisson_init_iter: int = 5,
    chunk_size: int = 256,
    irls_batch_size: int | None = 128,
    # ---- Filtering parameters ----
    min_cells_expressed: int = 0,
    min_total_count: float = 1.0,
    cook_filter: bool = False,
    # ---- Output parameters ----
    lfc_shrinkage_type: Literal["apeglm", "none"] = "none",
    lfc_base: Literal["log2", "ln"] = "log2",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    se_method: Literal["sandwich", "fisher"] = "sandwich",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    scanpy_format: bool = False,
    verbose: bool = False,
    profiling: bool = False,
    # ---- Resume/Memory parameters ----
    resume: bool = False,
    checkpoint_interval: int | None = None,
    memory_limit_gb: float | None = None,
    max_dense_fraction: float = 0.3,
    n_jobs: int | None = None,
    max_workers: int | None = None,
    use_control_cache: bool = True,
    freeze_control: bool | None = None,
) -> RankGenesGroupsResult:
    """Perform negative binomial GLM differential expression test.
    
    Returns a RankGenesGroupsResult containing differential expression statistics.
    Uses a negative binomial GLM framework that can incorporate covariates. 
    Results are stored in an h5ad file with layers containing the statistics.
    
    The RankGenesGroupsResult implements the Mapping interface, so it can be used 
    like a dict: `result[perturbation_label]` returns a DifferentialExpressionResult
    for that perturbation.
    
    Parameters
    ----------
    
    **Data parameters**
    
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        raw count data.
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
    
    **Size factor parameters**
    
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
    size_factor_scope
        Scope for size factor computation:
        - "global" (default): Compute size factors once on the full dataset. 
          Recommended for CRISPR screens where all cells come from the same 
          experiment and share a common sequencing depth distribution. Faster 
          when combined with use_control_cache=True. Note: produces different 
          results from PyDESeq2 (ρ ≈ 0.7-0.8) which uses per-comparison normalization.
        - "per_comparison": Compute size factors separately for each control + 
          perturbation comparison. This matches PyDESeq2's behavior exactly, 
          leading to near-perfect LFC, statistic, and p-value concordance 
          (ρ > 0.97 on Tian-crispra, ρ > 0.99 on Adamson_subset). Use this when 
          PyDESeq2 compatibility is required or for bulk RNA-seq style analysis.
    scale_size_factors
        If True (default), scale size factors so their geometric mean equals 1.
        This is the standard DESeq2/crispyx behavior. If False, use raw 
        median-of-ratios size factors without rescaling, which matches 
        PyDESeq2's default behavior and can improve numerical alignment.
    
    **Dispersion parameters**
    
    dispersion
        Fixed dispersion parameter for negative binomial. If None, estimates per gene.
    dispersion_method
        Method for estimating dispersion when ``dispersion`` is None:
        - "moments": Method-of-moments (fast but less accurate)
        - "cox-reid": Cox-Reid adjusted profile likelihood (slower but more
          accurate, similar to DESeq2). This is the default.
    dispersion_scope
        Scope for dispersion estimation:
        - "global" (default): Precompute dispersion once using all cells (control +
          all perturbations). This is ~10× faster for multi-perturbation datasets
          since MAP dispersion estimation is done once instead of per-comparison.
          Recommended when perturbation effects are expected to be small relative
          to baseline expression (typical for CRISPR screens).
        - "per_comparison": Estimate dispersion separately for each control + 
          perturbation comparison. More accurate when perturbations cause large
          changes in gene expression variance, but significantly slower.
    share_dispersion
        If True, estimate dispersion once using all cells, then use the same 
        dispersion values for all Wald tests. If False (default), estimate
        dispersion separately for each perturbation comparison.
    use_map_dispersion
        If True (default), use MAP dispersion estimation with mean-dispersion trend.
        If False, use MLE dispersion without trend-based shrinkage.
    shrink_dispersion
        If True, fit a mean-dispersion trend and shrink gene-wise dispersions
        toward the trend using an empirical Bayes prior.
    
    **Optimization parameters**
    
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
    min_mu
        Minimum mean threshold for IRLS numerical stability. Predicted means
        are clamped to max(min_mu, predicted) during fitting to prevent
        numerical instability from very small means. Default: 0.5, matching
        PyDESeq2's default. Set to 0.0 to disable clamping.
    poisson_init_iter
        Initial Poisson iterations before switching to negative binomial.
    chunk_size
        Number of genes to process per chunk (memory vs. speed tradeoff). Smaller
        values stream more, reducing peak memory at the cost of additional I/O.
    irls_batch_size
        Maximum number of genes to densify per IRLS step. Keep this small to
        limit per-iteration memory when working with large sparse matrices. Set
        to ``None`` to process each chunk without additional batching.
    
    **Filtering parameters**
    
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for testing.
    min_total_count
        Minimum total count across all cells for a gene to be tested.
    cook_filter
        Whether to apply Cook's distance outlier filtering when available.
    
    **Output parameters**
    
    lfc_shrinkage_type
        Type of log-fold change shrinkage to apply:
        - "none": No shrinkage (default)
        - "apeglm": Adaptive shrinkage using Cauchy prior (PyDESeq2-compatible).
          Preserves large effects while shrinking small/uncertain effects toward
          zero. Also updates standard errors to reflect posterior uncertainty.
    lfc_base
        Log base for fold change output:
        - "log2" (default): Output log2 fold change, matching PyDESeq2/edgeR.
        - "ln": Output natural log fold change (raw GLM coefficients).
        Standard error is also converted to match the selected log base.
        Wald statistics remain unchanged since both LFC and SE are scaled equally.
    corr_method
        Method for p-value correction: "benjamini-hochberg" or "bonferroni".
    se_method
        Method for computing standard errors:
        - "sandwich" (default): Sandwich estimator SE = sqrt(c' @ H @ M @ H @ c).
          More robust to model misspecification.
        - "fisher": Standard Fisher information SE = sqrt(diag(inv(X'WX + ridge*I))).
          Matches PyDESeq2's approach for better p-value parity.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "nb_glm" suffix.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.
    verbose
        If True, show a progress bar for perturbation fitting and log per-perturbation
        completion at DEBUG level. Requires tqdm to be installed for progress bar.
    profiling
        If True, enable timing and memory profiling. When enabled, stores
        profiling data in `adata.uns["profiling"]` with fields:
        - `fit_seconds`: Time for base NB-GLM fitting (excludes lfcShrink)
        - `fit_peak_memory_mb`: Peak memory during fitting
        - `profiling_enabled`: True
        When False (default), `adata.uns["profiling"]` is set to "NA" to
        avoid profiling overhead in production.
    
    **Resume/Memory parameters**
    
    resume
        If True, attempt to resume from a previous interrupted run. Reads the
        checkpoint file to determine which perturbations have already been
        completed and skips them. If the checkpoint file is missing or corrupted,
        falls back to scanning the output h5ad to detect completed perturbations.
    checkpoint_interval
        Number of perturbations to process between checkpoint saves. If None,
        auto-determined based on dataset size (1 for <100 perturbations, 10 for
        <1000, 50 for larger). The checkpoint file `<output>_progress.json` is
        written atomically to prevent corruption.
    memory_limit_gb
        Optional memory limit in GB. If provided, this is used together with
        available system memory to determine when to switch to streaming mode
        for global dispersion estimation. The effective limit is
        min(available_memory, memory_limit_gb). Default is None (use system memory only).
    max_dense_fraction
        Maximum fraction of available memory to use for dense matrix operations.
        If the estimated memory for densifying the full cell×gene matrix exceeds
        max_dense_fraction × min(available_memory, memory_limit_gb), the function
        switches to streaming mode. Default is 0.3 (30% of available memory).
    n_jobs
        Number of parallel workers for fitting GLMs across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
        If -1, uses all available cores.
    max_workers
        Alias for n_jobs (for compatibility). If both are specified, n_jobs takes precedence.
    use_control_cache
        If True (default), precompute control cell statistics (intercept, weights,
        XᵀWX contributions) once and reuse them across all perturbation comparisons.
        This can significantly reduce memory and computation time when there are
        many perturbations and the control group is large. Only applies when
        no covariates are specified and size_factor_scope="global".
    freeze_control
        Whether to use frozen control sufficient statistics instead of raw control
        matrix for parallel fitting. This dramatically reduces per-worker memory
        from ~5GB to ~1MB for large datasets, enabling more parallel workers.
        
        - None (default): Auto-detect based on dataset size. Frozen control is
          enabled when control_matrix serialization would limit workers to <4,
          AND the required settings (dispersion_scope='global', shrink_dispersion=True)
          are met. For most large datasets (>500K cells), this auto-enables.
        - True: Force frozen control mode. Raises ValueError if requirements not met.
        - False: Disable frozen control (use raw control matrix).
        
        Memory efficiency: Per-worker pickle size is reduced from (control_n × n_genes × 8)
        bytes to just ~1MB of sufficient statistics (W_sum, Wz_sum arrays).
        
        Example: For Replogle-GW-k562 (75K control cells × 8K genes):
        - Without freeze_control: ~4.7 GB per worker → 2 workers max @ 128GB
        - With freeze_control: ~1 MB per worker → 32 workers @ 128GB
        - Time reduction: ~300 hours → ~10 hours
        
        Requirements (enforced when True, auto-checked when None):
        - dispersion_scope="global" (per-comparison dispersion needs raw matrix)
        - shrink_dispersion=True (ensures global dispersion is computed)
        - use_control_cache=True (required for caching)
        
        Technical note: The intercept (β₀) is frozen to the control-only estimate.
        This is valid because control cells have perturbation indicator = 0, so
        μ_control = exp(β₀ + offset) is independent of the perturbation effect.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.
    """
    # Validate min_mu parameter
    if min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")

    covariates = list(covariates or [])

    # Initialize profiler if enabled (timing + memory sampling)
    profiler = None
    if profiling:
        from .profiling import Profiler
        profiler = Profiler(timing=True, memory=True, memory_method="rss", sampling=True)
        profiler.start("total")
        profiler.start("fit")  # Start fit timing (excludes shrinkage which is separate)

    # Check if dataset needs sorting for efficient I/O
    # Large datasets with many perturbations benefit from having cells sorted
    # by perturbation label, enabling contiguous reads instead of random access
    path = resolve_data_path(data)
    if needs_sorting_for_nbglm(path, perturbation_column=perturbation_column):
        sorted_path = path.parent / f"{path.stem}_sorted.h5ad"
        if not sorted_path.exists():
            logger.info(
                f"Large dataset detected with scattered cells. "
                f"Sorting by perturbation for efficient I/O..."
            )
            path = sort_by_perturbation(
                path,
                perturbation_column=perturbation_column,
                control_label=control_label,
                output_path=sorted_path,
            )
        else:
            logger.info(f"Using existing sorted dataset: {sorted_path}")
            path = sorted_path

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
        
        # Check if file is sorted by perturbation for efficient I/O
        perturbation_boundaries = None
        if "sorting_metadata" in backed.uns:
            metadata = backed.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                perturbation_boundaries = metadata.get("perturbation_boundaries", {})
                if perturbation_boundaries:
                    logger.debug(
                        f"Using sorted file with {len(perturbation_boundaries)} contiguous perturbation groups"
                    )
    finally:
        backed.file.close()

    n_cells_total = obs_df.shape[0]
    
    # For per_comparison size factors, we skip the expensive global computation
    # since size factors will be recomputed per-comparison anyway.
    # We use dummy size factors here (will be overwritten per-comparison).
    if size_factor_scope == "per_comparison" and size_factors is None:
        # Dummy size factors - will be replaced in worker
        size_factors = np.ones(n_cells_total, dtype=np.float64)
        logger.debug("Skipping global size factor computation for per_comparison mode")
    elif size_factors is None:
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
    # Independent fitting mode: per-perturbation approach
    # =========================================================================
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
        min_mu: float,
        poisson_init_iter: int,
        dispersion_method: str,
        global_dispersion: np.ndarray | None,
        shrink_dispersion: bool,
        use_map_dispersion: bool,
        lfc_shrinkage_type: str,
        pts_rest_shared: np.ndarray,
        full_X: np.ndarray | sp.csr_matrix | None = None,
        per_comparison_sf: bool = False,
        se_method: str = "sandwich",
        perturbation_boundaries: dict | None = None,
    ) -> dict:
        """Fit NB-GLM for a single perturbation group and return results."""
        group_mask = labels == label
        subset_mask = control_mask | group_mask
        subset_obs = obs_df.iloc[subset_mask]
        indicator = group_mask[subset_mask].astype(np.float64)
        
        # Compute per-comparison size factors if requested (matches PyDESeq2)
        if per_comparison_sf and full_X is not None:
            subset_sf = _compute_subset_size_factors(full_X, subset_mask, scale=True)
            subset_size_factors = subset_sf
            subset_offset = np.log(np.clip(subset_sf, 1e-8, None))
        else:
            subset_size_factors = np.asarray(size_factors)[subset_mask]
            subset_offset = offset[subset_mask] if offset is not None else np.log(np.clip(subset_size_factors, 1e-8, None))
        
        # Build design matrix
        
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
        # Use slice-based access if file is sorted, otherwise use mask
        backed = read_backed(path)
        try:
            if perturbation_boundaries is not None and label in perturbation_boundaries:
                # Slice-based access for sorted files (contiguous, fast)
                start, end = perturbation_boundaries[label]
                group_matrix = backed.X[start:end, :]
            else:
                # Mask-based access for unsorted files (random, slower)
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
            "intercept": np.full(n_genes, np.nan, dtype=np.float64),  # MLE intercept for shrink_lfc
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
            offset=subset_offset,
            max_iter=max_iter,
            tol=tol,
            poisson_init_iter=poisson_init_iter,
            dispersion_method=dispersion_method,
            min_mu=min_mu,
            min_total_count=min_total_count,
        )
        
        batch_result = batch_fitter.fit_batch(fit_matrix)

        # Extract results
        result["converged"][valid_indices] = batch_result.converged
        result["iterations"][valid_indices] = batch_result.n_iter
        result["dispersion_raw"][valid_indices] = batch_result.dispersion

        coefs = batch_result.coef[:, perturbation_column_index]
        ses = batch_result.se[:, perturbation_column_index]
        intercepts = batch_result.coef[:, 0]  # Intercept is always first column

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
            result["intercept"][gene_idx] = intercepts[local_idx]  # Store fitted intercept

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
                disp_map = estimate_dispersion_map(
                    Y, mu, trend, max_disp=max_disp
                )
                result["dispersion"] = disp_map
                
                # CRITICAL: Recompute SE using MAP dispersion (PyDESeq2 style)
                # SE from IRLS was computed using MoM dispersion, but Wald test
                # should use MAP dispersion for proper variance estimation
                ridge = 1e-6
                for local_idx, gene_idx in enumerate(valid_indices):
                    if not valid_results[local_idx]:
                        continue
                    # Compute weights with MAP dispersion: W = mu / (1 + mu * disp)
                    mu_gene = mu[:, gene_idx]
                    disp_gene = disp_map[gene_idx]
                    W = mu_gene / (1.0 + mu_gene * disp_gene)
                    # Compute (X'WX + ridge*I)^{-1}
                    XtW = design.T * W[None, :]
                    XtWX = XtW @ design
                    XtWX += ridge * np.eye(design.shape[1])
                    try:
                        inv_XtWX = np.linalg.inv(XtWX)
                        se_new = np.sqrt(np.maximum(inv_XtWX[perturbation_column_index, perturbation_column_index], 1e-10))
                    except np.linalg.LinAlgError:
                        se_new = result["se"][gene_idx]  # Keep original SE on failure
                    # Update SE, statistic, and p-value
                    coef = result["logfc"][gene_idx]
                    result["se"][gene_idx] = se_new
                    statistic = coef / se_new
                    result["statistic"][gene_idx] = statistic
                    result["pvalue"][gene_idx] = float(2.0 * norm.sf(abs(statistic)))
            else:
                result["dispersion"] = shrink_dispersions(result["dispersion_raw"], trend)
        else:
            result["dispersion_trend"] = result["dispersion_raw"].copy()
            result["dispersion"] = result["dispersion_raw"].copy()

        result["logfc_raw"] = result["logfc"].copy()
        if lfc_shrinkage_type == "apeglm":
            # Full NB-GLM re-fitting with Cauchy prior (matches PyDESeq2)
            # Build mle_coef matrix (n_params, n_genes) from batch_result
            n_params = design.shape[1]
            mle_coef = np.full((n_params, n_genes), np.nan, dtype=np.float64)
            for local_idx, gene_idx in enumerate(valid_indices):
                if valid_results[local_idx]:
                    mle_coef[:, gene_idx] = batch_result.coef[local_idx, :]
            
            # Densify the subset matrix if sparse
            if sp.issparse(subset_matrix):
                counts_dense = subset_matrix.toarray()
            else:
                counts_dense = np.asarray(subset_matrix, dtype=np.float64)
            
            # Call full apeGLM shrinkage with L-BFGS-B re-fitting
            # NOTE: PyDESeq2's lfc_shrink does NOT use min_mu during shrinkage
            shrunk_coef, shrunk_se_arr, shrink_converged = shrink_lfc_apeglm(
                counts=counts_dense,
                design_matrix=design,
                size_factors=subset_size_factors,
                dispersion=result["dispersion"],
                mle_coef=mle_coef,
                mle_se=result["se"],
                shrink_index=perturbation_column_index,
                prior_scale=None,  # Estimate globally from MLE LFC distribution
                n_jobs=1,  # Single-threaded within worker (parallelism at perturbation level)
                min_mu=0.0,  # No min_mu - match PyDESeq2's lfc_shrink behavior
            )
            # Extract shrunken LFC (perturbation coefficient)
            result["logfc"] = shrunk_coef[perturbation_column_index, :]
            result["effect"] = result["logfc"].copy()
            result["se"] = shrunk_se_arr  # Posterior SE from inverse Hessian
        else:
            result["effect"] = result["logfc"].copy()

        return result
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Optimized worker using precomputed control cache
    # -------------------------------------------------------------------------
    def _fit_perturbation_worker_cached(
        group_idx: int,
        label: str,
        path: str | Path,
        labels: np.ndarray,
        control_cache: ControlStatisticsCache,
        size_factors: np.ndarray,
        n_genes: int,
        min_cells_expressed: int,
        min_total_count: float,
        max_iter: int,
        tol: float,
        min_mu: float,
        dispersion_method: str,
        shrink_dispersion: bool,
        use_map_dispersion: bool,
        lfc_shrinkage_type: str,
        se_method: str = "sandwich",
        perturbation_boundaries: dict | None = None,
    ) -> dict:
        """Fit NB-GLM for a perturbation group using cached control statistics.
        
        This is an optimized version that reuses precomputed control cell
        statistics (intercept, weights, XᵀWX contributions) to avoid redundant
        computation across perturbation comparisons.
        """
        group_mask = labels == label
        group_n = int(group_mask.sum())
        n_control = control_cache.control_n
        subset_n = n_control + group_n
        
        # Initialize result arrays
        result = {
            "group_idx": group_idx,
            "effect": np.full(n_genes, np.nan, dtype=np.float64),
            "statistic": np.full(n_genes, np.nan, dtype=np.float64),
            "pvalue": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "intercept": np.full(n_genes, np.nan, dtype=np.float64),  # MLE intercept for shrink_lfc
            "se": np.full(n_genes, np.nan, dtype=np.float64),
            "pts": np.zeros(n_genes, dtype=np.float32),
            "pts_rest": control_cache.pts_rest.copy(),
            "dispersion": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_trend": np.full(n_genes, np.nan, dtype=np.float64),
            "mean": np.zeros(n_genes, dtype=np.float64),
            "iterations": np.zeros(n_genes, dtype=np.int32),
            "converged": np.zeros(n_genes, dtype=bool),
        }
        
        # Load perturbation group cells
        # Use slice-based access if file is sorted, otherwise use mask
        backed = read_backed(path)
        try:
            if perturbation_boundaries is not None and label in perturbation_boundaries:
                # Slice-based access for sorted files (contiguous, fast)
                start, end = perturbation_boundaries[label]
                group_matrix = backed.X[start:end, :]
            else:
                # Mask-based access for unsorted files (random, slower)
                group_matrix = backed.X[group_mask, :]
            if sp.issparse(group_matrix):
                group_matrix = sp.csr_matrix(group_matrix, dtype=np.float64)
            else:
                group_matrix = np.asarray(group_matrix, dtype=np.float64)
        finally:
            backed.file.close()
        
        # Compute expression counts for perturbation cells
        if sp.issparse(group_matrix):
            group_expr_counts = np.asarray(group_matrix.getnnz(axis=0)).ravel()
        else:
            group_expr_counts = np.sum(group_matrix > 0, axis=0)
        
        # Valid genes mask
        total_expr_counts = control_cache.control_expr_counts + group_expr_counts
        valid_mask = total_expr_counts >= min_cells_expressed
        
        # Compute pts
        pts = np.divide(
            group_expr_counts,
            group_n,
            out=np.zeros(n_genes, dtype=np.float32),
            where=group_n > 0,
        )
        result["pts"] = np.where(valid_mask, pts, 0.0).astype(np.float32)
        
        # Compute mean expression
        subset_size_factors_group = np.asarray(size_factors)[group_mask]
        if sp.issparse(group_matrix):
            normalized_group = group_matrix.multiply(1.0 / subset_size_factors_group[:, None])
            mean_group = np.asarray(normalized_group.sum(axis=0)).ravel()
        else:
            normalized_group = group_matrix / subset_size_factors_group[:, None]
            mean_group = normalized_group.sum(axis=0)
        
        # Combined mean
        result["mean"] = (control_cache.control_mean_expr * n_control + mean_group) / subset_n
        
        if not np.any(valid_mask):
            return result
        
        # Get perturbation offset
        perturbation_offset = np.log(np.clip(subset_size_factors_group, 1e-8, None))
        
        # Create perturbation indicator for the combined design (control first, then perturbation)
        perturbation_indicator = np.concatenate([
            np.zeros(n_control, dtype=np.float64),
            np.ones(group_n, dtype=np.float64)
        ])
        
        # Check if using frozen control mode (memory-efficient parallel fitting)
        if control_cache.use_frozen_control:
            # FROZEN CONTROL PATH: Use sufficient statistics instead of raw matrix
            # Per-worker memory: ~5GB → ~1MB (enables 32 workers instead of 2)
            
            # Create batch fitter with minimal design (only perturbation cells have data)
            batch_fitter = NBGLMBatchFitter(
                design=np.ones((group_n, 1)),  # Placeholder, not used in frozen mode
                offset=perturbation_offset,  # Only perturbation offsets needed
                max_iter=max_iter,
                tol=tol,
                dispersion_method=dispersion_method,
                min_mu=min_mu,
                min_total_count=min_total_count,
            )
            
            batch_result = batch_fitter.fit_batch_with_frozen_control(
                perturbation_matrix=group_matrix,
                perturbation_offset=perturbation_offset,
                control_cache=control_cache,
                valid_mask=valid_mask,
            )
        else:
            # STANDARD PATH: Full control_matrix available
            # Fit using the batch fitter with control cache
            batch_fitter = NBGLMBatchFitter(
                design=np.column_stack([np.ones(subset_n), perturbation_indicator]),
                offset=np.concatenate([control_cache.control_offset, perturbation_offset]),
                max_iter=max_iter,
                tol=tol,
                dispersion_method=dispersion_method,
                min_mu=min_mu,
                min_total_count=min_total_count,
            )
            
            batch_result = batch_fitter.fit_batch_with_control_cache(
                perturbation_matrix=group_matrix,
                perturbation_offset=perturbation_offset,
                control_cache=control_cache,
                perturbation_indicator=perturbation_indicator,
                valid_mask=valid_mask,
            )

        
        valid_indices = np.where(valid_mask)[0]
        
        # Extract results
        result["converged"][valid_indices] = batch_result.converged[valid_indices]
        result["iterations"][valid_indices] = batch_result.n_iter[valid_indices]
        
        # Perturbation coefficient is at index 1 - vectorized computation
        coefs = batch_result.coef[:, 1]  # (n_genes,)
        ses = batch_result.se[:, 1]  # (n_genes,)
        
        # Vectorized: compute statistics for all valid & converged genes at once
        # Build mask for valid, converged genes with finite coef/se
        valid_converged_mask = (
            valid_mask & 
            batch_result.converged &
            np.isfinite(coefs) & 
            np.isfinite(ses) & 
            (ses > 0)
        )
        
        # Compute Wald statistic and p-value vectorized
        statistics = np.divide(
            coefs, ses, 
            out=np.full(n_genes, np.nan), 
            where=valid_converged_mask
        )
        # Use norm.sf (survival function) to avoid underflow for large |z|
        # Note: ndtr(x) = CDF, so 1-ndtr(x) underflows for large x; sf(x) = 1-CDF is stable
        pvalues = np.where(
            valid_converged_mask,
            2.0 * norm.sf(np.abs(statistics)),  # 2-sided p-value
            np.nan
        )
        
        result["statistic"] = statistics
        result["pvalue"] = pvalues
        result["logfc"] = np.where(valid_converged_mask, coefs, np.nan)
        result["se"] = np.where(valid_converged_mask, ses, np.nan)
        result["intercept"] = np.where(valid_converged_mask, batch_result.coef[:, 0], np.nan)  # Store fitted intercept
        
        # Handle dispersion shrinkage
        if shrink_dispersion:
            # ================================================================
            # MEMORY-OPTIMIZED: Use batched computation for dispersion and SE
            # ================================================================
            # Instead of building full (n_control+n_group, n_genes) matrices,
            # process genes in batches to reduce peak memory.
            # ================================================================
            
            # Get coefficients (needed for SE recomputation)
            beta0_all = batch_result.coef[:, 0]  # (n_genes,)
            beta1_all = batch_result.coef[:, 1]  # (n_genes,)
            
            # ================================================================
            # OPTIMIZATION: Check for global dispersion FIRST (before MoM/trend)
            # When dispersion_scope='global', skip expensive per-comparison work
            # ================================================================
            if control_cache.global_dispersion is not None:
                # FAST PATH: Use precomputed global dispersion
                # Skip MoM dispersion computation entirely
                # Skip trend fitting entirely
                logger.debug(f"Using precomputed global dispersion for {label}")
                result["dispersion"] = control_cache.global_dispersion.copy()
                if control_cache.global_dispersion_trend is not None:
                    result["dispersion_trend"] = control_cache.global_dispersion_trend.copy()
                else:
                    # Fallback: use global dispersion as trend proxy
                    result["dispersion_trend"] = control_cache.global_dispersion.copy()
                # dispersion_raw not computed in global mode - set to NaN or copy global
                result["dispersion_raw"] = control_cache.global_dispersion.copy()
                
                # SE handling: frozen control vs standard
                if control_cache.use_frozen_control:
                    # FROZEN CONTROL PATH: Use SE from fit_batch_with_frozen_control directly
                    # No SE recomputation needed since dispersion is global (pre-fitted)
                    # The SE already uses the correct dispersion from control_cache
                    pass  # SE already set from batch_result
                else:
                    # STANDARD PATH: Recompute SE with global dispersion
                    n_control = control_cache.control_matrix.shape[0]
                    n_group = group_matrix.shape[0]
                    if sp.issparse(group_matrix):
                        Y_pert_dense = group_matrix.toarray()
                    else:
                        Y_pert_dense = np.asarray(group_matrix, dtype=np.float64)
                    
                    final_disp = result["dispersion"]
                    recomputed_se = _compute_se_batched(
                        Y_control=control_cache.control_matrix,
                        Y_pert=Y_pert_dense,
                        control_offset=control_cache.control_offset,
                        pert_offset=perturbation_offset,
                        beta0=beta0_all,
                        beta1=beta1_all,
                        dispersion=final_disp,
                        gene_batch_size=5000,
                        se_method=se_method,
                    )
                    
                    # Update result with recomputed SE
                    result["se"] = np.where(valid_converged_mask, recomputed_se, np.nan)
                    
                    # Recompute Wald statistic and p-value with new SE
                    coefs = batch_result.coef[:, 1]
                    statistics = np.divide(
                        coefs, recomputed_se, 
                        out=np.full(n_genes, np.nan), 
                        where=valid_converged_mask
                    )
                    pvalues = np.where(
                        valid_converged_mask,
                        2.0 * norm.sf(np.abs(statistics)),
                        np.nan
                    )
                    result["statistic"] = statistics
                    result["pvalue"] = pvalues
            else:
                # ============================================================
                # STANDARD PATH: Per-comparison dispersion (MoM → trend → MAP)
                # Requires control_matrix - not compatible with frozen control
                # ============================================================
                if control_cache.use_frozen_control:
                    raise ValueError(
                        "Frozen control mode requires global dispersion. "
                        "Set dispersion_scope='global' and shrink_dispersion=True when using freeze_control=True."
                    )
                
                # Prepare perturbation matrix as dense (needed for SE recomputation)
                n_control = control_cache.control_matrix.shape[0]
                n_group = group_matrix.shape[0]
                if sp.issparse(group_matrix):
                    Y_pert_dense = group_matrix.toarray()
                else:
                    Y_pert_dense = np.asarray(group_matrix, dtype=np.float64)
                
                # Compute MoM dispersion using batched processing
                mom_disp = _compute_mom_dispersion_batched(
                    Y_control=control_cache.control_matrix,
                    Y_pert=Y_pert_dense,
                    control_offset=control_cache.control_offset,
                    pert_offset=perturbation_offset,
                    beta0=beta0_all,
                    beta1=beta1_all,
                    converged=batch_result.converged,
                    gene_batch_size=5000,
                )
                result["dispersion_raw"][valid_indices] = mom_disp[valid_indices]
                
                # Fit trend using corrected MoM dispersion
                trend = fit_dispersion_trend(result["mean"], result["dispersion_raw"])
                result["dispersion_trend"] = trend
                
                if use_map_dispersion:
                    # For MAP dispersion, we still need full matrices (can optimize later)
                    # Build combined Y and mu for estimate_dispersion_map
                    Y = np.empty((n_control + n_group, n_genes), dtype=np.float64)
                    Y[:n_control, :] = control_cache.control_matrix
                    Y[n_control:, :] = Y_pert_dense
                    
                    # Compute full mu matrix for MAP
                    offset_combined = np.concatenate([control_cache.control_offset, perturbation_offset])
                    eta = (
                        beta0_all[None, :] + 
                        perturbation_indicator[:, None] * beta1_all[None, :] + 
                        offset_combined[:, None]
                    )
                    np.clip(eta, -30, 30, out=eta)
                    mu = np.exp(eta)
                    del eta
                    mu[:, ~batch_result.converged] = 1e-10
                    np.maximum(mu, 1e-10, out=mu)
                    
                    max_disp = max(10.0, float(subset_n))
                    result["dispersion"] = estimate_dispersion_map(Y, mu, trend, max_disp=max_disp)
                    del Y, mu  # Free large matrices
                else:
                    result["dispersion"] = shrink_dispersions(result["dispersion_raw"], trend)
                
                # ================================================================
                # Recompute SE using batched processing (memory-optimized)
                # ================================================================
                final_disp = result["dispersion"]
                recomputed_se = _compute_se_batched(
                    Y_control=control_cache.control_matrix,
                    Y_pert=Y_pert_dense,
                    control_offset=control_cache.control_offset,
                    pert_offset=perturbation_offset,
                    beta0=beta0_all,
                    beta1=beta1_all,
                    dispersion=final_disp,
                    gene_batch_size=5000,
                    se_method=se_method,
                )
                
                # Update result with recomputed SE
                result["se"] = np.where(valid_converged_mask, recomputed_se, np.nan)
                
                # Recompute Wald statistic and p-value with new SE
                coefs = batch_result.coef[:, 1]
                statistics = np.divide(
                    coefs, recomputed_se, 
                    out=np.full(n_genes, np.nan), 
                    where=valid_converged_mask
                )
                pvalues = np.where(
                    valid_converged_mask,
                    2.0 * norm.sf(np.abs(statistics)),  # Use sf() for numerical stability
                    np.nan
                )
                result["statistic"] = statistics
                result["pvalue"] = pvalues
        else:
            result["dispersion_trend"] = result["dispersion_raw"].copy()
            result["dispersion"] = result["dispersion_raw"].copy()
        
        result["logfc_raw"] = result["logfc"].copy()
        if lfc_shrinkage_type == "apeglm":
            # Full NB-GLM re-fitting with Cauchy prior (matches PyDESeq2)
            # Build mle_coef matrix (n_params, n_genes) from batch_result
            n_params = 2  # Intercept + perturbation
            mle_coef = np.full((n_params, n_genes), np.nan, dtype=np.float64)
            mle_coef[0, :] = batch_result.coef[:, 0]  # Intercept
            mle_coef[1, :] = batch_result.coef[:, 1]  # LFC
            
            # Build combined count matrix (control + perturbation)
            if sp.issparse(group_matrix):
                Y_pert = group_matrix.toarray()
            else:
                Y_pert = np.asarray(group_matrix, dtype=np.float64)
            counts_combined = np.vstack([control_cache.control_matrix, Y_pert])
            
            # Build combined size factors
            sf_combined = np.concatenate([
                np.exp(control_cache.control_offset),
                subset_size_factors_group
            ])
            
            # Build design matrix for combined data
            design_combined = np.zeros((n_control + group_n, 2), dtype=np.float64)
            design_combined[:, 0] = 1.0  # Intercept
            design_combined[n_control:, 1] = 1.0  # Perturbation indicator
            
            # Call full apeGLM shrinkage
            # NOTE: PyDESeq2's lfc_shrink does NOT use min_mu during shrinkage
            shrunk_coef, shrunk_se_arr, shrink_converged = shrink_lfc_apeglm(
                counts=counts_combined,
                design_matrix=design_combined,
                size_factors=sf_combined,
                dispersion=result["dispersion"],
                mle_coef=mle_coef,
                mle_se=result["se"],
                shrink_index=1,  # LFC is at index 1
                prior_scale=None,
                n_jobs=1,
                min_mu=0.0,  # No min_mu - match PyDESeq2's lfc_shrink behavior
            )
            result["logfc"] = shrunk_coef[1, :]  # Shrunken LFC
            result["effect"] = result["logfc"].copy()
            result["se"] = shrunk_se_arr
        else:
            result["effect"] = result["logfc"].copy()
        
        return result
    # -------------------------------------------------------------------------

    n_groups = len(candidates)
    
    # Determine output path first (needed for checkpoint)
    output_path = resolve_output_path(
        path,
        suffix="nb_glm",
        output_dir=output_dir,
        data_name=data_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")
    
    # Handle resume logic
    if resume:
        candidates_to_run, completed_labels, failed_labels = _get_resumable_candidates(
            checkpoint_path, output_path, candidates, retry_failed=True
        )
        # If all candidates are completed, load and return the existing result
        if len(candidates_to_run) == 0 and output_path.exists():
            logger.info("All perturbations already completed. Loading existing result...")
            return _load_existing_nb_glm_result(
                output_path=output_path,
                candidates=candidates,
                gene_symbols=gene_symbols,
                perturbation_column=perturbation_column,
                control_label=control_label,
                corr_method=corr_method,
            )
    else:
        candidates_to_run = candidates
        completed_labels = []
        failed_labels = []
    
    # Determine checkpoint interval
    eff_checkpoint_interval = _get_checkpoint_interval(len(candidates), checkpoint_interval)
    
    # Create index mappings
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}
    
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
        intercept_memmap = _create_memmap("intercept", np.float64)  # MLE intercept for shrink_lfc
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
        # Determine number of parallel workers with memory-awareness
        # For small n_groups, run sequentially to avoid joblib overhead
        # (profiling shows joblib.sleep takes 24s for 2 perturbations)
        cpu_count = os.cpu_count() or 1
        if max_workers is not None:
            # Explicit max_workers takes precedence
            effective_n_jobs = min(max_workers, cpu_count)
        elif n_jobs is None or n_jobs == 0:
            effective_n_jobs = cpu_count
        elif n_jobs == -1:
            effective_n_jobs = cpu_count
        elif n_jobs < 0:
            effective_n_jobs = max(1, cpu_count + n_jobs + 1)
        else:
            effective_n_jobs = min(n_jobs, cpu_count)
        effective_n_jobs = max(1, effective_n_jobs)
        
        # Save original requested workers for auto-detection logic
        # (before memory-based reduction)
        requested_n_jobs = effective_n_jobs
        
        # Memory-aware worker limiting: Adaptive estimation based on dataset statistics
        # Compute group size statistics without loading the full matrix
        # Use numpy unique with counts since labels is a numpy array
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        group_sizes = dict(zip(unique_labels, label_counts))
        pert_group_sizes = [group_sizes.get(g, 0) for g in candidates]
        max_group_size = max(pert_group_sizes) if pert_group_sizes else 1
        avg_group_size = max(1, (n_cells_total - control_n) // max(1, n_groups))
        
        # Use p95 group size for realistic estimate (avoids over-conservative from outliers)
        if len(pert_group_sizes) >= 10:
            p95_group_size = int(np.percentile(pert_group_sizes, 95))
            use_group_size = p95_group_size
        else:
            use_group_size = max_group_size
        
        # Decide early if we can use control cache (needed for memory estimation)
        can_use_cache_early = (
            use_control_cache and 
            len(covariates) == 0 and
            size_factor_scope == "global"
        )
        
        # =====================================================================
        # AUTO-DETECTION: Enable freeze_control for large datasets
        # =====================================================================
        # When freeze_control=None (default), auto-enable if:
        # 1. Control matrix serialization would severely limit workers (<4)
        # 2. Required settings are met (dispersion_scope='global', shrink_dispersion=True)
        #
        # This provides optimal parallelization without user intervention.
        
        if freeze_control is None:
            # Check if settings are compatible with frozen control
            settings_compatible = (
                can_use_cache_early and
                dispersion_scope == "global" and
                shrink_dispersion
            )
            
            if settings_compatible:
                # Estimate per-worker memory in standard mode (matching actual formula below)
                control_matrix_mb_est = control_n * n_genes * 8 / 1e6
                labels_mb_est = n_cells_total * 50 / 1e6  # ~50 bytes per string
                size_factors_mb_est = n_cells_total * 8 / 1e6
                work_arrays_mb_est = (control_n + use_group_size) * n_genes * 8 * 4 / 1e6
                serialized_args_mb_est = (control_matrix_mb_est + labels_mb_est + size_factors_mb_est) * 2.5
                per_worker_standard_mb = serialized_args_mb_est + work_arrays_mb_est + 2000
                
                # Get available memory
                if memory_limit_gb is not None:
                    available_mb = memory_limit_gb * 1000
                else:
                    try:
                        import psutil
                        available_mb = psutil.virtual_memory().available / 1e6
                    except ImportError:
                        available_mb = 8000.0
                
                # How many workers could we run without frozen control?
                base_memory_mb_est = control_matrix_mb_est + 1000
                usable_mb = available_mb * 0.8
                remaining_mb = max(usable_mb - base_memory_mb_est, per_worker_standard_mb)
                max_workers_standard = max(1, int(remaining_mb / per_worker_standard_mb))
                
                # Auto-enable if standard mode would limit to <4 workers AND user wants >4
                # Use requested_n_jobs (original request) not effective_n_jobs (may be reduced)
                if max_workers_standard < 4 and requested_n_jobs >= 4:
                    freeze_control = True
                    logger.info(
                        f"Auto-enabling freeze_control: standard mode would limit to {max_workers_standard} workers "
                        f"(control: {control_n:,} cells × {n_genes:,} genes = {control_matrix_mb_est:.0f} MB, "
                        f"per_worker: {per_worker_standard_mb:.0f} MB). "
                        f"Frozen control enables ~{requested_n_jobs} workers."
                    )
                else:
                    freeze_control = False
            else:
                freeze_control = False
        
        # Check if frozen control mode is valid (after auto-detection)
        can_use_frozen_control = (
            freeze_control and 
            can_use_cache_early and 
            dispersion_scope == "global" and 
            shrink_dispersion
        )
        
        # Memory estimation for joblib parallel execution
        # IMPORTANT: joblib's loky backend serializes (pickles) all function arguments
        # for each worker process. This means control_matrix is copied to each worker,
        # not shared via copy-on-write as it would be with fork().
        #
        # What each worker receives via pickle:
        # 1. control_cache.control_matrix: (control_n × n_genes) × 8 bytes
        #    OR with freeze_control=True: frozen stats only (~1MB total)
        # 2. labels array: n_cells_total strings (pickled as object array)
        # 3. size_factors: n_cells_total × 8 bytes
        # 4. Other small arrays (control_offset, etc.)
        #
        # What each worker allocates during execution:
        # 1. group_matrix (loaded from disk, small: ~200 cells × n_genes)
        # 2. Intermediate arrays for SE recomputation, mu, etc.
        # 3. When dispersion_scope='per_comparison': full Y and mu matrices
        #
        # Pickle overhead is typically 1.5-2× the raw array size due to protocol
        # serialization and Python object overhead.
        
        if can_use_frozen_control:
            # FROZEN CONTROL MODE: Optimized memory estimation
            # 
            # What each worker receives:
            # 1. control_cache with frozen stats (~1MB): W_sum, Wz_sum, etc.
            # 2. perturbation_boundaries dict: ~100KB for 10K perturbations
            # 3. size_factors: still full array (~16MB for 2M cells) - could optimize later
            # 4. labels: still needed for fallback, but could use boundaries only
            #
            # What each worker allocates:
            # 1. group_matrix from disk: (group_size × n_genes) × 8 bytes
            # 2. Work arrays: mu, W, z for perturbation cells only
            # 3. Result arrays: ~n_genes × 8 bytes × 10 arrays
            
            # Frozen stats: 6 arrays of shape (n_genes,)
            frozen_stats_mb = n_genes * 8 * 6 / 1e6  # W_sum, Wz_sum, mu_sum, etc.
            
            # Cache metadata (beta_intercept, dispersion, pts_rest, etc.)
            cache_metadata_mb = n_genes * 8 * 5 / 1e6
            
            # Perturbation boundaries: ~50 bytes per perturbation
            boundaries_mb = n_groups * 50 / 1e6
            
            # Labels and size_factors (still passed but could be optimized)
            labels_mb = n_cells_total * 50 / 1e6
            size_factors_mb = n_cells_total * 8 / 1e6
            
            # Serialized args with reduced pickle overhead (simpler objects)
            control_matrix_mb_for_pickle = frozen_stats_mb + cache_metadata_mb + boundaries_mb
            serialized_args_mb = (control_matrix_mb_for_pickle + labels_mb + size_factors_mb) * 2.0
            
            # Work arrays: only perturbation cells (much smaller!)
            # mu_pert, W_pert, z_pert, Y_pert_valid for fitting
            work_arrays_mb = use_group_size * n_genes * 8 * 5 / 1e6
            
            # Result arrays per worker
            result_arrays_mb = n_genes * 8 * 12 / 1e6  # 12 result fields
            
            # Reduced Python overhead for frozen control (simpler computation)
            python_overhead_mb = 500  # 500 MB instead of 2 GB
            
            per_worker_mb = serialized_args_mb + work_arrays_mb + result_arrays_mb + python_overhead_mb
            
            logger.debug(
                f"Frozen control memory estimate: serialized={serialized_args_mb:.1f}MB, "
                f"work_arrays={work_arrays_mb:.1f}MB, per_worker={per_worker_mb:.1f}MB"
            )
        else:
            control_matrix_mb_for_pickle = control_n * n_genes * 8 / 1e6  # float64
            
            # Context-aware work arrays based on dispersion mode
            if can_use_cache_early and dispersion_scope == "global":
                # Global dispersion: skip MoM/trend, but still need SE recomputation arrays
                work_arrays_mb = (control_n + use_group_size) * n_genes * 8 * 4 / 1e6
            else:
                # Per-comparison: need full Y and mu matrices for MAP dispersion
                work_arrays_mb = (control_n + use_group_size) * n_genes * 8 * 6 / 1e6
            
            # Standard mode: full labels and size_factors arrays
            labels_mb = n_cells_total * 50 / 1e6  # ~50 bytes per string (pickled)
            size_factors_mb = n_cells_total * 8 / 1e6
            
            # Pickle overhead is 2-3× for complex objects due to Python object structure
            serialized_args_mb = (control_matrix_mb_for_pickle + labels_mb + size_factors_mb) * 2.5
            
            # Per-worker total: serialized args + work arrays + Python/process overhead
            per_worker_mb = serialized_args_mb + work_arrays_mb + 2000
        
        # For base memory and logging
        control_matrix_mb = control_n * n_genes * 8 / 1e6
        
        # Base memory: parent process + one copy of control matrix + misc arrays
        base_memory_mb = control_matrix_mb + 1000  # Parent process overhead
        
        # Calculate available memory
        if memory_limit_gb is not None:
            available_mb = memory_limit_gb * 1000
        else:
            try:
                import psutil
                available_mb = psutil.virtual_memory().available / 1e6
            except ImportError:
                available_mb = 8000.0  # 8 GB default
        
        # Reserve 20% headroom for safety
        usable_mb = available_mb * 0.8
        remaining_mb = max(usable_mb - base_memory_mb, per_worker_mb)
        max_workers_by_memory = max(1, int(remaining_mb / per_worker_mb))
        
        # Dataset-size-aware caps
        full_matrix_gb = (control_n + avg_group_size * n_groups) * n_genes * 8 / 1e9
        if full_matrix_gb < 1.0:
            # Tiny dataset: cap workers to reduce parallelization overhead
            max_workers_by_size = max(4, n_groups // 2)
        else:
            max_workers_by_size = n_groups
        
        # Apply all constraints
        # Use candidates_to_run (not n_groups) for worker limit since that's what we're actually running
        n_to_run = len(candidates_to_run)
        memory_limited_workers = min(max_workers_by_memory, max_workers_by_size, n_to_run)
        
        if max_workers is None and memory_limited_workers < effective_n_jobs:
            # Only apply memory limiting if max_workers not explicitly set
            # Determine limiting factor for logging
            if memory_limited_workers == n_to_run and n_to_run < max_workers_by_memory and n_to_run < max_workers_by_size:
                limit_reason = "perturbation_count"
            elif memory_limited_workers == max_workers_by_memory:
                limit_reason = "memory"
            elif memory_limited_workers == max_workers_by_size:
                limit_reason = "small_dataset"
            else:
                limit_reason = "perturbation_count"
            logger.info(
                f"Memory-aware limiting: {effective_n_jobs} -> {memory_limited_workers} workers "
                f"(reason: {limit_reason}, base: {base_memory_mb:.0f}MB, per_worker: {per_worker_mb:.0f}MB, "
                f"available: {available_mb:.0f}MB, full_matrix: {full_matrix_gb:.1f}GB)"
            )
            effective_n_jobs = memory_limited_workers
        effective_n_jobs = max(1, effective_n_jobs)
        
        # For small number of perturbations to run, run sequentially to avoid overhead
        use_parallel = n_to_run >= 4 and effective_n_jobs > 1
        
        # Decide whether to use control cache optimization
        # Control cache is used when: no covariates, use_control_cache=True, global SF
        # Per-comparison size factors require fresh computation per comparison
        can_use_cache = (
            use_control_cache and 
            len(covariates) == 0 and
            size_factor_scope == "global"
        )
        
        # =====================================================================
        # Early memory check: determine if we need streaming mode
        # =====================================================================
        # For very large datasets (e.g., Replogle-GW-k562: 2M cells × 8K genes),
        # loading the full matrix would exceed memory. Check this ONCE before
        # any full matrix loads (full_X for per-comparison SF, all_cell_matrix
        # for global dispersion).
        estimated_matrix_gb = n_cells_total * n_genes * 8 / 1e9  # float64
        if memory_limit_gb is not None:
            effective_memory_limit_gb = min(available_mb / 1000, memory_limit_gb)
        else:
            effective_memory_limit_gb = available_mb / 1000
        memory_budget_gb = max_dense_fraction * effective_memory_limit_gb
        use_streaming_mode = estimated_matrix_gb > memory_budget_gb
        
        if use_streaming_mode:
            logger.info(
                f"Large dataset detected: {n_cells_total:,} cells × {n_genes:,} genes = "
                f"{estimated_matrix_gb:.1f} GB > {memory_budget_gb:.1f} GB budget. "
                f"Using streaming mode for memory efficiency."
            )
        
        # For per-comparison size factors, we need the full count matrix
        # Skip if streaming mode - worker will fall back to global SF
        if size_factor_scope == "per_comparison":
            if use_streaming_mode:
                logger.warning(
                    f"Dataset too large ({estimated_matrix_gb:.1f} GB) for per-comparison "
                    f"size factors (requires loading full matrix). "
                    f"Falling back to global size factors for memory efficiency."
                )
                full_X = None  # Worker will use global SF
            else:
                logger.info("Using per-comparison size factors for PyDESeq2 compatibility...")
                backed = read_backed(path)
                try:
                    full_X = backed.X[:]
                    if sp.issparse(full_X):
                        full_X = sp.csr_matrix(full_X, dtype=np.float64)
                    else:
                        full_X = np.asarray(full_X, dtype=np.float64)
                finally:
                    backed.file.close()
        else:
            full_X = None
        
        # Precompute control statistics once if using cache
        control_cache = None
        if can_use_cache:
            # Validate freeze_control requirements (for explicit freeze_control=True)
            if freeze_control:
                if dispersion_scope != "global":
                    raise ValueError(
                        "freeze_control=True requires dispersion_scope='global'. "
                        "Per-comparison dispersion needs raw control matrix."
                    )
                if not shrink_dispersion:
                    raise ValueError(
                        "freeze_control=True requires shrink_dispersion=True. "
                        "Global dispersion must be computed for frozen control mode."
                    )
                # Note: Auto-detected freeze_control already logged in auto-detection block
            
            logger.info("Precomputing control cell statistics for cache optimization...")
            control_offset = offset[control_mask]
            control_cache = precompute_control_statistics(
                control_matrix=control_matrix,
                control_offset=control_offset,
                max_iter=max_iter,
                tol=tol,
                min_mu=min_mu,
                dispersion_method="moments",  # Fast initial estimate
                global_size_factors=size_factors,  # Store global SF in cache
                freeze_control=freeze_control,  # Enable frozen control mode
            )
            
            # Precompute global dispersion if dispersion_scope='global'
            if dispersion_scope == "global" and shrink_dispersion and use_map_dispersion:
                logger.info("Precomputing global dispersion trend (dispersion_scope='global')...")
                
                if use_streaming_mode:
                    # Use path-based streaming for large datasets
                    # Reads chunks from disk, never loads full matrix
                    control_cache = precompute_global_dispersion_from_path(
                        path=path,
                        control_cache=control_cache,
                        all_cell_offset=offset,
                        fit_type="parametric",
                    )
                else:
                    # Load all cells for global dispersion estimation
                    backed = read_backed(path)
                    try:
                        all_cell_matrix = backed.X[:]
                        if sp.issparse(all_cell_matrix):
                            all_cell_matrix = sp.csr_matrix(all_cell_matrix, dtype=np.float64)
                        else:
                            all_cell_matrix = np.asarray(all_cell_matrix, dtype=np.float64)
                    finally:
                        backed.file.close()
                    
                    # Compute global dispersion using all cells
                    # Use fast_mode=True for speed (MoM + trend shrinkage instead of MAP)
                    # Memory-adaptive: switches to streaming if matrix too large
                    control_cache = precompute_global_dispersion(
                        control_cache=control_cache,
                        all_cell_matrix=all_cell_matrix,
                        all_cell_offset=offset,
                        n_grid=25,
                        fit_type="parametric",
                        fast_mode=True,  # ~50× faster than full MAP
                        max_dense_fraction=max_dense_fraction,
                        memory_limit_gb=memory_limit_gb,
                    )
                    del all_cell_matrix  # Free memory
                    gc.collect()  # Force garbage collection before spawning workers
                
                logger.info(f"Global dispersion precomputed: prior_var={control_cache.global_disp_prior_var:.4f}")
        
        # Log progress info
        if resume and completed_labels:
            logger.info(f"Fitting {n_to_run}/{n_groups} remaining perturbations with {effective_n_jobs} workers...")
        else:
            logger.info(f"Fitting {n_groups} perturbations with {effective_n_jobs} workers...")
        
        # Track completed labels during this run
        newly_completed = list(completed_labels)  # Start with already completed
        newly_failed = list(failed_labels)
        n_processed = 0
        
        # Helper function to write result to memmap
        def _write_result_to_memmap(res: dict, label: str) -> None:
            idx = candidate_to_idx[label]
            effect_memmap[idx, :] = res["effect"]
            statistic_memmap[idx, :] = res["statistic"]
            pvalue_memmap[idx, :] = res["pvalue"]
            logfc_memmap[idx, :] = res["logfc"]
            logfc_raw_memmap[idx, :] = res["logfc_raw"]
            intercept_memmap[idx, :] = res["intercept"]  # MLE intercept for shrink_lfc
            se_memmap[idx, :] = res["se"]
            pts_memmap[idx, :] = res["pts"]
            pts_rest_memmap[idx, :] = res["pts_rest"]
            dispersion_memmap[idx, :] = res["dispersion"]
            dispersion_raw_memmap[idx, :] = res["dispersion_raw"]
            dispersion_trend_memmap[idx, :] = res["dispersion_trend"]
            mean_memmap[idx, :] = res["mean"]
            iter_memmap[idx, :] = res["iterations"]
            convergence_memmap[idx, :] = res["converged"]
        
        # Helper to save checkpoint
        def _save_checkpoint() -> None:
            checkpoint_data = {
                "total": n_groups,
                "completed": newly_completed,
                "failed": newly_failed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "nb_glm",
                "control_label": control_label,
            }
            _write_checkpoint_atomic(checkpoint_path, checkpoint_data)
        
        # Run fitting with progress tracking
        with _create_progress_context(n_to_run, "NB-GLM DE", verbose) as pbar:
            if use_parallel:
                # Use joblib.Parallel with loky backend for true process-based parallelism
                # This avoids GIL contention that limits ThreadPoolExecutor performance
                if can_use_cache:
                    results = Parallel(
                        n_jobs=effective_n_jobs,
                        backend="loky",
                        prefer="processes",
                        return_as="generator",  # Stream results for progress updates
                    )(
                        delayed(_fit_perturbation_worker_cached)(
                            group_idx=candidate_to_idx[label],
                            label=label,
                            path=path,
                            labels=labels,
                            control_cache=control_cache,
                            size_factors=size_factors,
                            n_genes=n_genes,
                            min_cells_expressed=min_cells_expressed,
                            min_total_count=min_total_count,
                            max_iter=max_iter,
                            tol=tol,
                            min_mu=min_mu,
                            dispersion_method=dispersion_method,
                            shrink_dispersion=shrink_dispersion,
                            use_map_dispersion=use_map_dispersion,
                            lfc_shrinkage_type=lfc_shrinkage_type,
                            se_method=se_method,
                            perturbation_boundaries=perturbation_boundaries,
                        )
                        for label in candidates_to_run
                    )
                else:
                    results = Parallel(
                        n_jobs=effective_n_jobs,
                        backend="loky",
                        prefer="processes",
                        return_as="generator",
                    )(
                        delayed(_fit_perturbation_worker)(
                            group_idx=candidate_to_idx[label],
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
                            full_X=full_X,
                            per_comparison_sf=(size_factor_scope == "per_comparison"),
                            se_method=se_method,
                            perturbation_boundaries=perturbation_boundaries,
                        )
                        for label in candidates_to_run
                    )
                
                # Process results as they stream in
                for idx, res in enumerate(results):
                    label = candidates_to_run[idx]
                    try:
                        _write_result_to_memmap(res, label)
                        newly_completed.append(label)
                        logger.debug(f"Completed perturbation: {label}")
                    except Exception as e:
                        logger.error(f"Failed perturbation {label}: {e}")
                        newly_failed.append(label)
                    
                    n_processed += 1
                    pbar.update(1)
                    
                    # Save checkpoint periodically
                    if n_processed % eff_checkpoint_interval == 0:
                        _save_checkpoint()
            else:
                # Run sequentially
                for label in candidates_to_run:
                    group_idx = candidate_to_idx[label]
                    try:
                        if can_use_cache:
                            res = _fit_perturbation_worker_cached(
                                group_idx=group_idx,
                                label=label,
                                path=path,
                                labels=labels,
                                control_cache=control_cache,
                                size_factors=size_factors,
                                n_genes=n_genes,
                                min_cells_expressed=min_cells_expressed,
                                min_total_count=min_total_count,
                                max_iter=max_iter,
                                tol=tol,
                                min_mu=min_mu,
                                dispersion_method=dispersion_method,
                                shrink_dispersion=shrink_dispersion,
                                use_map_dispersion=use_map_dispersion,
                                lfc_shrinkage_type=lfc_shrinkage_type,
                                se_method=se_method,
                                perturbation_boundaries=perturbation_boundaries,
                            )
                        else:
                            res = _fit_perturbation_worker(
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
                                min_mu=min_mu,
                                poisson_init_iter=poisson_init_iter,
                                dispersion_method=dispersion_method,
                                global_dispersion=global_dispersion,
                                shrink_dispersion=shrink_dispersion,
                                use_map_dispersion=use_map_dispersion,
                                lfc_shrinkage_type=lfc_shrinkage_type,
                                pts_rest_shared=pts_rest_shared,
                                full_X=full_X,
                                per_comparison_sf=(size_factor_scope == "per_comparison"),
                                se_method=se_method,
                                perturbation_boundaries=perturbation_boundaries,
                            )
                        _write_result_to_memmap(res, label)
                        newly_completed.append(label)
                        logger.debug(f"Completed perturbation: {label}")
                    except Exception as e:
                        logger.error(f"Failed perturbation {label}: {e}")
                        newly_failed.append(label)
                    
                    n_processed += 1
                    pbar.update(1)
                    
                    # Save checkpoint periodically
                    if n_processed % eff_checkpoint_interval == 0:
                        _save_checkpoint()
        
        # Final checkpoint save
        _save_checkpoint()
        logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")
        if newly_failed:
            logger.warning(f"Failed {len(newly_failed)} perturbations: {newly_failed[:5]}{'...' if len(newly_failed) > 5 else ''}")

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
        intercept_matrix = np.array(intercept_memmap)  # MLE intercept for shrink_lfc
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

    # Store ln-scale raw values BEFORE log2 conversion (for shrink_lfc post-hoc)
    logfc_raw_ln_matrix = logfc_raw_matrix.copy()
    se_ln_matrix = se_matrix.copy()

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
    adata.layers["logfoldchange_raw_ln"] = logfc_raw_ln_matrix  # Always ln-scale for shrink_lfc
    adata.layers["intercept"] = intercept_matrix  # MLE intercept (ln-scale) for shrink_lfc
    adata.layers["standard_error"] = se_matrix
    adata.layers["standard_error_ln"] = se_ln_matrix  # Always ln-scale for shrink_lfc
    adata.layers["dispersion"] = dispersion_matrix
    adata.layers["dispersion_raw"] = dispersion_raw_matrix
    adata.layers["dispersion_trend"] = dispersion_trend_matrix
    adata.layers["converged"] = convergence_matrix.astype(np.float32)
    adata.layers["iterations"] = iter_matrix.astype(np.float32)
    adata.layers["pts"] = pts_matrix
    adata.layers["pts_rest"] = pts_rest_matrix
    adata.uns["lfc_base"] = lfc_base  # Store for downstream tools
    adata.uns["method"] = "nb_glm"
    adata.uns["fit_method"] = "independent"
    adata.uns["control_label"] = control_label
    adata.uns["perturbation_column"] = perturbation_column
    adata.uns["covariates"] = covariates
    adata.uns["size_factors"] = size_factors
    adata.uns["original_dataset_path"] = str(path)  # For shrink_lfc to reload data
    adata.uns["size_factor_method"] = size_factor_method
    adata.uns["size_factor_scope"] = size_factor_scope
    adata.uns["dispersion_scope"] = dispersion_scope

    # Store profiling results or "NA" for production
    if profiling and profiler is not None:
        profiler.stop("fit")
        profiler.snapshot("fit_end")
        profiler.stop("total")
        profiler.stop_sampling()
        stats = profiler.get_stats()
        adata.uns["profiling"] = {
            "profiling_enabled": True,
            "fit_seconds": stats.get("timing", {}).get("sections", {}).get("fit", {}).get("seconds", 0.0),
            "fit_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
            "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
        }
    else:
        adata.uns["profiling"] = "NA"

    # output_path already resolved earlier for checkpoint
    adata.write(output_path)
    
    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors

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
    
    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format:
        _write_rank_genes_groups_hdf5(output_path, result)
        # Reload to pick up the new uns structure
        result.result = AnnData(output_path)
    
    return result


def wilcoxon_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    chunk_size: int | None = None,
    tie_correct: bool = True,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    n_jobs: int | None = None,
    verbose: bool = False,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    scanpy_format: bool = False,
) -> RankGenesGroupsResult:
    """Perform a Wilcoxon rank-sum (Mann-Whitney U) test for each gene.

    Input data **must already be library-size normalised and log-transformed**.
    The function operates directly on the provided matrix without additional
    preprocessing. As a safeguard, the first sparse chunk is inspected and a
    warning is emitted if the data appear to be raw counts (integer or
    count-like floats), encouraging explicit preprocessing upstream.
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        normalised, log-transformed data.
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
        Genes below this threshold are assigned p-value=1 and effect_size=0.
    chunk_size
        Number of genes to process per chunk (memory vs. speed tradeoff). Smaller
        values stream more, reducing peak memory at the cost of additional I/O.
    tie_correct
        Whether to apply tie correction to the U statistic. Default True for
        more accurate p-values when ties are present in the data.
    corr_method
        Method for p-value correction: "benjamini-hochberg" or "bonferroni".
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "wilcoxon" suffix.
    n_jobs
        Number of parallel workers for computing statistics across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
    verbose
        If True, show a progress bar for gene chunk processing. Requires tqdm.
    resume
        If True, attempt to resume from a previous interrupted run. Reads the
        checkpoint file to determine which gene chunks have already been
        completed and skips them.
    checkpoint_interval
        Number of gene chunks between checkpoint saves. If None, auto-determined
        based on dataset size. The checkpoint file `<output>.progress.json` is
        written atomically to prevent corruption.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.

    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.
    """

    path = resolve_data_path(data)
    backed = read_backed(path)
    try:
        # Calculate adaptive gene chunk_size if not provided
        # Wilcoxon iterates over genes (columns), so use gene chunk calculator
        if chunk_size is None:
            chunk_size = calculate_optimal_gene_chunk_size(backed.n_obs, backed.n_vars)
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
    
    # Determine output path and checkpoint path
    output_path = resolve_output_path(path, suffix="wilcoxon", output_dir=output_dir, data_name=data_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")
    
    # For wilcoxon, we track gene chunk progress (not perturbation progress)
    # Resume logic: read checkpoint to get last completed gene chunk
    last_completed_chunk = -1
    if resume and checkpoint_path.exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        if checkpoint is not None:
            last_completed_chunk = checkpoint.get("last_gene_chunk", -1)
            logger.info(f"Resuming from gene chunk {last_completed_chunk + 1}")
    
    # Determine checkpoint interval (number of gene chunks between saves)
    n_gene_chunks = (n_genes + chunk_size - 1) // chunk_size
    eff_checkpoint_interval = _get_checkpoint_interval(n_gene_chunks, checkpoint_interval)
    
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

            # Track progress
            current_chunk = 0
            n_chunks_processed = 0
            
            # Helper to save checkpoint
            def _save_wilcoxon_checkpoint(chunk_idx: int) -> None:
                checkpoint_data = {
                    "total_gene_chunks": n_gene_chunks,
                    "last_gene_chunk": chunk_idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": "wilcoxon",
                    "control_label": control_label,
                }
                _write_checkpoint_atomic(checkpoint_path, checkpoint_data)

            with _create_progress_context(n_gene_chunks, "Wilcoxon DE (gene chunks)", verbose) as pbar:
                for slc, block in iter_matrix_chunks(
                    backed, axis=1, chunk_size=chunk_size, convert_to_dense=False
                ):
                    # Skip already processed chunks on resume
                    if current_chunk <= last_completed_chunk:
                        current_chunk += 1
                        pbar.update(1)
                        continue
                    
                    if not dtype_checked:
                        if not sp.issparse(block):
                            raise ValueError(
                                "wilcoxon_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                            )
                        _warn_if_count_like(block)
                        dtype_checked = True

                    csr_block = sp.csr_matrix(block, dtype=np.float64)
                    n_chunk_genes = csr_block.shape[1]

                    # ===== OPTIMIZED BATCH PROCESSING =====
                    # 1. Extract control and perturbation data once
                    control_values = csr_block[control_mask, :]
                    control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()
                    control_mean = (
                        np.asarray(control_values.mean(axis=0)).ravel()
                        if control_values.nnz
                        else np.zeros(n_chunk_genes, dtype=np.float64)
                    )
                    control_mean_expm1 = np.expm1(control_mean) + 1e-9
                    control_pts = np.divide(
                        control_expr,
                        control_n,
                        out=np.zeros_like(control_expr, dtype=float),
                        where=control_n > 0,
                    )
                    chunk_gene_indices = np.arange(slc.start, slc.stop)

                    # 2. Pre-compute perturbation masks and expression counts
                    pert_expr_counts = []
                    pert_means = []
                    pert_n_cells = []
                    for label in candidates:
                        mask = pert_masks[label]
                        group_values = csr_block[mask, :]
                        n_pert_cells = group_values.shape[0]
                        pert_n_cells.append(n_pert_cells)
                        group_expr = np.asarray(group_values.getnnz(axis=0)).ravel()
                        pert_expr_counts.append(group_expr)
                        group_mean = (
                            np.asarray(group_values.mean(axis=0)).ravel()
                            if group_values.nnz
                            else np.zeros(n_chunk_genes, dtype=np.float64)
                        )
                        pert_means.append(group_mean)

                    # 3. Determine valid genes per perturbation
                    valid_masks = []
                    for idx, label in enumerate(candidates):
                        group_expr = pert_expr_counts[idx]
                        total_expr = control_expr + group_expr
                        low_expr = (control_expr < min_cells_expressed) & (
                            group_expr < min_cells_expressed
                        )
                        valid = (total_expr >= min_cells_expressed) & ~low_expr
                        valid_masks.append(valid)

                    # 4. Find union of all valid genes (to minimize dense conversion)
                    any_valid = np.zeros(n_chunk_genes, dtype=bool)
                    for valid in valid_masks:
                        any_valid |= valid
                    valid_gene_indices = np.where(any_valid)[0]
                    n_valid_genes = len(valid_gene_indices)

                    # 5. Initialize output arrays for this chunk
                    chunk_u = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                    chunk_z = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                    chunk_p = np.ones((n_groups, n_chunk_genes), dtype=np.float64)
                    chunk_effect = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                    chunk_lfc = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                    chunk_pts = np.zeros((n_groups, n_chunk_genes), dtype=np.float32)
                    chunk_pts_rest = np.zeros((n_groups, n_chunk_genes), dtype=np.float32)

                    if n_valid_genes > 0:
                        # 6. Convert to dense ONCE for ALL cells and valid genes
                        # This is the key optimization - avoid per-perturbation sparse ops
                        all_cells_dense = csr_block[:, valid_gene_indices].toarray().astype(np.float64)
                        
                        # Pre-extract control rows (constant across all perturbations)
                        control_dense = all_cells_dense[control_mask, :]
                        
                        # 7. Pre-allocate output arrays for valid genes
                        valid_u = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                        valid_z = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                        valid_p = np.ones((n_groups, n_valid_genes), dtype=np.float64)
                        valid_effect = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                        
                        # 8. Process each perturbation using fast Numba kernel
                        for idx, label in enumerate(candidates):
                            mask = pert_masks[label]
                            pert_dense = all_cells_dense[mask, :]
                            valid_genes_arr = valid_masks[idx][valid_gene_indices]
                            
                            # Call optimized single-perturbation kernel
                            _wilcoxon_sparse_batch_numba(
                                control_dense,
                                pert_dense,
                                valid_genes_arr,
                                tie_correct,
                                _ZERO_PARTITION_THRESHOLD,
                                valid_u[idx],
                                valid_z[idx],
                                valid_p[idx],
                                valid_effect[idx],
                            )
                        
                        # 9. Map results back to full chunk gene indices
                        for idx in range(n_groups):
                            chunk_u[idx, valid_gene_indices] = valid_u[idx]
                            chunk_z[idx, valid_gene_indices] = valid_z[idx]
                            chunk_p[idx, valid_gene_indices] = valid_p[idx]
                            chunk_effect[idx, valid_gene_indices] = valid_effect[idx]

                    # 10. Compute LFC and pts for all perturbations (vectorized)
                    for idx, label in enumerate(candidates):
                        group_expr = pert_expr_counts[idx]
                        group_mean = pert_means[idx]
                        n_pert = pert_n_cells[idx]
                        valid = valid_masks[idx]
                        
                        # pts
                        pts = np.divide(
                            group_expr,
                            float(n_pert),
                            out=np.zeros_like(group_expr, dtype=float),
                            where=n_pert > 0,
                        )
                        pts = np.where(valid, pts, 0.0)
                        pts_rest = np.where(valid, control_pts, 0.0)
                        
                        # Log2 fold change
                        lfc = np.log2((np.expm1(group_mean) + 1e-9) / control_mean_expm1)
                        lfc = np.where(valid, lfc, 0.0)
                        
                        chunk_pts[idx] = pts
                        chunk_pts_rest[idx] = pts_rest
                        chunk_lfc[idx] = lfc

                    # 13. Write results to memmap
                    for idx in range(n_groups):
                        gene_indices = chunk_gene_indices
                        u_matrix[idx, gene_indices] = chunk_u[idx]
                        pvalue_matrix[idx, gene_indices] = chunk_p[idx]
                        effect_matrix[idx, gene_indices] = chunk_effect[idx]
                        z_matrix[idx, gene_indices] = chunk_z[idx]
                        lfc_matrix[idx, gene_indices] = chunk_lfc[idx]
                        pts_matrix[idx, gene_indices] = chunk_pts[idx]
                        pts_rest_matrix[idx, gene_indices] = chunk_pts_rest[idx]
                    
                    # Update progress and checkpoint
                    n_chunks_processed += 1
                    pbar.update(1)
                    if n_chunks_processed % eff_checkpoint_interval == 0:
                        _save_wilcoxon_checkpoint(current_chunk)
                    current_chunk += 1
                
                # Final checkpoint
                _save_wilcoxon_checkpoint(current_chunk - 1)
                logger.info(f"Completed {n_chunks_processed} gene chunks")
        finally:
            backed.file.close()

        gene_symbols = pd.Index(gene_symbols).astype(str)
        gene_array = gene_symbols.to_numpy()
        pvalue_adj_matrix = _create_memmap("pvalue_adj", np.float64)
        _adjust_pvalue_matrix(pvalue_matrix, corr_method, out=pvalue_adj_matrix)

        for idx in range(n_groups):
            order_matrix[idx] = np.argsort(-z_matrix[idx], kind="mergesort")

        # Convert memmap arrays to regular arrays before tempdir cleanup
        z_arr = np.array(z_matrix)
        pval_arr = np.array(pvalue_matrix)
        pval_adj_arr = np.array(pvalue_adj_matrix)
        lfc_arr = np.array(lfc_matrix)
        effect_arr = np.array(effect_matrix)
        u_arr = np.array(u_matrix)
        pts_arr = np.array(pts_matrix, dtype=np.float32)
        pts_rest_arr = np.array(pts_rest_matrix, dtype=np.float32)
        order_arr = np.array(order_matrix)
        
        result = RankGenesGroupsResult(
            genes=gene_symbols,
            groups=candidates,
            statistics=z_arr,
            pvalues=pval_arr,
            pvalues_adj=pval_adj_arr,
            logfoldchanges=lfc_arr,
            effect_size=effect_arr,
            u_statistics=u_arr,
            pts=pts_arr,
            pts_rest=pts_rest_arr,
            order=order_arr,
            groupby=perturbation_column,
            method="wilcoxon",
            control_label=control_label,
            tie_correct=tie_correct,
            pvalue_correction=corr_method,
        )

    # Create AnnData with layer-based storage (avoid recarray-based rank_genes_groups
    # which fails with HDF5 header size limits for large group counts)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_arr, obs=obs, var=var)
    adata.layers["z_score"] = z_arr
    adata.layers["pvalue"] = pval_arr
    adata.layers["pvalue_adj"] = pval_adj_arr
    adata.layers["logfoldchange"] = lfc_arr
    adata.layers["u_statistic"] = u_arr
    adata.layers["pts"] = pts_arr
    adata.layers["pts_rest"] = pts_rest_arr
    adata.uns["method"] = "wilcoxon"
    adata.uns["control_label"] = control_label
    adata.uns["perturbation_column"] = perturbation_column
    adata.uns["tie_correct"] = tie_correct
    adata.uns["pvalue_correction"] = corr_method
    adata.write(output_path)
    
    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format:
        _write_rank_genes_groups_hdf5(output_path, result)
    
    result.result = AnnData(output_path)
    
    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    return result


def shrink_lfc(
    data: str | Path | AnnData | ad.AnnData,
    *,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    method: Literal["stats", "full"] = "stats",
    prior_scale_mode: Literal["global", "per_comparison"] = "global",
    min_mu: float = 0.0,
    n_jobs: int = -1,
    batch_size: int = 128,
    profiling: bool = False,
) -> RankGenesGroupsResult:
    """Apply apeGLM log-fold change shrinkage to existing NB-GLM results.
    
    This function applies apeGLM shrinkage using a Cauchy prior on the LFC
    coefficient. Two methods are available:
    
    - **stats** (default, recommended): Uses pre-computed MLE LFC and SE from
      the h5ad file with vectorized Newton-Raphson optimization. ~35× faster
      than full and maintains consistency with stored MLE coefficients.
    - **full**: Re-loads original count data and runs per-gene L-BFGS-B
      optimization. May produce different results from stored MLE for lowly
      expressed genes due to min_mu clamping differences in the likelihood.
    
    .. note::
        The "stats" method is recommended because CRISPYx NB-GLM fitting uses
        min_mu=0.5 clamping for numerical stability, which affects the stored
        MLE coefficients. The "full" method re-evaluates the likelihood without
        this constraint, potentially finding different optima for lowly expressed
        genes. The "stats" method preserves shrinkage direction (always toward
        zero) by working directly with the stored statistics.
    
    This enables separating the base NB-GLM fitting from shrinkage for:
    - Benchmarking: measure base fitting and shrinkage times separately
    - Flexibility: apply shrinkage to existing results
    - Speed: use stats method for production
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        NB-GLM results from `nb_glm_test`. Must have required layers and
        metadata in `uns`.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, appends "_shrunk" to input name.
    method
        Shrinkage method to use:
        
        - ``"stats"`` (default): Fast vectorized shrinkage using pre-computed
          MLE statistics. Uses Newton-Raphson optimization across all genes
          simultaneously. ~35× faster than "full" and maintains consistency
          with stored MLE coefficients.
        - ``"full"``: Full model re-fitting with L-BFGS-B per gene. Re-loads
          original count data. Note: May produce different results for lowly
          expressed genes due to min_mu clamping differences. Use "stats" for
          consistent shrinkage behavior.
    prior_scale_mode
        How to estimate the Cauchy prior scale parameter:
        
        - ``"global"`` (default): Estimate prior scale once from all 
          perturbations' MLE LFCs. Faster and often more stable.
        - ``"per_comparison"``: Estimate prior scale separately for each
          perturbation vs control comparison. Matches PyDESeq2's behavior
          exactly. Use for benchmarking to demonstrate parity.
    min_mu
        Minimum mean threshold for shrinkage likelihood evaluation. Default: 0.0
        (no clamping), matching PyDESeq2's lfc_shrink which omits min_mu entirely.
        PyDESeq2 uses min_mu=0.5 for NB-GLM fitting but does NOT pass min_mu to
        the shrinkage optimizer. This is intentional: the MLE coefficients represent
        the best fit, and shrinkage should evaluate the same likelihood surface.
    n_jobs
        Number of parallel jobs for per-gene optimization (only used when
        method="full"). Default -1 uses all available cores.
    batch_size
        Number of genes per joblib batch (only used when method="full").
        Default: 128, matching PyDESeq2.
    profiling
        If True, enable timing and memory profiling. When enabled, stores
        profiling data in `adata.uns["profiling"]` with fields:
        - `shrinkage_seconds`: Time for lfcShrink operation
        - `shrinkage_peak_memory_mb`: Peak memory during shrinkage
        - `profiling_enabled`: True
        When False (default), `adata.uns["profiling"]` is set to "NA".
    
    Returns
    -------
    RankGenesGroupsResult
        Updated differential expression results with shrunken LFCs.
        The result h5ad has:
        - `logfoldchange`: shrunken LFC values
        - `logfoldchange_raw`: original MLE LFC values (preserved)
        - `standard_error`: posterior SE reflecting shrinkage uncertainty
        - `X`: updated to shrunken LFC (effect_size)
    
    Examples
    --------
    >>> # Fast default (recommended for production)
    >>> shrunk = crispyx.de.shrink_lfc("nb_glm_result.h5ad")
    
    >>> # Benchmark-accurate (matches PyDESeq2 exactly)
    >>> shrunk = crispyx.de.shrink_lfc(
    ...     "nb_glm_result.h5ad",
    ...     method="full",
    ...     prior_scale_mode="per_comparison",
    ... )
    
    >>> # First run NB-GLM without shrinkage
    >>> result = crispyx.de.nb_glm_test(
    ...     "data.h5ad",
    ...     perturbation_column="perturbation",
    ...     lfc_shrinkage_type="none",  # No shrinkage during fitting
    ... )
    >>> # Then apply shrinkage as a separate step
    >>> shrunk_result = crispyx.de.shrink_lfc(result.result_path)
    """
    path = resolve_data_path(data)
    
    # Validate min_mu parameter
    if min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")
    
    # Initialize profiler if enabled (timing + memory sampling)
    profiler = None
    if profiling:
        from .profiling import Profiler
        profiler = Profiler(timing=True, memory=True, memory_method="rss", sampling=True)
        profiler.start("total")

    # Load the NB-GLM result
    adata = ad.read_h5ad(path)
    
    # Establish memory baseline AFTER loading h5ad for fair benchmarking
    # This way, profiling measures only shrinkage memory, not h5ad loading
    if profiling and profiler is not None:
        profiler.snapshot("after_load")
        profiler.reset_peak()  # Reset peak memory after h5ad load
        profiler.start("shrinkage")
    
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
    if "dispersion" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' does not have 'dispersion' layer. "
            "This function requires NB-GLM results with dispersion estimates."
        )
    
    # Get required metadata
    control_label = adata.uns.get("control_label", "control")
    perturbation_column = adata.uns.get("perturbation_column", "perturbation")
    
    # Get raw LFC, SE, and dispersion in ln-scale (required for apeGLM optimization)
    # Use ln-scale layers if available (v0.5.0+), otherwise fall back with conversion
    lfc_base = adata.uns.get("lfc_base", "log2")
    if "logfoldchange_raw_ln" in adata.layers:
        raw_lfc = adata.layers["logfoldchange_raw_ln"]  # Already ln-scale
        se = adata.layers["standard_error_ln"]
    else:
        # Backward compatibility: convert from log2 to ln if needed
        raw_lfc = adata.layers["logfoldchange_raw"]
        se = adata.layers["standard_error"]
        if lfc_base == "log2":
            ln2 = np.log(2)
            raw_lfc = raw_lfc * ln2  # Convert log2 -> ln
            se = se * ln2
            logger.warning(
                "Input h5ad lacks ln-scale layers; converting from log2. "
                "For best accuracy, re-run nb_glm_test with crispyx>=0.5.0."
            )
    dispersion = adata.layers["dispersion"]
    
    # Get fitted intercept from NB-GLM (ln-scale, critical for accurate shrinkage)
    if "intercept" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' lacks 'intercept' layer. "
            "method='full' requires NB-GLM results from nb_glm_test v0.5.1+. "
            "Use method='stats' or re-run nb_glm_test."
        )
    fitted_intercept = adata.layers["intercept"]  # Already ln-scale
    logger.info("Using fitted intercept from NB-GLM for shrinkage")
    
    n_groups, n_genes = raw_lfc.shape
    candidates = list(adata.obs_names.astype(str))
    gene_symbols = pd.Index(adata.var_names).astype(str)
    
    # Estimate global prior scale from ALL perturbations' MLE LFCs
    all_mle_lfc = raw_lfc.ravel()
    all_mle_se = se.ravel()
    valid_mask = np.isfinite(all_mle_lfc) & np.isfinite(all_mle_se) & (all_mle_se > 0)
    global_prior_scale = _estimate_apeglm_prior_scale(
        all_mle_lfc[valid_mask], 
        all_mle_se[valid_mask]
    )
    logger.info(f"Global prior scale for apeGLM: {global_prior_scale:.4f}")
    
    # Initialize output arrays
    shrunk_lfc = np.zeros_like(raw_lfc)
    shrunk_se = np.zeros_like(se)
    total_converged = 0
    total_genes_processed = 0
    
    if method == "stats":
        # Fast stats-based shrinkage using vectorized Newton-Raphson
        # No need to load original dataset - uses pre-computed MLE stats
        logger.info(f"Using fast stats-based shrinkage (method='stats')")
        
        # Try to derive base_mean from fitted intercept for gene-specific priors
        # intercept is ln(mu_control), so base_mean ≈ mean(exp(intercept)) across perturbations
        # This is a proxy for expression level used in gene-specific prior scaling
        if np.any(np.isfinite(fitted_intercept)):
            # Use mean intercept across perturbations for each gene
            # Suppress "Mean of empty slice" RuntimeWarning when all values are NaN for a gene
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                mean_intercept = np.nanmean(fitted_intercept, axis=0)  # Shape: (n_genes,)
            base_mean_proxy = np.exp(np.clip(mean_intercept, -20, 20))  # Clamp to avoid overflow
            logger.debug("Using intercept-derived base_mean for gene-specific priors")
        else:
            base_mean_proxy = None
            logger.debug("base_mean not available, using uniform prior scale")
        
        # Track genes that need full re-fitting
        all_needs_refit = np.zeros((n_groups, n_genes), dtype=bool)
        
        for group_idx, pert_label in enumerate(candidates):
            logger.debug(f"Shrinking LFC for perturbation {group_idx + 1}/{n_groups}: {pert_label}")
            
            mle_lfc_group = raw_lfc[group_idx]
            se_group = se[group_idx]
            
            # Determine prior scale based on mode
            if prior_scale_mode == "per_comparison":
                pert_prior_scale = _estimate_apeglm_prior_scale(mle_lfc_group, se_group)
            else:
                pert_prior_scale = global_prior_scale
            
            # Vectorized shrinkage using Newton-Raphson
            # NOTE: use_gene_specific_prior=False for PyDESeq2 parity (gene-specific
            # prior scaling is non-standard and causes accuracy regression)
            shrunk_lfc_group, shrunk_se_group, converged, needs_refit = shrink_lfc_apeglm_from_stats(
                mle_lfc=mle_lfc_group,
                mle_se=se_group,
                prior_scale=pert_prior_scale,
                base_mean=base_mean_proxy,
                use_gene_specific_prior=False,
                hybrid_fallback=True,
            )
            
            shrunk_lfc[group_idx] = shrunk_lfc_group
            shrunk_se[group_idx] = shrunk_se_group
            all_needs_refit[group_idx] = needs_refit
            
            n_converged = converged.sum()
            total_converged += n_converged
            total_genes_processed += n_genes
            logger.debug(f"  {n_converged}/{n_genes} genes converged for {pert_label}")
        
        # Log warning if many genes didn't converge
        convergence_rate = total_converged / total_genes_processed if total_genes_processed > 0 else 1.0
        if convergence_rate < 0.95:
            logger.warning(
                f"Low convergence rate: {total_converged}/{total_genes_processed} "
                f"({convergence_rate:.1%}) genes converged within max_iter. "
                "Consider using method='full' for better accuracy."
            )
        
        # Log info about genes that might need full re-fitting
        total_needs_refit = all_needs_refit.sum()
        if total_needs_refit > 0:
            refit_rate = total_needs_refit / (n_groups * n_genes)
            logger.info(
                f"Hybrid fallback: {total_needs_refit} gene-perturbation pairs "
                f"({refit_rate:.1%}) flagged for potential accuracy improvement with method='full'"
            )
    
    else:  # method == "full"
        # Full model re-fitting with L-BFGS-B per gene
        logger.info(f"Using full model re-fitting (method='full')")
        
        # Validate original dataset path for full method
        original_dataset_path = adata.uns.get("original_dataset_path")
        if original_dataset_path is None:
            raise ValueError(
                f"Input file '{path}' does not have 'original_dataset_path' in uns. "
                "method='full' requires NB-GLM results from nb_glm_test v0.4.0+. "
                "Use method='stats' or re-run nb_glm_test."
            )
        
        # Strip /workspace/ prefix from Docker paths if present
        if original_dataset_path.startswith("/workspace/"):
            original_dataset_path = original_dataset_path[len("/workspace/"):]
        
        original_path = Path(original_dataset_path)
        if not original_path.exists():
            # Also try the input file's parent as base directory
            input_parent = path.parent
            relative_name = Path(original_dataset_path).name
            alternative_paths = [
                input_parent.parent / ".cache" / relative_name,  # Try ../..cache/
                input_parent / ".cache" / relative_name,  # Try ../.cache/
                input_parent.parent.parent / ".cache" / relative_name,  # Try ../../../.cache/
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    original_path = alt_path
                    break
            else:
                raise FileNotFoundError(
                    f"Original dataset not found: {original_dataset_path}. "
                    "method='full' requires access to the original count data. "
                    "Use method='stats' to shrink without original data."
                )
        
        # Load original dataset for streaming
        backed = read_backed(original_path)
        try:
            labels = backed.obs[perturbation_column].astype(str).to_numpy()
            
            # Get or compute size factors
            size_factors_global = adata.uns.get("size_factors")
            if size_factors_global is not None:
                size_factors_all = np.asarray(size_factors_global, dtype=np.float64)
            else:
                size_factors_all = _median_of_ratios_size_factors(original_path)
            
            # Control cell indices
            control_mask = labels == control_label
            control_idx = np.where(control_mask)[0]
            
            if len(control_idx) == 0:
                raise ValueError(f"No control cells found with label '{control_label}'")
            
            # Load control cells once (they're reused for all perturbations)
            control_counts = backed.X[control_idx, :].toarray() if sp.issparse(backed.X[control_idx, :]) else np.asarray(backed.X[control_idx, :])
            control_size_factors = size_factors_all[control_idx]
            
            # Process each perturbation
            for group_idx, pert_label in enumerate(candidates):
                logger.debug(f"Shrinking LFC for perturbation {group_idx + 1}/{n_groups}: {pert_label}")
                
                # Get perturbation cell indices
                pert_mask = labels == pert_label
                pert_idx = np.where(pert_mask)[0]
                
                if len(pert_idx) == 0:
                    shrunk_lfc[group_idx] = raw_lfc[group_idx]
                    shrunk_se[group_idx] = se[group_idx]
                    continue
                
                # Load perturbation cells
                pert_counts = backed.X[pert_idx, :].toarray() if sp.issparse(backed.X[pert_idx, :]) else np.asarray(backed.X[pert_idx, :])
                pert_size_factors = size_factors_all[pert_idx]
                
                # Combine control and perturbation
                combined_counts = np.vstack([control_counts, pert_counts])
                combined_size_factors = np.concatenate([control_size_factors, pert_size_factors])
                
                # Build design matrix
                n_control = len(control_idx)
                n_pert = len(pert_idx)
                n_combined = n_control + n_pert
                
                design_matrix = np.zeros((n_combined, 2), dtype=np.float64)
                design_matrix[:, 0] = 1.0
                design_matrix[n_control:, 1] = 1.0
                
                # Get MLE coefficients from NB-GLM
                mle_intercept = fitted_intercept[group_idx]
                mle_lfc_group = raw_lfc[group_idx]
                mle_coef = np.vstack([mle_intercept, mle_lfc_group])
                
                disp_group = dispersion[group_idx]
                se_group = se[group_idx]
                
                # Determine prior scale based on mode
                if prior_scale_mode == "per_comparison":
                    pert_prior_scale = _estimate_apeglm_prior_scale(mle_lfc_group, se_group)
                else:
                    pert_prior_scale = global_prior_scale
                
                # Full apeGLM shrinkage with L-BFGS-B per gene
                # NOTE: By default (min_mu=0.0), no min_mu clamping is applied,
                # matching PyDESeq2's lfc_shrink which omits min_mu entirely.
                # PyDESeq2 uses min_mu=0.5 for NB-GLM fitting but does NOT pass
                # min_mu to the shrinkage optimizer.
                shrunk_coef, shrunk_se_group, converged = shrink_lfc_apeglm(
                    counts=combined_counts,
                    design_matrix=design_matrix,
                    size_factors=combined_size_factors,
                    dispersion=disp_group,
                    mle_coef=mle_coef,
                    mle_se=se_group,
                    shrink_index=1,
                    prior_scale=pert_prior_scale,
                    n_jobs=n_jobs,
                    batch_size=batch_size,
                    min_mu=min_mu,
                )
                
                shrunk_lfc[group_idx] = shrunk_coef[1, :]
                shrunk_se[group_idx] = shrunk_se_group
                
                n_converged = converged.sum()
                total_converged += n_converged
                total_genes_processed += n_genes
                logger.debug(f"  {n_converged}/{n_genes} genes converged for {pert_label}")
            
        finally:
            backed.file.close()
    
    # Convert shrunk results from ln-scale to log2 if original output was log2
    if lfc_base == "log2":
        ln2 = np.log(2)
        shrunk_lfc = shrunk_lfc / ln2
        shrunk_se = shrunk_se / ln2
    
    # Update layers
    adata.layers["logfoldchange"] = shrunk_lfc
    adata.layers["standard_error"] = shrunk_se  # Posterior SE
    adata.X = shrunk_lfc  # Update effect_size matrix
    
    # Update metadata
    adata.uns["lfc_shrinkage_type"] = "apeglm"
    adata.uns["apeglm_prior_scale"] = global_prior_scale
    adata.uns["shrinkage_method"] = method
    adata.uns["prior_scale_mode"] = prior_scale_mode
    
    # Store profiling results or "NA" for production
    if profiling and profiler is not None:
        profiler.stop("shrinkage")
        profiler.snapshot("shrinkage_end")
        profiler.stop("total")
        profiler.stop_sampling()
        stats = profiler.get_stats()
        adata.uns["profiling"] = {
            "profiling_enabled": True,
            "shrinkage_seconds": stats.get("timing", {}).get("sections", {}).get("shrinkage", {}).get("seconds", 0.0),
            "shrinkage_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
            "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
        }
    else:
        adata.uns["profiling"] = "NA"
    
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
        f"Applied apeGLM LFC shrinkage (method={method}, prior_scale_mode={prior_scale_mode}) "
        f"to {n_groups} perturbations, {n_genes} genes (prior_scale={global_prior_scale:.4f}). "
        f"Output: {output_path}"
    )
    
    return result
