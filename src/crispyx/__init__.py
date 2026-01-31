"""Streamlined CRISPR screen analysis toolkit with Scanpy-style entry points."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Literal

import anndata as ad
import numpy as np

try:
    __version__ = version("crispyx")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    ensure_gene_symbol_column,
    normalize_total_log1p,
    read_h5ad_ondisk,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    sort_by_perturbation,
)
from .de import (
    RankGenesGroupsResult,
    _adjust_pvalue_matrix,
    nb_glm_test,
    shrink_lfc,
    t_test,
    wilcoxon_test,
)
from .profiling import (
    Profiler,
    MemoryProfiler,
    TimingProfiler,
    plot_benchmark_comparison,
)
from .plotting import (
    materialize_rank_genes_groups,
    plot_ma,
    plot_qc_perturbation_counts,
    plot_qc_summary,
    plot_top_genes_bar,
    plot_volcano,
    rank_genes_groups as plot_rank_genes_groups,
    rank_genes_groups_df,
)
from .pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from .qc import (
    filter_cells_by_gene_count,
    filter_genes_by_cell_count,
    filter_perturbations_by_cell_count,
    quality_control_summary,
)


# Alias for backward compatibility
_resolve_backed_path = resolve_data_path


def _infer_control_label(
    path: Path,
    perturbation_column: str,
    control_label: str | None,
) -> str:
    if control_label is not None:
        return str(control_label)
    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                "Perturbation column '%s' was not found in adata.obs. Available columns: %s"
                % (perturbation_column, list(backed.obs.columns))
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()
    return resolve_control_label(labels, None)


def _t_test_results_to_rank_genes(
    path: Path,
    results,
    *,
    gene_name_column: str | None,
    perturbation_column: str,
    control_label: str,
    corr_method: str,
    output_dir: str | Path | None,
    data_name: str | None,
) -> RankGenesGroupsResult:
    groups = list(results.keys())
    if groups:
        first = results[groups[0]]
        genes = first.genes
        effect_matrix = np.vstack([results[group].effect_size for group in groups])
        statistic_matrix = np.vstack([results[group].statistic for group in groups])
        pvalue_matrix = np.vstack([results[group].pvalue for group in groups])
        result_view = first.result
    else:
        backed = read_backed(path)
        try:
            if gene_name_column is None:
                genes = backed.var_names.astype(str)
            else:
                genes = ensure_gene_symbol_column(backed, gene_name_column)
        finally:
            backed.file.close()
        effect_matrix = np.zeros((0, genes.size), dtype=float)
        statistic_matrix = np.zeros_like(effect_matrix)
        pvalue_matrix = np.ones_like(effect_matrix)
        result_path = resolve_output_path(
            path,
            suffix="t_test_de",
            output_dir=output_dir,
            data_name=data_name,
        )
        result_view = AnnData(result_path)

    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        raise ValueError(
            "corr_method must be 'benjamini-hochberg' or 'bonferroni' for t-tests"
        )

    pvalue_adj = (
        _adjust_pvalue_matrix(pvalue_matrix, corr_method)
        if pvalue_matrix.size
        else np.zeros_like(pvalue_matrix)
    )
    order = (
        np.argsort(-np.abs(statistic_matrix), axis=1, kind="mergesort")
        if statistic_matrix.size
        else np.zeros(statistic_matrix.shape, dtype=int)
    )
    zeros = np.zeros_like(statistic_matrix)

    result = RankGenesGroupsResult(
        genes=genes,
        groups=groups,
        statistics=statistic_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj,
        logfoldchanges=effect_matrix,
        effect_size=effect_matrix,
        u_statistics=zeros,
        pts=zeros,
        pts_rest=zeros,
        order=order,
        groupby=perturbation_column,
        method="t_test",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=result_view,
    )
    if result.result is not None:
        memory = result.result.to_memory()
        memory.uns["rank_genes_groups"] = result.to_rank_genes_groups_dict()
        memory.uns["genes"] = genes.to_numpy()
        memory.uns["method"] = "t_test"
        memory.uns["control_label"] = control_label
        memory.uns["tie_correct"] = False
        memory.uns["pvalue_correction"] = corr_method
        memory.write(result.result.path)
        result.result.close()
        result.result = AnnData(result.result.path)
    return result


class _PreprocessingNamespace:
    """Scanpy-style preprocessing entry points."""

    def filter_cells(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_genes: int = 100,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
    ):
        path = _resolve_backed_path(data)
        return filter_cells_by_gene_count(
            path,
            min_genes=min_genes,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )

    def filter_genes(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_cells: int = 100,
        cell_mask: np.ndarray | None = None,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
    ):
        path = _resolve_backed_path(data)
        return filter_genes_by_cell_count(
            path,
            min_cells=min_cells,
            cell_mask=cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )

    def filter_perturbations(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str,
        control_label: str | None = None,
        min_cells: int = 50,
        base_mask: np.ndarray | None = None,
    ):
        path = _resolve_backed_path(data)
        return filter_perturbations_by_cell_count(
            path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            min_cells=min_cells,
            base_mask=base_mask,
        )

    def qc_summary(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_genes: int = 100,
        min_cells_per_perturbation: int = 50,
        min_cells_per_gene: int = 100,
        perturbation_column: str,
        control_label: str | None = None,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        cache_mode: Literal['memory', 'memmap', 'none'] = 'memmap',
    ):
        path = _resolve_backed_path(data)
        result = quality_control_summary(
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
            cache_mode=cache_mode,
        )
        return result.filtered

    def normalize_total_log1p(
        self,
        data: str | Path | ad.AnnData,
        output_path: str | Path | None = None,
        *,
        normalize: bool = True,
        log1p: bool = True,
        target_sum: float = 1e4,
        chunk_size: int = 4096,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        verbose: bool = True,
    ) -> AnnData:
        """Stream normalize and/or log-transform an h5ad file.
        
        This is the streaming equivalent of calling ``scanpy.pp.normalize_total``
        followed by ``scanpy.pp.log1p``. Processes data in chunks to avoid OOM.
        
        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData.
        output_path
            Path for output. If None, uses output_dir/data_name pattern.
        normalize
            Apply total-count normalization. Default True.
        log1p
            Apply log1p transformation. Default True.
        target_sum
            Target counts per cell. Default 1e4.
        chunk_size
            Cells per chunk. Default 4096.
        output_dir
            Output directory. Defaults to input file's directory.
        data_name
            Custom output name suffix.
        verbose
            Print progress.
        
        Returns
        -------
        AnnData
            Read-only AnnData wrapper pointing to output file.
        """
        return normalize_total_log1p(
            data,
            output_path,
            normalize=normalize,
            log1p=log1p,
            target_sum=target_sum,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
            verbose=verbose,
        )


class _PseudobulkNamespace:
    """Pseudo-bulk estimators exposed under a dedicated namespace."""

    def average_log_expression(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str,
        control_label: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        chunk_size: int = 2048,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
    ):
        path = _resolve_backed_path(data)
        return compute_average_log_expression(
            path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
        )

    def pseudobulk(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str,
        control_label: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        baseline_count: float = 1.0,
        chunk_size: int = 2048,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
    ):
        path = _resolve_backed_path(data)
        return compute_pseudobulk_expression(
            path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            baseline_count=baseline_count,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
        )


class _ToolsNamespace:
    """Differential expression entry points mirroring Scanpy's ``tl`` API."""

    def rank_genes_groups(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str,
        method: str = "wilcoxon",
        control_label: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        corr_method: str = "benjamini-hochberg",
        **kwargs,
    ) -> RankGenesGroupsResult:
        path = _resolve_backed_path(data)
        method_key = method.lower().replace("_", "-")
        method_map = {
            "wilcoxon": "wilcoxon",
            "wilcox": "wilcoxon",
            "wilcoxon-test": "wilcoxon",
            "wilcox-test": "wilcoxon",
            "t-test": "t_test",
            "ttest": "t_test",
            "wald": "t_test",  # backward compatibility
            "nb-glm": "nb_glm",
            "nb-glm-test": "nb_glm",
        }
        normalised = method_map.get(method_key, method_key)
        control = _infer_control_label(path, perturbation_column, control_label)

        base_kwargs = dict(
            perturbation_column=perturbation_column,
            control_label=control,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            output_dir=output_dir,
            data_name=data_name,
        )

        if normalised == "wilcoxon":
            allowed = {"min_cells_expressed", "chunk_size", "tie_correct", "n_jobs"}
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    "Unexpected keyword arguments for wilcoxon method: %s"
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            result = wilcoxon_test(
                path,
                corr_method=corr_method,
                **base_kwargs,
                **method_kwargs,
            )
            if result.result is None:
                raise RuntimeError("Wilcoxon test did not produce an AnnData result.")
            return result.result

        if normalised == "nb_glm":
            allowed = {
                "covariates",
                "dispersion",
                "fit_method",
                "share_dispersion",
                "max_iter",
                "tol",
                "poisson_init_iter",
                "min_cells_expressed",
                "min_total_count",
                "chunk_size",
                "n_jobs",
            }
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    "Unexpected keyword arguments for nb_glm method: %s"
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            result = nb_glm_test(
                path,
                corr_method=corr_method,
                **base_kwargs,
                **method_kwargs,
            )
            if result.result is None:
                raise RuntimeError("NB-GLM test did not produce an AnnData result.")
            return result.result

        if normalised == "t_test":
            allowed = {"min_cells_expressed", "cell_chunk_size", "n_jobs"}
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    (
                        "Unexpected keyword arguments for t_test method: %s. "
                        "Supported options include cell_chunk_size (cells per chunk), "
                        "min_cells_expressed, and n_jobs; perturbation_chunk_size is not yet supported."
                    )
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            results = t_test(
                path,
                **base_kwargs,
                **method_kwargs,
            )
            mapped = _t_test_results_to_rank_genes(
                path,
                results,
                gene_name_column=gene_name_column,
                perturbation_column=perturbation_column,
                control_label=control,
                corr_method=corr_method,
                output_dir=output_dir,
                data_name=data_name,
            )
            if mapped.result is None:
                raise RuntimeError("t-test did not produce an AnnData result.")
            return mapped.result

        raise ValueError(
            f"Unsupported differential expression method: {method}. "
            "Choose from 'wilcoxon', 't-test', or 'nb_glm'."
        )

    def shrink_lfc(
        self,
        data: str | Path | ad.AnnData,
        *,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        method: str = "stats",
        prior_scale_mode: str = "global",
        min_mu: float = 0.0,
        n_jobs: int = -1,
        batch_size: int = 128,
        profiling: bool = False,
    ):
        """Apply apeGLM LFC shrinkage to NB-GLM results.

        Parameters
        ----------
        data
            Path to h5ad file from ``nb_glm_test()`` or a backed AnnData object.
        output_dir
            Directory for output. Defaults to input file's directory.
        data_name
            Custom name for output file.
        method
            Shrinkage computation method: "stats" (faster) or "full".
        prior_scale_mode
            Prior estimation scope: "global" or "per_comparison".
        min_mu
            Minimum mean for numerical stability.
        n_jobs
            Number of parallel workers. -1 uses all cores.
        batch_size
            Number of genes per batch.
        profiling
            Enable timing/memory profiling.

        Returns
        -------
        RankGenesGroupsResult
            Shrunk differential expression results.
        """
        path = _resolve_backed_path(data)
        return shrink_lfc(
            path,
            output_dir=output_dir,
            data_name=data_name,
            method=method,
            prior_scale_mode=prior_scale_mode,
            min_mu=min_mu,
            n_jobs=n_jobs,
            batch_size=batch_size,
            profiling=profiling,
        )


class _PlottingNamespace:
    """Scanpy-style plotting entry points for crispyx."""

    def rank_genes_groups(self, data, **kwargs):
        return plot_rank_genes_groups(data, **kwargs)

    def rank_genes_groups_df(self, data, group, **kwargs):
        return rank_genes_groups_df(data, group, **kwargs)

    def volcano(self, **kwargs):
        return plot_volcano(**kwargs)

    def ma(self, **kwargs):
        return plot_ma(**kwargs)

    def top_genes_bar(self, **kwargs):
        return plot_top_genes_bar(**kwargs)

    def qc_perturbation_counts(self, **kwargs):
        return plot_qc_perturbation_counts(**kwargs)

    def qc_summary(self, qc_result, **kwargs):
        return plot_qc_summary(qc_result, **kwargs)

    def materialize_rank_genes_groups(self, data, **kwargs):
        return materialize_rank_genes_groups(data, **kwargs)

pp = _PreprocessingNamespace()
pb = _PseudobulkNamespace()
tl = _ToolsNamespace()
pl = _PlottingNamespace()

__all__ = [
    "__version__",
    "pp",
    "pb",
    "tl",
    "pl",
    "filter_cells_by_gene_count",
    "filter_genes_by_cell_count",
    "filter_perturbations_by_cell_count",
    "quality_control_summary",
    "compute_average_log_expression",
    "compute_pseudobulk_expression",
    "RankGenesGroupsResult",
    "t_test",
    "wilcoxon_test",
    "nb_glm_test",
    "shrink_lfc",
    "ensure_gene_symbol_column",
    "AnnData",
    "read_h5ad_ondisk",
    "read_backed",
    "resolve_control_label",
    "resolve_output_path",
    "calculate_optimal_chunk_size",
    "calculate_optimal_gene_chunk_size",
    "normalize_total_log1p",
    # Profiling utilities
    "Profiler",
    "MemoryProfiler",
    "TimingProfiler",
    "plot_benchmark_comparison",
    # Plotting utilities
    "materialize_rank_genes_groups",
    "rank_genes_groups_df",
    "plot_volcano",
    "plot_ma",
    "plot_top_genes_bar",
    "plot_qc_perturbation_counts",
    "plot_qc_summary",
]
