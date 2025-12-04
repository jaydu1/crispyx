"""Streamlined CRISPR screen analysis toolkit with Scanpy-style entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np

from .data import (
    AnnData,
    ensure_gene_symbol_column,
    read_h5ad_ondisk,
    read_backed,
    resolve_control_label,
    resolve_output_path,
)
from .de import (
    RankGenesGroupsResult,
    _adjust_pvalue_matrix,
    nb_glm_test,
    t_test,
    wilcoxon_test,
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


def _resolve_backed_path(data: str | Path | ad.AnnData | AnnData) -> Path:
    """Return the on-disk path for a backed AnnData object or path-like input."""

    if isinstance(data, (str, Path)):
        return Path(data)
    if isinstance(data, AnnData):
        return data.path
    if isinstance(data, ad.AnnData):
        filename = getattr(data, "filename", None)
        if filename:
            return Path(filename)
        raise TypeError(
            "Operations in streamlined_crispr expect a backed AnnData object or file path."
        )
    raise TypeError(
        f"Expected a path-like value or backed AnnData; received {type(data)!r}."
    )


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
        )
        return result.filtered


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


pp = _PreprocessingNamespace()
pb = _PseudobulkNamespace()
tl = _ToolsNamespace()

__all__ = [
    "pp",
    "pb",
    "tl",
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
    "ensure_gene_symbol_column",
    "AnnData",
    "read_h5ad_ondisk",
    "read_backed",
    "resolve_control_label",
    "resolve_output_path",
]
