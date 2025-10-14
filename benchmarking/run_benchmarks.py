"""Utilities to benchmark streaming CRISPR screen analysis methods."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import pandas as pd

# Ensure the local package is importable when the project has not been installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from streamlined_crispr.data import read_backed
from streamlined_crispr.de import wald_test, wilcoxon_test
from streamlined_crispr.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from streamlined_crispr.qc import quality_control_summary
from streamlined_crispr.scanpy_validation import ComparisonResult, compare_with_scanpy


@dataclass
class BenchmarkMethod:
    """Description of a method that should be benchmarked."""

    name: str
    description: str
    function: Callable[..., Any]
    kwargs: Dict[str, Any]
    summary: Callable[[Any, Dict[str, Any]], Dict[str, Any]]


@dataclass
class DifferentialComparisonSummary:
    """Summary statistics for comparing streaming DE results to a reference tool."""

    test_type: str
    reference_tool: str
    effect_max_abs_diff: float | None
    statistic_max_abs_diff: float | None
    pvalue_max_abs_diff: float | None
    streaming_result_path: str | None
    reference_result_path: str | None
    error: Optional[str] = None


_STANDARD_DE_COLUMNS = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]


def _streaming_de_to_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a streaming differential expression mapping to a tidy DataFrame."""

    frames = []
    for perturbation, entry in result.items():
        genes = getattr(entry, "genes", None)
        if genes is None:
            continue
        gene_index = pd.Index(genes).astype(str)
        n_rows = len(gene_index)
        frame = pd.DataFrame(
            {
                "perturbation": [str(perturbation)] * n_rows,
                "gene": gene_index,
                "effect_size": getattr(entry, "effect_size", pd.NA),
                "statistic": getattr(entry, "statistic", pd.NA),
                "pvalue": getattr(entry, "pvalue", pd.NA),
            }
        )
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=_STANDARD_DE_COLUMNS)


def _standardise_de_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``df`` with standard differential expression column names."""

    if df is None or df.empty:
        return pd.DataFrame(columns=_STANDARD_DE_COLUMNS)

    result = df.copy()
    lower_to_original = {col.lower(): col for col in result.columns}

    def _resolve_column(candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                return original
        return None

    rename: Dict[str, str] = {}
    perturbation_col = _resolve_column(["perturbation", "group", "cluster", "label"])
    gene_col = _resolve_column(["gene", "genes", "name", "names", "feature"])
    effect_col = _resolve_column(["effect_size", "logfoldchange", "logfoldchanges", "lfc", "coefficient"])
    stat_col = _resolve_column(["statistic", "statistics", "score", "scores", "wald_statistic", "zscore", "t_stat", "u_stat"])
    pvalue_col = _resolve_column(["pvalue", "p_value", "pval", "pvals", "pvalue_raw", "pvalue_adj", "pvals_adj"])

    if perturbation_col is not None:
        rename[perturbation_col] = "perturbation"
    if gene_col is not None:
        rename[gene_col] = "gene"
    if effect_col is not None:
        rename[effect_col] = "effect_size"
    if stat_col is not None:
        rename[stat_col] = "statistic"
    if pvalue_col is not None:
        rename[pvalue_col] = "pvalue"

    result = result.rename(columns=rename)

    for column in _STANDARD_DE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[_STANDARD_DE_COLUMNS]
    result["perturbation"] = result["perturbation"].astype(str)
    result["gene"] = result["gene"].astype(str)
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["statistic"] = pd.to_numeric(result["statistic"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    return result


def _compare_de_frames(streaming: pd.DataFrame, reference: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Return max absolute differences between streaming and reference metrics."""

    if streaming.empty or reference.empty:
        return {"effect_size": None, "statistic": None, "pvalue": None}

    merged = streaming.merge(reference, on=["perturbation", "gene"], suffixes=("_stream", "_reference"))
    if merged.empty:
        return {"effect_size": None, "statistic": None, "pvalue": None}

    metrics: Dict[str, Optional[float]] = {}
    for column in ["effect_size", "statistic", "pvalue"]:
        stream_col = f"{column}_stream"
        ref_col = f"{column}_reference"
        if stream_col not in merged or ref_col not in merged:
            metrics[column] = None
            continue
        differences = (merged[stream_col] - merged[ref_col]).abs().dropna()
        metrics[column] = float(differences.max()) if not differences.empty else None
    return metrics


def _prepare_reference_anndata(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    perturbation_column: str,
    control_label: str,
):
    """Return a filtered in-memory AnnData object for reference comparisons."""

    import scanpy as sc  # Imported lazily to keep runtime dependencies optional

    adata = sc.read_h5ad(str(dataset_path))
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells_per_gene:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if min_cells_per_perturbation:
        labels = adata.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata = adata[keep].copy()
    return adata


def _run_scanpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    method: str,
    output_dir: Path,
    data_name: str,
) -> tuple[pd.DataFrame | None, Optional[Path], Optional[str]]:
    """Execute Scanpy's differential expression workflow and return a DataFrame."""

    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        return None, None, str(exc)

    adata = _prepare_reference_anndata(
        dataset_path,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        perturbation_column=perturbation_column,
        control_label=control_label,
    )

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_column,
        method=method,
        reference=control_label,
        n_genes=adata.n_vars,
    )
    df = sc.get.rank_genes_groups_df(adata, None)

    reference_path: Optional[Path] = None
    if not df.empty:
        reference_path = output_dir / f"{data_name}_{method}_scanpy_de.csv"
        df.to_csv(reference_path, index=False)
    return df, reference_path, None


def _resolve_pertpy_runner(module: Any, method: str) -> Optional[Callable[..., Any]]:
    """Best-effort resolution of a Pertpy differential expression runner."""

    candidates = [
        method,
        f"{method}_de",
        f"run_{method}",
        f"run_{method}_de",
        method.lower(),
        method.upper(),
    ]
    for name in candidates:
        runner = getattr(module, name, None)
        if callable(runner):
            return runner
    return None


def _convert_reference_result_to_dataframe(result: Any) -> Optional[pd.DataFrame]:
    """Normalise a Pertpy reference result to a ``DataFrame`` when possible."""

    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, Mapping):
        frames = []
        for perturbation, value in result.items():
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                if "perturbation" not in frame.columns:
                    frame["perturbation"] = str(perturbation)
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    if hasattr(result, "to_df"):
        return result.to_df()
    return None


def _run_pertpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    backend: str,
    output_dir: Path,
    data_name: str,
) -> tuple[pd.DataFrame | None, Optional[Path], Optional[str]]:
    """Execute a Pertpy-backed differential expression method."""

    try:
        import pertpy as pt
        import scanpy as sc  # Needed for AnnData IO
    except ImportError as exc:  # pragma: no cover - optional dependency
        return None, None, str(exc)

    _ = sc  # Silence unused import warnings in environments without Scanpy

    module = getattr(pt, "tools", None)
    if module is None:
        return None, None, "pertpy.tools module unavailable"
    de_module = getattr(module, "differential_expression", None)
    if de_module is None:
        return None, None, "pertpy.tools.differential_expression module unavailable"

    runner = _resolve_pertpy_runner(de_module, backend)
    if runner is None:
        return None, None, f"Pertpy differential expression runner '{backend}' not found"

    try:
        adata = _prepare_reference_anndata(
            dataset_path,
            min_genes=min_genes,
            min_cells_per_gene=min_cells_per_gene,
            min_cells_per_perturbation=min_cells_per_perturbation,
            perturbation_column=perturbation_column,
            control_label=control_label,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        return None, None, str(exc)

    call_attempts = (
        {"groupby": perturbation_column, "control": control_label},
        {"group_key": perturbation_column, "control": control_label},
        {"groupby": perturbation_column, "reference": control_label},
    )
    last_type_error: Optional[Exception] = None
    result = None
    for kwargs in call_attempts:
        try:
            result = runner(adata, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive
            return None, None, str(exc)
        else:
            break
    else:
        if last_type_error is not None:
            return None, None, str(last_type_error)
        return None, None, "Pertpy differential expression runner failed to execute"

    df = _convert_reference_result_to_dataframe(result)
    reference_path: Optional[Path] = None
    if df is not None and not df.empty:
        reference_path = output_dir / f"{data_name}_{backend}_pertpy_de.csv"
        df.to_csv(reference_path, index=False)
    return df, reference_path, None


def compare_scanpy_quality_control(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
) -> ComparisonResult:
    """Run the full Scanpy validation workflow for QC comparisons."""

    return compare_with_scanpy(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_perturbation=min_cells_per_perturbation,
        min_cells_per_gene=min_cells_per_gene,
        gene_name_column=gene_name_column,
        output_dir=output_dir,
        data_name="benchmark_scanpy_qc",
    )


def compare_scanpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
    test_type: str,
) -> DifferentialComparisonSummary:
    """Compare streaming differential expression against Scanpy."""

    if test_type == "wilcoxon":
        stream_result = wilcoxon_test(
            path=dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            output_dir=output_dir,
            data_name="benchmark_scanpy_wilcoxon",
        )
        reference_method = "wilcoxon"
    else:
        stream_result = wald_test(
            path=dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            output_dir=output_dir,
            data_name="benchmark_scanpy_wald",
        )
        reference_method = "t-test"

    streaming_frame = _streaming_de_to_frame(stream_result)
    streaming_path = None
    if not streaming_frame.empty:
        any_result = next(iter(stream_result.values()))
        streaming_path = str(getattr(any_result, "result_path", "")) or None

    reference_df, reference_path, error = _run_scanpy_de(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        method=reference_method,
        output_dir=output_dir,
        data_name="benchmark_scanpy_reference",
    )

    metrics = {"effect_size": None, "statistic": None, "pvalue": None}
    if reference_df is not None:
        metrics = _compare_de_frames(streaming_frame, _standardise_de_dataframe(reference_df))

    return DifferentialComparisonSummary(
        test_type=test_type,
        reference_tool=f"scanpy_{reference_method}",
        effect_max_abs_diff=metrics["effect_size"],
        statistic_max_abs_diff=metrics["statistic"],
        pvalue_max_abs_diff=metrics["pvalue"],
        streaming_result_path=streaming_path,
        reference_result_path=str(reference_path) if reference_path else None,
        error=error,
    )


def compare_pertpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
    backend: str,
) -> DifferentialComparisonSummary:
    """Compare streaming GLM-based tests against a Pertpy backend."""

    stream_result = wald_test(
        path=dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        output_dir=output_dir,
        data_name=f"benchmark_{backend}_wald",
    )
    streaming_frame = _streaming_de_to_frame(stream_result)
    streaming_path = None
    if not streaming_frame.empty:
        any_result = next(iter(stream_result.values()))
        streaming_path = str(getattr(any_result, "result_path", "")) or None

    reference_df, reference_path, error = _run_pertpy_de(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        backend=backend,
        output_dir=output_dir,
        data_name="benchmark_pertpy_reference",
    )

    metrics = {"effect_size": None, "statistic": None, "pvalue": None}
    if reference_df is not None:
        metrics = _compare_de_frames(streaming_frame, _standardise_de_dataframe(reference_df))

    return DifferentialComparisonSummary(
        test_type="glm",
        reference_tool=f"pertpy_{backend}",
        effect_max_abs_diff=metrics["effect_size"],
        statistic_max_abs_diff=metrics["statistic"],
        pvalue_max_abs_diff=metrics["pvalue"],
        streaming_result_path=streaming_path,
        reference_result_path=str(reference_path) if reference_path else None,
        error=error,
    )

def _summarise_quality_control(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    total_cells = context.get("dataset_cells", 0)
    total_genes = context.get("dataset_genes", 0)
    kept_cells = int(getattr(result, "cell_mask").sum())
    kept_genes = int(getattr(result, "gene_mask").sum())
    return {
        "total_cells": total_cells,
        "kept_cells": kept_cells,
        "cells_removed": max(total_cells - kept_cells, 0),
        "total_genes": total_genes,
        "kept_genes": kept_genes,
        "genes_removed": max(total_genes - kept_genes, 0),
        "output_path": str(getattr(result, "filtered_path")),
    }


def _summarise_dataframe(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(result, "shape"):
        rows, cols = result.shape
    else:
        rows = cols = 0
    return {"rows": int(rows), "columns": int(cols)}


def _summarise_de_mapping(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    groups: list[str] = []
    if isinstance(result, Mapping):
        groups = list(result.keys())
    elif hasattr(result, "groups"):
        groups = list(getattr(result, "groups"))

    n_groups = len(groups)
    n_genes = 0
    output_path = getattr(result, "result_path", None)
    if groups:
        first_key = groups[0]
        try:
            first = result[first_key]
        except Exception:  # pragma: no cover - defensive
            first = None
        if first is not None:
            genes = getattr(first, "genes", None)
            if genes is not None:
                n_genes = int(len(genes))
            if output_path is None:
                output_path = getattr(first, "result_path", None)
    summary = {
        "groups": n_groups,
        "genes": n_genes,
    }
    if output_path:
        summary["output_path"] = str(output_path)
    return summary


def _summarise_scanpy_comparison(result: ComparisonResult, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise quality control and preprocessing comparisons with Scanpy."""

    summary = {
        "comparison_category": "quality_control_preprocessing",
        "reference_tool": "scanpy",
        "normalization_max_abs_diff": result.normalization_max_abs_diff,
        "log1p_max_abs_diff": result.log1p_max_abs_diff,
        "streamlined_cell_count": result.streamlined_cell_count,
        "reference_cell_count": result.reference_cell_count,
        "streamlined_gene_count": result.streamlined_gene_count,
        "reference_gene_count": result.reference_gene_count,
        "avg_log_effect_max_abs_diff": result.avg_log_effect_max_abs_diff,
        "pseudobulk_effect_max_abs_diff": result.pseudobulk_effect_max_abs_diff,
        "streamlined_peak_memory_mb": result.streamlined_peak_memory_mb,
        "reference_peak_memory_mb": result.reference_peak_memory_mb,
        "stream_timings": json.dumps(result.streamlined_timings),
        "reference_timings": json.dumps(result.reference_timings),
    }
    return summary


def _summarise_de_comparison(
    result: DifferentialComparisonSummary, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarise differential expression comparisons."""

    summary = {
        "comparison_category": "differential_expression",
        "test_type": result.test_type,
        "reference_tool": result.reference_tool,
        "effect_max_abs_diff": result.effect_max_abs_diff,
        "statistic_max_abs_diff": result.statistic_max_abs_diff,
        "pvalue_max_abs_diff": result.pvalue_max_abs_diff,
        "streaming_result_path": result.streaming_result_path,
        "reference_result_path": result.reference_result_path,
    }
    if result.error:
        summary["error"] = result.error
    return summary


def _worker(
    queue: mp.Queue,
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
) -> None:
    """Execute ``method`` with optional resource limits and report the outcome."""

    if memory_limit and memory_limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    if time_limit and time_limit > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit))

    start = time.perf_counter()
    try:
        result = method.function(**method.kwargs)
        elapsed = time.perf_counter() - start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss_kb = usage.ru_maxrss
        summary = method.summary(result, context)
        queue.put(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "max_rss_kb": max_rss_kb,
                "summary": summary,
            }
        )
    except MemoryError as exc:
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "memory_limit",
                "elapsed_seconds": elapsed,
                "max_rss_kb": None,
                "error": f"MemoryError: {exc}",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "error",
                "elapsed_seconds": elapsed,
                "max_rss_kb": None,
                "error": f"{exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_with_limits(
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
) -> Dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_worker,
        args=(queue, method, context, memory_limit, time_limit),
        name=f"benchmark-{method.name}",
    )
    process.start()
    join_timeout = None
    if time_limit and time_limit > 0:
        join_timeout = time_limit + 5
    process.join(timeout=join_timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "status": "timeout",
            "elapsed_seconds": None,
            "max_memory_mb": None,
            "summary": {},
            "error": f"Exceeded time limit of {time_limit} seconds",
        }

    if not queue.empty():
        payload = queue.get()
    else:
        payload = {
            "status": "error",
            "elapsed_seconds": None,
            "max_rss_kb": None,
            "error": f"Process exited with code {process.exitcode}",
        }

    payload.setdefault("summary", {})
    max_rss_kb = payload.pop("max_rss_kb", None)
    if max_rss_kb is not None:
        payload["max_memory_mb"] = max_rss_kb / 1024
    else:
        payload.setdefault("max_memory_mb", None)
    return payload


def _load_dataset_context(path: Path) -> Dict[str, Any]:
    backed = read_backed(path)
    try:
        context = {
            "dataset_cells": backed.n_obs,
            "dataset_genes": backed.n_vars,
        }
    finally:
        backed.file.close()
    return context


def build_methods(dataset_path: Path, output_dir: Path) -> Dict[str, BenchmarkMethod]:
    """Return the available benchmark methods for the provided dataset."""

    min_genes = 5
    min_cells_per_perturbation = 5
    min_cells_per_gene = 5

    shared_kwargs = {
        "perturbation_column": "perturbation",
        "control_label": "ctrl",
        "gene_name_column": "gene_symbols",
    }

    methods = {
        "quality_control": BenchmarkMethod(
            name="quality_control",
            description="Streaming quality control filters",
            function=quality_control_summary,
            kwargs={
                "path": dataset_path,
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark",
            },
            summary=_summarise_quality_control,
        ),
        "average_log_expression": BenchmarkMethod(
            name="average_log_expression",
            description="Average log-normalised expression per perturbation",
            function=compute_average_log_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_avg_log",
            },
            summary=_summarise_dataframe,
        ),
        "pseudobulk_expression": BenchmarkMethod(
            name="pseudobulk_expression",
            description="Pseudo-bulk log fold-change per perturbation",
            function=compute_pseudobulk_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_pseudobulk",
            },
            summary=_summarise_dataframe,
        ),
        "wald_test": BenchmarkMethod(
            name="wald_test",
            description="Wald differential expression test",
            function=wald_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_wald",
            },
            summary=_summarise_de_mapping,
        ),
        "wilcoxon_test": BenchmarkMethod(
            name="wilcoxon_test",
            description="Wilcoxon rank-sum differential expression",
            function=wilcoxon_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_wilcoxon",
            },
            summary=_summarise_de_mapping,
        ),
        "scanpy_quality_control_comparison": BenchmarkMethod(
            name="scanpy_quality_control_comparison",
            description="Quality control comparison against Scanpy",
            function=compare_scanpy_quality_control,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
            },
            summary=_summarise_scanpy_comparison,
        ),
        "scanpy_wald_comparison": BenchmarkMethod(
            name="scanpy_wald_comparison",
            description="Wald/t-test comparison against Scanpy",
            function=compare_scanpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "test_type": "wald",
            },
            summary=_summarise_de_comparison,
        ),
        "scanpy_wilcoxon_comparison": BenchmarkMethod(
            name="scanpy_wilcoxon_comparison",
            description="Wilcoxon comparison against Scanpy",
            function=compare_scanpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "test_type": "wilcoxon",
            },
            summary=_summarise_de_comparison,
        ),
        "pertpy_edger_comparison": BenchmarkMethod(
            name="pertpy_edger_comparison",
            description="GLM comparison against edgeR via Pertpy",
            function=compare_pertpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "backend": "edger",
            },
            summary=_summarise_de_comparison,
        ),
        "pertpy_pydeseq2_comparison": BenchmarkMethod(
            name="pertpy_pydeseq2_comparison",
            description="GLM comparison against PyDESeq2 via Pertpy",
            function=compare_pertpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "backend": "pydeseq2",
            },
            summary=_summarise_de_comparison,
        ),
        "pertpy_statsmodels_comparison": BenchmarkMethod(
            name="pertpy_statsmodels_comparison",
            description="GLM comparison against statsmodels via Pertpy",
            function=compare_pertpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "backend": "statsmodels",
            },
            summary=_summarise_de_comparison,
        ),
    }
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamlined CRISPR analysis methods")
    default_output = Path(__file__).resolve().parent / "results"
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "demo_benchmark.h5ad",
        help="Path to an AnnData .h5ad file to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to store benchmark outputs and summaries",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of methods to run. Defaults to all available methods.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Maximum number of CPU seconds allowed per method (0 disables the limit)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Maximum memory per method in gigabytes (0 disables the limit)",
    )
    return parser.parse_args()


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub-friendly Markdown table without extra dependencies."""

    if df.empty:
        return "| |\n|---|\n"

    table = df.copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        table[numeric_cols] = table[numeric_cols].round(3)

    headers = list(table.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = []
    for _, row in table.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


def main() -> None:
    args = parse_args()
    dataset_path = args.data_path
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_path}' was not found. "
            "Generate it with 'python benchmarking/generate_demo_dataset.py' "
            "or supply --data-path to an existing .h5ad file."
        )

    context = _load_dataset_context(dataset_path)
    context["dataset_path"] = str(dataset_path)

    available_methods = build_methods(dataset_path, output_dir)
    selected_names = args.methods or sorted(available_methods)

    rows = []
    memory_limit_bytes = None
    if args.memory_limit and args.memory_limit > 0:
        memory_limit_bytes = int(args.memory_limit * (1024**3))

    for name in selected_names:
        if name not in available_methods:
            raise ValueError(f"Unknown method '{name}'. Available methods: {sorted(available_methods)}")
        method = available_methods[name]
        result = _run_with_limits(method, context, memory_limit_bytes, args.time_limit)
        row = {
            "method": method.name,
            "description": method.description,
            "status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "max_memory_mb": result.get("max_memory_mb"),
        }
        summary = result.get("summary", {})
        if summary:
            row.update(summary)
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_dataframe_to_markdown(df))

    print(f"Benchmark complete. Saved results to {csv_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
